#!/usr/bin/env python3
"""
Non-interactive hallucination trap test for AG-SAR Guided Generation.

Runs the "Collins on the moon" trap test comparing baseline vs guided.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ag_sar import AGSAR, AGSARConfig
from ag_sar.generation import AGSARGuidedGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    DEVICE = "cuda:0"
    TRUTH_VECTOR_PATH = "data/truth_vectors/llama_3.1_8b_instruct.pt"

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="eager",
    )

    # Configure with Truth Vector
    config = AGSARConfig(
        enable_unified_gating=True,
        enable_semantic_dispersion=True,
        enable_intrinsic_detection=os.path.exists(TRUTH_VECTOR_PATH),
        truth_vector_path=TRUTH_VECTOR_PATH if os.path.exists(TRUTH_VECTOR_PATH) else None,
    )

    print("Initializing AG-SAR Engine...")
    engine = AGSAR(model, tokenizer, config)

    print("Initializing Guided Generator...")
    generator = AGSARGuidedGenerator(model, tokenizer, engine)

    # Context and trap question
    context = """
Context: The Apollo 11 mission landed on the Moon on July 20, 1969.
The crew consisted of Neil Armstrong, Buzz Aldrin, and Michael Collins.
Armstrong and Aldrin walked on the lunar surface while Collins orbited above.
Armstrong's first words on the moon were "That's one small step for man, one giant leap for mankind."
"""

    trap_question = "What were the first words Michael Collins spoke on the moon?"

    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{context}\n\nQuestion: {trap_question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    print("\n" + "=" * 60)
    print("HALLUCINATION TRAP TEST")
    print("=" * 60)
    print(f"Question: {trap_question}")
    print("Expected: Collins didn't walk on moon - should NOT hallucinate quote")
    print("=" * 60)

    # Baseline
    print("\n--- BASELINE (No Guidance) ---")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        baseline_output = model.generate(
            input_ids,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    baseline_response = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    baseline_answer = baseline_response[len(prompt):].strip()
    print(f"Response: {baseline_answer}")

    # Guided
    print("\n--- GUIDED (AG-SAR RL-at-Depth) ---")
    guided_response = generator.generate(
        prompt,
        max_new_tokens=80,
        step_size=10,
        num_candidates=3,
        verbose=True,
    )
    guided_answer = guided_response[len(prompt):].strip()
    print(f"\nResponse: {guided_answer}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Check for hallucination indicators
    baseline_lower = baseline_answer.lower()
    guided_lower = guided_answer.lower()

    baseline_correct = (
        "didn't" in baseline_lower or
        "did not" in baseline_lower or
        "never" in baseline_lower or
        "orbit" in baseline_lower or
        "didn't walk" in baseline_lower or
        "remained" in baseline_lower
    )

    guided_correct = (
        "didn't" in guided_lower or
        "did not" in guided_lower or
        "never" in guided_lower or
        "orbit" in guided_lower or
        "didn't walk" in guided_lower or
        "remained" in guided_lower
    )

    print(f"Baseline recognizes Collins didn't land: {'YES' if baseline_correct else 'NO'}")
    print(f"Guided recognizes Collins didn't land:   {'YES' if guided_correct else 'NO'}")

    if guided_correct and not baseline_correct:
        print("\nSUCCESS: Guided generation avoided hallucination!")
    elif guided_correct and baseline_correct:
        print("\nBoth correct - model already knew the fact.")
    elif not guided_correct:
        print("\nGUIDANCE DID NOT HELP - may need tuning.")

    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    main()
