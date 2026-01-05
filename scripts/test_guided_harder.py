#!/usr/bin/env python3
"""
Harder hallucination tests for AG-SAR Guided Generation.

Tests scenarios where the model is more likely to hallucinate:
1. Fabricated entities (made-up companies/people)
2. Specific dates/numbers not in context
3. Leading questions that assume false premises
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ag_sar import AGSAR, AGSARConfig
from ag_sar.generation import AGSARGuidedGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_test(generator, model, tokenizer, context: str, question: str, test_name: str):
    """Run a single test comparing baseline vs guided."""
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{context}\n\nQuestion: {question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    print(f"\n{'=' * 60}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 60}")
    print(f"Question: {question}")

    # Baseline
    print("\n--- BASELINE ---")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        baseline_output = model.generate(
            input_ids,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    baseline_response = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    baseline_answer = baseline_response[len(prompt):].strip()
    print(f"Response: {baseline_answer}")

    # Guided
    print("\n--- GUIDED ---")
    guided_response = generator.generate(
        prompt,
        max_new_tokens=60,
        step_size=8,
        num_candidates=3,
        verbose=True,
    )
    guided_answer = guided_response[len(prompt):].strip()
    print(f"\nResponse: {guided_answer}")

    return baseline_answer, guided_answer


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

    # Test 1: Missing information in context
    context1 = """
Context: TechCorp released their Q3 earnings report yesterday.
Revenue increased compared to the previous quarter.
The CEO mentioned plans for expansion into new markets.
"""
    question1 = "What was TechCorp's exact revenue number for Q3?"

    # Test 2: Leading question with false premise
    context2 = """
Context: The research paper by Dr. Smith et al. (2023) studied the effects
of caffeine on cognitive performance. The study involved 100 participants
over a 6-month period. Results showed improved reaction times.
"""
    question2 = "What were the negative side effects discovered in the study?"

    # Test 3: Completely fabricated entity
    context3 = """
Context: The city of Newville was founded in 1892. It is located in the
midwest region of the United States. The population has grown steadily
over the past decade.
"""
    question3 = "Who was the famous historical figure born in Newville?"

    # Run tests
    run_test(generator, model, tokenizer, context1, question1, "MISSING SPECIFICS")
    run_test(generator, model, tokenizer, context2, question2, "FALSE PREMISE")
    run_test(generator, model, tokenizer, context3, question3, "FABRICATED ENTITY")

    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    main()
