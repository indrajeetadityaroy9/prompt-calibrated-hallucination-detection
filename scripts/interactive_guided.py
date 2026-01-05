#!/usr/bin/env python3
"""
Interactive AG-SAR Guided Generation CLI.

This script allows real-time interaction with the RL-at-Depth controller
to stress-test hallucination prevention.

Usage:
    python scripts/interactive_guided.py
    python scripts/interactive_guided.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/interactive_guided.py --step-size 10 --candidates 5

Commands:
    - Type a question to generate a response
    - 'context' - View current context
    - 'set context' - Set new context
    - 'compare' - Run baseline vs guided comparison
    - 'trap' - Run the Collins hallucination trap test
    - 'settings' - View current settings
    - 'exit' - Quit

The "Trap" Test:
    Ask: "What were the first words Michael Collins spoke on the moon?"
    Expected: Model might try to hallucinate a quote.
    AG-SAR should detect the conflict (Collins didn't land) and select
    a path that correctly states he didn't walk on the moon.
"""

import argparse
import os
import sys
import torch

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ag_sar import AGSAR, AGSARConfig
from ag_sar.generation import AGSARGuidedGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_engine(model_id: str, device: str, truth_vector_path: str):
    """Load model and initialize AG-SAR engine."""
    print(f"Loading {model_id} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",  # Required for AG-SAR hooks
    )

    # Configure AG-SAR with Universal mode if Truth Vector exists
    config_kwargs = {
        "enable_unified_gating": True,
        "enable_semantic_dispersion": True,
        "dispersion_method": "top1_projection",
    }

    if os.path.exists(truth_vector_path):
        print(f"Loading Truth Vector from {truth_vector_path}")
        config_kwargs["enable_intrinsic_detection"] = True
        config_kwargs["truth_vector_path"] = truth_vector_path
    else:
        print(f"Warning: Truth Vector not found at {truth_vector_path}")
        print("   Running in Extrinsic-Only (JEPA) mode.")
        config_kwargs["enable_intrinsic_detection"] = False

    config = AGSARConfig(**config_kwargs)

    print("Initializing Universal AG-SAR Engine...")
    engine = AGSAR(model, tokenizer, config)

    return model, tokenizer, engine


def baseline_generate(model, tokenizer, prompt: str, max_tokens: int = 100) -> str:
    """Standard generation without AG-SAR guidance."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_comparison(generator, model, tokenizer, context: str, question: str):
    """Run side-by-side comparison of baseline vs guided generation."""
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{context}\n\nQuestion: {question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    print("\n" + "=" * 60)
    print("BASELINE (No Guidance)")
    print("=" * 60)
    baseline_response = baseline_generate(model, tokenizer, prompt, max_tokens=100)
    baseline_answer = baseline_response[len(prompt):].strip()
    print(f"Response: {baseline_answer}")

    print("\n" + "=" * 60)
    print("GUIDED (AG-SAR RL-at-Depth)")
    print("=" * 60)
    guided_response = generator.generate(prompt, max_new_tokens=100, verbose=True)
    guided_answer = guided_response[len(prompt):].strip()
    print(f"\nResponse: {guided_answer}")

    return baseline_answer, guided_answer


def run_trap_test(generator, model, tokenizer, context: str):
    """
    Run the hallucination trap test.

    The trap: "What were the first words Michael Collins spoke on the moon?"
    Collins didn't walk on the moon - he orbited above.
    """
    trap_question = "What were the first words Michael Collins spoke on the moon?"

    print("\n" + "=" * 60)
    print("HALLUCINATION TRAP TEST")
    print("=" * 60)
    print(f"Context: Apollo 11 - Armstrong, Aldrin walked; Collins orbited")
    print(f"Question: {trap_question}")
    print(f"\nExpected behavior: Model should NOT hallucinate a quote.")
    print("It should recognize Collins didn't walk on the moon.")
    print("=" * 60)

    baseline_answer, guided_answer = run_comparison(
        generator, model, tokenizer, context, trap_question
    )

    # Simple heuristic check
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    collins_walked_baseline = "spoke" in baseline_answer.lower() and "moon" in baseline_answer.lower()
    collins_walked_guided = "spoke" in guided_answer.lower() and "moon" in guided_answer.lower()

    didnt_walk_baseline = "didn't" in baseline_answer.lower() or "did not" in baseline_answer.lower() or "orbit" in baseline_answer.lower()
    didnt_walk_guided = "didn't" in guided_answer.lower() or "did not" in guided_answer.lower() or "orbit" in guided_answer.lower()

    print(f"Baseline hallucinated quote: {'YES (BAD)' if collins_walked_baseline and not didnt_walk_baseline else 'NO (GOOD)'}")
    print(f"Guided hallucinated quote:   {'YES (BAD)' if collins_walked_guided and not didnt_walk_guided else 'NO (GOOD)'}")


def main():
    parser = argparse.ArgumentParser(description="Interactive AG-SAR Guided Generation")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to run on"
    )
    parser.add_argument(
        "--truth-vector",
        default="data/truth_vectors/llama_3.1_8b_instruct.pt",
        help="Path to Truth Vector file"
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=10,
        help="Tokens per evaluation step"
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=3,
        help="Number of candidate paths"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    args = parser.parse_args()

    # Load model and engine
    model, tokenizer, engine = load_model_and_engine(
        args.model, args.device, args.truth_vector
    )

    # Initialize guided generator
    print("Initializing Guided Generator (RL-at-Depth)...")
    generator = AGSARGuidedGenerator(model, tokenizer, engine)

    # Default context (factually strict for testing)
    context = """
Context: The Apollo 11 mission landed on the Moon on July 20, 1969.
The crew consisted of Neil Armstrong, Buzz Aldrin, and Michael Collins.
Armstrong and Aldrin walked on the lunar surface while Collins orbited above.
Armstrong's first words on the moon were "That's one small step for man, one giant leap for mankind."
"""

    print(f"\n{'=' * 60}")
    print("AG-SAR Guided Generation - Interactive CLI")
    print(f"{'=' * 60}")
    print(f"Model: {args.model}")
    print(f"Step Size: {args.step_size} tokens")
    print(f"Candidates: {args.candidates}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Truth Vector: {'ENABLED' if engine.config.enable_intrinsic_detection else 'DISABLED'}")
    print(f"\nDefault Context:\n{context}")
    print(f"\nCommands: 'exit', 'context', 'set context', 'compare', 'trap', 'settings'")
    print(f"{'=' * 60}\n")

    while True:
        try:
            user_input = input("USER >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if user_input.lower() == "context":
            print(f"\nCurrent Context:\n{context}\n")
            continue

        if user_input.lower() == "set context":
            print("Enter new context (end with empty line):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            context = "\n".join(lines)
            print(f"Context updated.\n")
            continue

        if user_input.lower() == "settings":
            print(f"\nSettings:")
            print(f"  Model: {args.model}")
            print(f"  Step Size: {args.step_size}")
            print(f"  Candidates: {args.candidates}")
            print(f"  Max Tokens: {args.max_tokens}")
            print(f"  Truth Vector: {'ENABLED' if engine.config.enable_intrinsic_detection else 'DISABLED'}")
            print()
            continue

        if user_input.lower() == "compare":
            question = input("Enter question for comparison: ").strip()
            if question:
                run_comparison(generator, model, tokenizer, context, question)
            continue

        if user_input.lower() == "trap":
            run_trap_test(generator, model, tokenizer, context)
            continue

        # Regular question - run guided generation
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{context}\n\nQuestion: {user_input}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )

        print("\n... Thinking (Stepwise Search) ...")
        response = generator.generate(
            prompt,
            max_new_tokens=args.max_tokens,
            step_size=args.step_size,
            num_candidates=args.candidates,
            verbose=True,
        )

        # Extract just the assistant's reply
        answer = response[len(prompt):].strip()
        print(f"\nASSISTANT >> {answer}\n")

    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    main()
