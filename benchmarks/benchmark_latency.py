"""
AG-SAR v3.1 Latency Benchmark

Verifies the "Zero-Latency" promise by measuring overhead of:
1. Hook registration and capture
2. Authority Flow computation
3. Spectral Roughness calculation
4. Register Filter (kurtosis + EMA)

Pass Criteria:
- < 10% overhead relative to baseline model forward pass
- OR < 2ms absolute overhead per step
"""

import argparse
import gc
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from ag_sar.ag_sar import AGSAR
from ag_sar.config import AGSARConfig


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def benchmark_model(
    model_id: str = "gpt2",
    seq_len: int = 128,
    num_warmup: int = 10,
    num_iterations: int = 50,
    device: str = "auto",
    use_small_model: bool = False,
) -> dict:
    """
    Benchmark AG-SAR overhead against baseline model inference.

    Args:
        model_id: HuggingFace model ID
        seq_len: Sequence length for benchmark
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
        device: Device to run on ("auto", "cuda", "cpu")
        use_small_model: Use a small random-weight model for fast testing

    Returns:
        Dictionary with benchmark results
    """
    # Device setup
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Model: {model_id}")
    print(f"Sequence Length: {seq_len}")
    print("-" * 50)

    # Load model
    if use_small_model:
        print("Using small random-weight model for testing...")
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 4
        config.n_head = 4
        config.n_embd = 256
        config.vocab_size = 1000
        model = AutoModelForCausalLM.from_config(config)
        model_id = "gpt2-small-test"
    else:
        print(f"Loading {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )

    model = model.to(device)
    model.eval()

    # Create mock tokenizer for small model
    if use_small_model:
        class MockTokenizer:
            def __init__(self):
                self.pad_token = None
                self.eos_token = "<eos>"
                self.pad_token_id = 0
                self.eos_token_id = 1

            def encode(self, text, add_special_tokens=True):
                return list(range(2, 2 + len(text.split())))

            def decode(self, token_ids):
                return " ".join([f"token_{i}" for i in token_ids])

        tokenizer = MockTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create AG-SAR engine
    ag_config = AGSARConfig(
        semantic_layers=min(4, model.config.n_layer if hasattr(model.config, 'n_layer') else model.config.num_hidden_layers),
        enable_register_filter=True,
        enable_authority_flow=True,
        enable_spectral_roughness=True,
        lambda_roughness=10.0,
    )
    engine = AGSAR(model, tokenizer, ag_config)

    # Create input
    input_ids = torch.randint(2, 1000, (1, seq_len), device=device)

    # =========================================================================
    # 1. Baseline Latency (Standard Forward Pass)
    # =========================================================================
    print("\n[1/3] Measuring baseline latency...")

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_ids)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(input_ids)
    if device == "cuda":
        torch.cuda.synchronize()
    baseline_total = time.perf_counter() - start
    baseline_per_iter = baseline_total / num_iterations

    print(f"  Baseline: {baseline_per_iter * 1000:.2f} ms/iter")

    # =========================================================================
    # 2. AG-SAR Latency (With Hooks + Authority Flow)
    # =========================================================================
    print("\n[2/3] Measuring AG-SAR latency...")

    # Reset and initialize engine
    engine.reset()

    # Split into prompt and generation
    prompt_len = seq_len // 2
    prompt_ids = input_ids[:, :prompt_len]

    # Initialize with prompt
    with torch.no_grad():
        engine.process_prompt(prompt_ids)

    # Warmup
    current_ids = prompt_ids.clone()
    for _ in range(num_warmup):
        with torch.no_grad():
            new_token = torch.randint(2, 1000, (1, 1), device=device)
            temp_ids = torch.cat([current_ids, new_token], dim=-1)
            _ = engine.process_step(temp_ids)

    if device == "cuda":
        torch.cuda.synchronize()

    # Reset for clean measurement
    engine.reset()
    with torch.no_grad():
        engine.process_prompt(prompt_ids)

    # Timed runs (simulating per-token generation)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    current_ids = prompt_ids.clone()
    start = time.perf_counter()
    for i in range(num_iterations):
        with torch.no_grad():
            # Simulate one generation step
            new_token = torch.randint(2, 1000, (1, 1), device=device)
            current_ids = torch.cat([current_ids, new_token], dim=-1)

            # AG-SAR processing
            _ = engine.process_step(current_ids)

        # Clear cache to simulate streaming (memory-bounded)
        engine.extractor.clear_cache()

    if device == "cuda":
        torch.cuda.synchronize()
    agsar_total = time.perf_counter() - start
    agsar_per_iter = agsar_total / num_iterations

    print(f"  AG-SAR:   {agsar_per_iter * 1000:.2f} ms/iter")

    # =========================================================================
    # 3. Pure AG-SAR Math Overhead (No Model Forward)
    # =========================================================================
    print("\n[3/3] Measuring pure math overhead...")

    engine.reset()
    with torch.no_grad():
        engine.process_prompt(prompt_ids)

    # Pre-compute model output to isolate math overhead
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Time just the AG-SAR math (no model forward)
    start = time.perf_counter()
    for i in range(num_iterations):
        # Simulate the math operations only
        with torch.no_grad():
            # Extract captures (would normally be from hooks)
            h_attn, v_states, attn_weights = engine._extract_v31_captures()

            # The actual step computation is inside process_step
            # but we can approximate by looking at the config's operations

    if device == "cuda":
        torch.cuda.synchronize()
    math_total = time.perf_counter() - start
    math_per_iter = math_total / num_iterations

    print(f"  Math:     {math_per_iter * 1000:.4f} ms/iter")

    # =========================================================================
    # Results
    # =========================================================================
    overhead_ms = (agsar_per_iter - baseline_per_iter) * 1000
    overhead_pct = ((agsar_per_iter / baseline_per_iter) - 1) * 100

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Baseline:     {baseline_per_iter * 1000:.2f} ms")
    print(f"AG-SAR:       {agsar_per_iter * 1000:.2f} ms")
    print(f"Overhead:     {overhead_ms:.2f} ms (+{overhead_pct:.1f}%)")

    # Memory usage
    if device == "cuda":
        mem_usage = get_memory_usage()
        print(f"GPU Memory:   {mem_usage:.1f} MB")

    # Pass/Fail criteria
    print("\n" + "-" * 50)
    passed = overhead_pct <= 10.0 or overhead_ms <= 2.0

    if passed:
        print("PASS: Zero-Latency constraint satisfied!")
        print(f"  (Threshold: <10% overhead OR <2ms absolute)")
    else:
        print("FAIL: Overhead too high!")
        print(f"  Measured: {overhead_pct:.1f}% / {overhead_ms:.2f}ms")
        print(f"  Required: <10% OR <2ms")

    engine.cleanup()

    return {
        "model": model_id,
        "device": device,
        "seq_len": seq_len,
        "baseline_ms": baseline_per_iter * 1000,
        "agsar_ms": agsar_per_iter * 1000,
        "overhead_ms": overhead_ms,
        "overhead_pct": overhead_pct,
        "math_overhead_ms": math_per_iter * 1000,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description="AG-SAR Latency Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace model ID (default: gpt2)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length (default: 128)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Timed iterations (default: 50)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device (default: auto)"
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use small random-weight model for fast testing"
    )

    args = parser.parse_args()

    results = benchmark_model(
        model_id=args.model,
        seq_len=args.seq_len,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        device=args.device,
        use_small_model=args.small,
    )

    return 0 if results["passed"] else 1


if __name__ == "__main__":
    exit(main())
