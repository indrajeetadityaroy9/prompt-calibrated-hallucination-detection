"""
AG-SAR Guided Decoding: RL-at-Depth Controller.

Implements Stepwise Best-of-N Search guided by Universal AG-SAR Trust.

Key Mechanism:
    1. Expand: Generate N candidate phrases in parallel
    2. Evaluate: Score each candidate using Universal AG-SAR (JEPA + Truth Vector)
    3. Select: Commit to the safest candidate (Highest Trust)
    4. Repeat until max_tokens or EOS

This transforms the Universal Sensor (passive detection) into an Active Controller
that "thinks" (evaluates trust) before it "speaks" (commits to tokens).

Physical Interpretation:
    - Each step is a "System 2" deliberation checkpoint
    - The model explores multiple futures and prunes hallucination-prone paths
    - Trust score acts as the Reward Function in a local search

Example:
    >>> generator = AGSARGuidedGenerator(model, tokenizer, engine)
    >>> response = generator.generate(prompt, step_size=15, num_candidates=3)
"""

import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class GenerationStep:
    """Record of a single generation step for analysis."""
    step_idx: int
    candidates: List[str]
    scores: List[float]
    selected_idx: int
    selected_text: str
    selected_trust: float


class AGSARGuidedGenerator:
    """
    RL-at-Depth: Stepwise Best-of-N Generation guided by Universal AG-SAR.

    Attributes:
        model: HuggingFace language model
        tokenizer: Corresponding tokenizer
        engine: Initialized AG-SAR engine (with or without Truth Vector)
        device: Compute device

    Example:
        >>> from ag_sar import AGSAR, AGSARConfig
        >>> config = AGSARConfig(enable_intrinsic_detection=True,
        ...                      truth_vector_path="data/truth_vectors/llama.pt")
        >>> engine = AGSAR(model, tokenizer, config)
        >>> generator = AGSARGuidedGenerator(model, tokenizer, engine)
        >>> response = generator.generate("What is 2+2?", step_size=10, num_candidates=3)
    """

    def __init__(self, model, tokenizer, ag_sar_engine):
        """
        Initialize the guided generator.

        Args:
            model: HuggingFace CausalLM model
            tokenizer: Corresponding tokenizer
            ag_sar_engine: Initialized AGSAR engine instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.engine = ag_sar_engine
        self.device = next(model.parameters()).device

        # Ensure pad token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        step_size: int = 15,
        num_candidates: int = 3,
        temperature: float = 0.7,
        trust_threshold: float = 0.3,
        hysteresis_alpha: float = 0.05,
        outlier_sigma: float = 1.0,
        verbose: bool = True,
        return_trace: bool = False,
    ) -> str:
        """
        Generate text using stepwise Best-of-N search guided by AG-SAR trust.

        Mechanism:
            1. Expand: Generate N candidate phrases in parallel
            2. Evaluate: Score each using Universal AG-SAR (Gate-blended JEPA + Truth)
            3. Select: Commit to highest-trust candidate (with adaptive thresholds)
            4. Repeat until max_tokens or EOS

        v12.3 Relaxed Adaptive Selection:
            Adaptive Hysteresis: Threshold = α × (1 - baseline_score)
            - High trust (0.90): requires gain of 0.005 (very sensitive)
            - Low trust (0.10): requires gain of 0.045 (robust)

            Note: Outlier check DISABLED for small batches (N < 10) to avoid
            deadlock where dual constraints prevent any interventions.

        Args:
            prompt: Input prompt (can include system context)
            max_new_tokens: Maximum tokens to generate
            step_size: Tokens per evaluation step (default 15 = ~1 phrase)
            num_candidates: Number of parallel paths to consider (default 3)
            temperature: Sampling temperature for diversity (default 0.7)
            trust_threshold: Minimum trust to continue (default 0.3)
            hysteresis_alpha: Fraction of uncertainty gap required to switch (default 0.05)
            outlier_sigma: Reserved for large batches (N >= 10), currently unused
            verbose: Print step-by-step decisions
            return_trace: Return generation trace for analysis

        Returns:
            Generated text (full prompt + response)
            If return_trace=True, returns (text, List[GenerationStep])
        """
        if verbose:
            print(f"Starting Guided Generation (Step={step_size}, N={num_candidates})...")

        # Encode initial prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_length = input_ids.shape[1]
        current_text = prompt

        # Track generation trace
        trace: List[GenerationStep] = []

        # Generation loop
        step_idx = 0
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # 1. Expand: Generate N candidates in parallel
            batch_input_ids = input_ids.repeat(num_candidates, 1)

            with torch.no_grad():
                outputs = self.model.generate(
                    batch_input_ids,
                    max_new_tokens=step_size,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Extract new tokens for each candidate
            new_tokens_batch = outputs[:, input_ids.shape[1]:]

            # Decode to text
            candidates_text = self.tokenizer.batch_decode(
                new_tokens_batch, skip_special_tokens=True
            )

            # 2. Evaluate: Score each candidate with AG-SAR
            scores = []
            valid_indices = []

            if verbose:
                print(f"\n--- Step {step_idx + 1} Evaluation ---")

            for i, cand_text in enumerate(candidates_text):
                # Handle empty generation (EOS reached)
                if not cand_text.strip():
                    scores.append(-1.0)
                    continue

                # Score the response portion (everything after prompt)
                # current_text already contains prompt + previous generations
                response_so_far = current_text[len(prompt):] + cand_text

                try:
                    # Run Universal AG-SAR Engine
                    result = self.engine.compute_uncertainty(
                        prompt=prompt,
                        response=response_so_far,
                        return_details=True
                    )

                    # Convert uncertainty to trust (higher is better)
                    uncertainty = result.get("score", result.get("uncertainty", 0.5))
                    trust = 1.0 - uncertainty
                    scores.append(trust)
                    valid_indices.append(i)

                except Exception as e:
                    if verbose:
                        print(f"  Cand {i}: Error scoring - {e}")
                    scores.append(-1.0)
                    continue

                # Log decision process
                if verbose:
                    clean_preview = cand_text.replace('\n', ' ')[:50]
                    print(f"  Cand {i}: '{clean_preview}...' -> Trust: {trust:.4f}")

            # 3. Select: Pick best path (with adaptive thresholds)
            if not valid_indices or max(scores) <= 0:
                if verbose:
                    print("  No valid candidates. Stopping generation.")
                break

            # v12.3 Relaxed Adaptive Selection:
            # - Adaptive Hysteresis: Require α × (1 - baseline) gain
            # - Outlier check DISABLED for small batches (N < 10) to avoid deadlock
            baseline_score = scores[0] if scores[0] > 0 else 0.0
            best_idx = 0  # Default to baseline
            current_best_score = baseline_score

            # Compute batch statistics for logging only
            valid_scores = [s for s in scores if s > 0]
            if len(valid_scores) >= 2:
                batch_mean = sum(valid_scores) / len(valid_scores)
                batch_var = sum((s - batch_mean) ** 2 for s in valid_scores) / len(valid_scores)
                batch_std = batch_var ** 0.5 + 1e-6
            else:
                batch_mean = baseline_score
                batch_std = 1.0

            # Adaptive hysteresis: threshold scales with "room for improvement"
            # α=0.05: If baseline=0.5 (unsure), need +0.025. If baseline=0.9, need +0.005.
            adaptive_threshold = hysteresis_alpha * (1.0 - baseline_score)

            # Relaxed selection: only check adaptive threshold, no outlier constraint
            # (Outlier check causes deadlock with N=3 candidates)
            for i in range(1, len(scores)):
                if scores[i] <= 0:
                    continue

                # Single condition: beat baseline by adaptive threshold
                if scores[i] > baseline_score + adaptive_threshold:
                    # Greedy max among candidates that pass threshold
                    if scores[i] > current_best_score:
                        best_idx = i
                        current_best_score = scores[i]

            # If no candidate passed threshold, use baseline if valid
            if best_idx == 0 and scores[0] <= 0:
                best_idx = max(range(len(scores)), key=lambda i: scores[i])

            best_score = scores[best_idx]
            best_text = candidates_text[best_idx]
            best_tokens = new_tokens_batch[best_idx].unsqueeze(0)

            if verbose:
                if best_idx != 0 and scores[0] > 0:
                    gain = best_score - scores[0]
                    print(f"  --> SWITCH Cand {best_idx} (Trust: {best_score:.4f}, Gain: +{gain:.4f} > {adaptive_threshold:.4f})")
                else:
                    print(f"  --> KEEP Cand 0 (Trust: {best_score:.4f}, Threshold: {adaptive_threshold:.4f})")

            # Check trust threshold
            if best_score < trust_threshold:
                if verbose:
                    print(f"  Trust {best_score:.4f} below threshold {trust_threshold}. Stopping.")
                break

            # Record trace
            trace.append(GenerationStep(
                step_idx=step_idx,
                candidates=candidates_text,
                scores=scores,
                selected_idx=best_idx,
                selected_text=best_text,
                selected_trust=best_score,
            ))

            # 4. Commit: Append to history
            input_ids = torch.cat([input_ids, best_tokens], dim=-1)
            current_text += best_text
            tokens_generated += best_tokens.shape[1]
            step_idx += 1

            # Check for EOS in best candidate
            if self.tokenizer.eos_token_id in best_tokens:
                if verbose:
                    print("  EOS reached. Generation complete.")
                break

        if verbose:
            print(f"\nGeneration complete. {tokens_generated} tokens in {step_idx} steps.")

        if return_trace:
            return current_text, trace
        return current_text

    def generate_with_context(
        self,
        context: str,
        question: str,
        system_template: str = "llama3",
        **kwargs
    ) -> str:
        """
        Convenience method for RAG-style generation with context.

        Args:
            context: Retrieved context/documents
            question: User question
            system_template: Prompt template style ("llama3", "chatml", "plain")
            **kwargs: Passed to generate()

        Returns:
            Generated response (without prompt template)
        """
        # Construct prompt based on template
        if system_template == "llama3":
            prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
                f"{context}\n\nQuestion: {question}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )
        elif system_template == "chatml":
            prompt = (
                f"<|im_start|>user\n{context}\n\nQuestion: {question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:  # plain
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        # Generate
        full_response = self.generate(prompt, **kwargs)

        # Extract just the assistant's reply
        response = full_response[len(prompt):].strip()
        return response

    def generate_with_stats(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        step_size: int = 15,
        num_candidates: int = 3,
        temperature: float = 0.7,
        trust_threshold: float = 0.3,
        hysteresis_alpha: float = 0.05,
        outlier_sigma: float = 1.0,
        verbose: bool = False,
    ) -> tuple:
        """
        Generate text and return statistics for evaluation.

        This method is designed for benchmarking the "Rejection Rate" -
        how often the controller intervenes (selects non-greedy candidate).

        v12.3 Relaxed Adaptive Selection:
            Adaptive Hysteresis: Threshold = α × (1 - baseline_score)
            - Sensitive at high trust, robust at low trust
            - Outlier check DISABLED for small batches to avoid deadlock

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            step_size: Tokens per evaluation step
            num_candidates: Number of parallel paths
            temperature: Sampling temperature
            trust_threshold: Minimum trust to continue
            hysteresis_alpha: Fraction of uncertainty gap required to switch (default 0.05)
            outlier_sigma: Reserved for large batches (N >= 10), currently unused
            verbose: Print step-by-step decisions

        Returns:
            Tuple of (generated_text, avg_trust, intervention_count, total_steps)
            - generated_text: Full generated text
            - avg_trust: Average trust score across all steps
            - intervention_count: Number of steps where non-greedy candidate was selected
            - total_steps: Total number of generation steps
        """
        # Encode initial prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        current_text = prompt

        # Statistics tracking
        trust_scores = []
        intervention_count = 0
        total_steps = 0
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # 1. Expand: Generate N candidates in parallel
            batch_input_ids = input_ids.repeat(num_candidates, 1)

            with torch.no_grad():
                outputs = self.model.generate(
                    batch_input_ids,
                    max_new_tokens=step_size,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Extract new tokens for each candidate
            new_tokens_batch = outputs[:, input_ids.shape[1]:]

            # Decode to text
            candidates_text = self.tokenizer.batch_decode(
                new_tokens_batch, skip_special_tokens=True
            )

            # 2. Evaluate: Score each candidate with AG-SAR
            scores = []

            for i, cand_text in enumerate(candidates_text):
                # Handle empty generation (EOS reached)
                if not cand_text.strip():
                    scores.append(-1.0)
                    continue

                # Score the response portion
                response_so_far = current_text[len(prompt):] + cand_text

                try:
                    result = self.engine.compute_uncertainty(
                        prompt=prompt,
                        response=response_so_far,
                        return_details=True
                    )
                    uncertainty = result.get("score", result.get("uncertainty", 0.5))
                    trust = 1.0 - uncertainty
                    scores.append(trust)

                except Exception:
                    scores.append(-1.0)

            # 3. Select: Pick best path (with relaxed adaptive threshold)
            valid_scores = [s for s in scores if s > 0]
            if not valid_scores:
                break

            # v12.3 Relaxed Adaptive Selection:
            # - Adaptive Hysteresis only (outlier check disabled for small batches)
            baseline_score = scores[0] if scores[0] > 0 else 0.0
            best_idx = 0  # Default to baseline
            current_best_score = baseline_score

            # Adaptive hysteresis: threshold scales with "room for improvement"
            # α=0.05: If baseline=0.5 (unsure), need +0.025. If baseline=0.9, need +0.005.
            adaptive_threshold = hysteresis_alpha * (1.0 - baseline_score)

            # Relaxed selection: only check adaptive threshold
            for i in range(1, len(scores)):
                if scores[i] <= 0:
                    continue

                # Single condition: beat baseline by adaptive threshold
                if scores[i] > baseline_score + adaptive_threshold:
                    if scores[i] > current_best_score:
                        best_idx = i
                        current_best_score = scores[i]

            # If no candidate passed threshold, use baseline if valid
            if best_idx == 0 and scores[0] <= 0:
                best_idx = max(range(len(scores)), key=lambda i: scores[i])

            best_score = scores[best_idx]
            best_text = candidates_text[best_idx]
            best_tokens = new_tokens_batch[best_idx].unsqueeze(0)

            # Track intervention: Did we select something other than candidate 0?
            if best_idx != 0 and scores[0] > 0:
                intervention_count += 1

            trust_scores.append(best_score)
            total_steps += 1

            if verbose:
                print(f"Step {total_steps}: Selected Cand {best_idx} (Trust: {best_score:.4f})")

            # Check trust threshold
            if best_score < trust_threshold:
                break

            # 4. Commit: Append to history
            input_ids = torch.cat([input_ids, best_tokens], dim=-1)
            current_text += best_text
            tokens_generated += best_tokens.shape[1]

            # Check for EOS
            if self.tokenizer.eos_token_id in best_tokens:
                break

        # Calculate average trust
        avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0

        return current_text, avg_trust, intervention_count, total_steps
