         v_next = torch.bmm(adj_matrix, v)

         # Normalize to prevent overflow
         norm = torch.norm(v_next, dim=1, keepdim=True)
         v_next = v_next / (norm + 1e-10)

         # Check convergence
         if torch.norm(v_next - v) < tol:
             break
         v = v_next

     return v.squeeze(-1)  # (batch, n)

 Critical Equation:
 R(t_i) = C_eigen(t_i) × ||v_i||_2

 Why this works: Attention sinks have massive attention (high centrality) but near-zero value norms. The product naturally filters them out.

 ---
 Module 5: Graph-Shifted Entropy (gse.py)

 Purpose: Compute uncertainty weighted by token relevance

 Functions:
 def compute_token_entropy(logits):
     """
     H(t_i) = -Σ_v p(v|context) * log p(v|context)
     Local entropy for predicting token t_i
     """

 def compute_graph_shifted_entropy(token_entropies, relevance_scores):
     """
     GSE(T) = Σ_i H(t_i) × R̃(t_i

     Down-weights uncertainty from:
     - Leaf nodes (irrelevant tokens)
     - Attention sinks (high attention, low value norm)
     """

 def compute_uncertainty_threshold(gse_scores, method='percentile', q=95):
     """Determine hallucination threshold from GSE distribution"""

 Core Equation:
 GSE(T) = Σ_{i=1}^{N} H(t_i) × (R(t_i) / Σ_{j=1}^{N} R(t_j))

 ---
 Module 6: Spectral Consensus (spectral.py)

 Purpose: Compare reasoning topology across multiple generations

 PERFORMANCE WARNING: Full eigenvalue computation is O(N³). For 2048-token sequences, this is slow.

 Functions:
 def compute_graph_laplacian(adjacency_matrix):
     """L = D - A where D is degree matrix"""

 def compute_spectral_gap(laplacian):
     """
     FAST DEFAULT: Only compute top 2 eigenvalues.
     Spectral gap = λ_1 - λ_2 (difference between largest eigenvalues)
     Use scipy.sparse.linalg.eigsh with k=2 for speed.
     """

 def compute_full_spectrum(laplacian):
     """
     SLOW: Full scipy.linalg.eigvalsh - only run if uncertainty_threshold triggered.
     Do NOT run on every query if latency matters.
     """

 def compute_spectral_divergence(spectrum_a, spectrum_b, method='jsd'):
     """
     Consensus(S_a, S_b) = 1 - JSD(Spectrum(A_a) || Spectrum(A_b))
     """

 def cluster_by_spectrum(spectra_list, n_clusters='auto'):
     """
     Group generations by spectral signature.
     High internal cluster consistency → high confidence.
     """

 GPU-Native Implementation (NO scipy - avoids CUDA sync stall):
 def compute_spectral_gap_torch(laplacian):
     """
     Pure PyTorch GPU implementation - stays entirely on VRAM.

     WARNING: scipy.sparse.linalg.eigsh requires .cpu().numpy()
     which triggers CUDA Stream Synchronization and kills parallelism.

     For N < 4096, computing ALL eigenvalues on GPU is faster than
     moving data to CPU to compute just 2.

     Input: laplacian (batch, N, N) - must be symmetric
     """
     # torch.linalg.eigvalsh is optimized for symmetric matrices
     # Laplacians are always symmetric by construction
     eigenvalues = torch.linalg.eigvalsh(laplacian)  # Stays on GPU!

     # eigenvalues are sorted ascending: 0, λ_2, ..., λ_N
     # Spectral gap = difference between largest eigenvalues
     return eigenvalues[:, -1] - eigenvalues[:, -2]

 def compute_spectral_variance_torch(laplacian):
     """For fracture detection (TOHA-style), measure eigenvalue spread."""
     eigenvalues = torch.linalg.eigvalsh(laplacian)
     return eigenvalues.var(dim=-1)

 Optimization Strategy:
 def compute_spectral_consensus(adj_matrices, fast_mode=True):
     """
     All operations stay on GPU - no CPU transfer.
     fast_mode=True: Only compute spectral gaps
     fast_mode=False: Full spectrum comparison (still GPU-native)
     """
     laplacians = [compute_laplacian_torch(A) for A in adj_matrices]
     gaps = torch.stack([compute_spectral_gap_torch(L) for L in laplacians])
     return gaps.std()  # Low std = consistent reasoning across generations

 Key Equation:
 Consensus(S_a, S_b) = 1 - JSD(Spectrum(A_a) || Spectrum(A_b))

 ---
 Module 7: Main Pipeline (ag_sar.py)

 Purpose: Orchestrate full AG-SAR pipeline

 CRITICAL Performance Requirements:
 1. Inference Mode: Entire pipeline under torch.inference_mode() context
 2. Half Precision: All graph matrices and centrality vectors in float16/bfloat16

 Class Structure:
 class AGSAR:
     def __init__(self, model, tokenizer, config=None):
         self.model = model
         self.tokenizer = tokenizer
         self.config = config or AGSARConfig()
         self.dtype = next(model.parameters()).dtype  # Match model precision
         self._register_hooks()

     @torch.inference_mode()  # CRITICAL: Disable gradient tracking
     def compute_uncertainty(self, prompt, response):
         """
         Full pipeline:
         1. Extract attention maps + value vectors (via hooks)
         2. Filter semantic heads (entropy threshold)
         3. Build global attention graph (with padding mask)
         4. Compute sink-aware centrality (power iteration)
         5. Calculate Graph-Shifted Entropy
         """
         # Cast all intermediate tensors to model dtype
         rollout = rollout.to(self.dtype)
         centrality = centrality.to(self.dtype)

     @torch.inference_mode()
     def detect_hallucination(self, prompt, response, threshold=None):
         """Returns: (is_hallucination: bool, confidence: float, details: dict)"""

     @torch.inference_mode()
     def compare_generations(self, prompt, responses):
         """
         Multi-generation analysis:
         1. Compute GSE for each response
         2. Compute spectral consensus (GPU-native eigvalsh)
         3. Return confidence scores and clusters
         """

 Memory Optimization:
 # WRONG: float32 doubles memory usage for no gain
 rollout = compute_attention_rollout(attention_maps)  # float32 by default

 # RIGHT: Match model precision
 rollout = compute_attention_rollout(attention_maps).to(model.dtype)  # float16/bfloat16

 ---
 Module 8: Configuration (config.py)

 @dataclass
 class AGSARConfig:
     # Head filtering
     entropy_threshold_low: float = 0.3
     entropy_threshold_high: float = 0.95
     semantic_layers: int = 4  # Use last N layers

     # Centrality
     centrality_method: str = 'eigenvector'  # or 'pagerank'
     value_norm_type: str = 'l2'  # or 'l1'

     # Attention flow (Rollout method - efficient O(d*n^2))
     use_residual_correction: bool = True
     residual_weight: float = 0.5
     use_attention_rollout: bool = True  # Recursive matrix multiplication

     # Thresholds
     hallucination_threshold: float = 0.7
     sink_token_count: int = 4  # First N tokens to flag as potential sinks

     # Spectral analysis
     spectral_divergence_method: str = 'jsd'
     cluster_method: str = 'spectral'

 ---
 File Structure

 ag_sar/
 ├── __init__.py
 ├── config.py              # Configuration dataclass
 ├── attention_extractor.py # Extract attention & values from model
 ├── head_filter.py         # Entropy-based head filtering
 ├── attention_graph.py     # Graph construction & flow
 ├── centrality.py          # Eigenvector centrality + sink correction
 ├── gse.py                 # Graph-Shifted Entropy computation
 ├── spectral.py            # Laplacian spectrum & consensus
 ├── ag_sar.py              # Main pipeline class
 ├── utils.py               # Helper functions
 └── tests/
     ├── test_attention.py
     ├── test_centrality.py
     ├── test_gse.py
     └── test_pipeline.py

 ---
 Dependencies

 torch>=2.0          # torch.linalg.eigvalsh for GPU-native spectral analysis
 transformers>=4.30  # HuggingFace model support
 numpy               # Minimal usage (avoid for hot path)
 scikit-learn        # For clustering (spectral consensus)

 # NOTE: NO NetworkX - all graph ops in pure PyTorch for GPU acceleration
 # NOTE: NO scipy for spectral - torch.linalg.eigvalsh stays on GPU

 ---
 Critical Implementation Traps & Edge Cases

 1. The NetworkX Trap (AVOID)

 - Problem: NetworkX operates on CPU and converts tensors to Python lists
 - Impact: 100ms+ latency per token - destroys "zero-latency" goal
 - Solution: Pure PyTorch power iteration (shown in Module 4)

 2. The Padding Poison

 - Problem: Padding tokens act as perfect sinks (everyone attends to nothing-tokens)
 - Impact: Artificially high centrality for meaningless positions
 - Solution: Apply attention_mask to zero rows/columns BEFORE centrality (Module 3)

 3. Value Vector Hook Memory Leaks

 - Problem: Storing full (B, L, H, D) tensors explodes VRAM
 - Solution: Compute norms INSIDE hook, detach immediately

 4. Lower-Triangular Decay

 - Problem: Causal attention is lower-triangular; matrix multiplication decays early tokens
 - Solution: MUST add identity matrix (0.5A + 0.5I) BEFORE rollout

 5. GQA Dimension Mismatch (Llama-3, Mistral, Yi)

 - Problem: Attention has 32 Q heads, but Value has only 8 KV heads
 - Impact: Cannot multiply Centrality (32 dims) by ValueNorms (8 dims)
 - Solution: Broadcast KV norms to match Q heads: expand_value_norms_for_gqa() (Module 1)
 - Detection: Check model.config.num_attention_heads vs model.config.num_key_value_heads

 6. Scipy CPU Bottleneck (AVOID)

 - Problem: scipy.linalg.eigsh requires .cpu().numpy() - CUDA sync stall
 - Impact: Kills GPU parallelism even for "fast" spectral gap
 - Solution: Use torch.linalg.eigvalsh - stays entirely on GPU (Module 6)

 7. Precision Mismatch

 - Problem: Graph operations default to float32
 - Impact: Doubles memory usage when model is float16/bfloat16
 - Solution: Cast all tensors to model.dtype immediately after creation

 ---
 Implementation Order

 Phase 1: Core Infrastructure

 1. config.py - Define all hyperparameters
 2. attention_extractor.py - Hook into HuggingFace models (v_proj hooks)
 3. utils.py - Common helper functions

 Phase 2: Graph Construction

 4. head_filter.py - Entropy computation and filtering
 5. attention_graph.py - Pure PyTorch rollout with padding mask

 Phase 3: Relevance Computation

 6. centrality.py - PyTorch power iteration + sink-aware weighting

 Phase 4: Uncertainty Estimation

 7. gse.py - Graph-Shifted Entropy
 8. spectral.py - Spectral gap (fast) + full spectrum (on-demand)

 Phase 5: Integration

 9. ag_sar.py - Main pipeline orchestration
 10. Tests and validation

 ---
 Validation Strategy

 Unit Tests

 - Verify attention extraction matches model outputs
 - Test entropy filtering removes expected heads
 - Validate centrality computation against NetworkX

 Integration Tests

 - End-to-end pipeline on known examples
 - Compare GSE vs standard entropy on hallucination benchmarks

 Benchmarks

 - TruthfulQA: Measure hallucination detection accuracy
 - SQuAD: Measure calibration (uncertainty correlates with errors)
 - Compare latency: AG-SAR vs SAR (external semantic model)

 ---
 Key Equations Summary

 | Component              | Equation                           |
 |------------------------|------------------------------------|
 | Global Attention Graph | `A_global = (1/                    |
 | Residual Correction    | A = 0.5 * W_att + 0.5 * I          |
 | Sink-Aware Relevance   | `R(t_i) = C_eigen(t_i) ×           |
 | Normalized Relevance   | R̃(t_i) = R(t_i) / Σ_j R(t_j)
 | Graph-Shifted Entropy  | GSE(T) = Σ_i H(t_i) × R̃(t_i)
 | Spectral Consensus     | `Consensus = 1 - JSD(Spectrum(A_a) |

 ---
 Complexity Analysis

 | Operation              | Complexity   | Notes                  |
 |------------------------|--------------|------------------------|
 | Attention Extraction   | O(1)         | Single forward pass    |
 | Head Filtering         | O(l×h×t²)    | Entropy over all heads |
 | Graph Construction     | O(t²)        | Matrix operations      |
 | Eigenvector Centrality | O(t² × iter) | Power iteration        |
 | Value Norm             | O(t × d_v)   | Per-token norm         |
 | GSE Computation        | O(t)         | Weighted sum           |
 | Total                  | O(t²)        | Dominated by graph ops |

 Comparison to Original SAR: O(K×N) external model calls → O(t²) internal matrix ops
