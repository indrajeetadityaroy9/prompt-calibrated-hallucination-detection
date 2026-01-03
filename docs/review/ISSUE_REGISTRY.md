# AG-SAR Issue Registry

**Review Date:** 2026-01-03
**Review Scope:** Llama family architectures, Documentation only
**Framework Version:** 0.4.0

---

## Issue Severity Definitions

| Severity | Definition | Impact |
|----------|------------|--------|
| CRITICAL | Fundamental correctness bug | Incorrect results, system failure |
| HIGH | Significant issue affecting core functionality | Degraded accuracy or stability |
| MEDIUM | Notable issue with workarounds | Suboptimal behavior |
| LOW | Minor issue or improvement opportunity | Code quality concern |

---

## Stage 1: Configuration & Initialization Layer

### ISSUE-001: TF32 Auto-Enabled at Import Time
| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **File** | `src/ag_sar/__init__.py` |
| **Lines** | 28-29 |
| **Impact** | Stability, Precision |

**Description:**
`enable_h100_optimizations()` is called unconditionally at module import time, changing global PyTorch settings without user consent.

**Code:**
```python
from .utils import enable_h100_optimizations
enable_h100_optimizations()
```

**Root Cause:**
Library assumes all users want H100 optimizations. TF32 reduces precision (19-bit vs 32-bit mantissa) which may affect numerical reproducibility.

**Impact:**
- Changes global PyTorch state for ALL code in the process
- Users importing `ag_sar` for inspection will have TF32 enabled
- No opt-out mechanism before import
- Silently reduces precision on older GPUs that don't benefit from TF32

**Suggested Remediation:**
Make TF32 opt-in via explicit function call or environment variable:
```python
# Option 1: Explicit call required
# from ag_sar import enable_optimizations; enable_optimizations()

# Option 2: Environment variable
# AG_SAR_ENABLE_H100_OPT=1 python script.py
```

---

### ISSUE-002: Missing hallucination_threshold Validation
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/config.py` |
| **Lines** | 50, 92-115 |
| **Impact** | Accuracy |

**Description:**
`hallucination_threshold` (default 0.7) is not validated to be in [0, 1] range. Values outside this range produce undefined detection behavior.

**Code:**
```python
hallucination_threshold: float = 0.7  # Line 50
# __post_init__ does NOT validate this field
```

**Root Cause:**
Validation was not added to `__post_init__` method.

**Impact:**
- `threshold < 0`: All samples classified as hallucinations
- `threshold > 1`: No samples classified as hallucinations (if score is [0,1])
- Runtime errors or silent misbehavior depending on metric

**Suggested Remediation:**
```python
if not 0.0 <= self.hallucination_threshold <= 1.0:
    raise ValueError(f"hallucination_threshold must be in [0, 1]")
```

---

### ISSUE-003: Missing sink_token_count Validation
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/config.py` |
| **Lines** | 78, 92-115 |
| **Impact** | Accuracy |

**Description:**
`sink_token_count` (default 4) is not validated. Can be negative or exceed sequence length.

**Code:**
```python
sink_token_count: int = 4  # Line 78
# No validation in __post_init__
```

**Root Cause:**
No upper bound check at config time; lower bound check missing entirely.

**Impact:**
- `sink_token_count < 0`: Negative indexing in sink masking
- `sink_token_count >= seq_len`: All tokens masked as sinks → no valid relevance scores
- Power iteration converges to zero vector

**Suggested Remediation:**
```python
if self.sink_token_count < 0:
    raise ValueError(f"sink_token_count must be >= 0")
# Upper bound validated at runtime when sequence length is known
```

---

### ISSUE-004: Missing power_iteration_tol Validation
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/config.py` |
| **Lines** | 47, 92-115 |
| **Impact** | Performance |

**Description:**
`power_iteration_tol` (default 1e-4) is not validated to be positive. Zero or negative values break early stopping.

**Code:**
```python
power_iteration_tol: float = 1e-4  # Line 47
# No validation
```

**Root Cause:**
Validation oversight.

**Impact:**
- `tol <= 0`: Power iteration never converges early, always runs max steps
- `tol = 0`: Convergence check `|v_new - v| < tol` never satisfied due to floating point

**Suggested Remediation:**
```python
if self.power_iteration_tol <= 0:
    raise ValueError(f"power_iteration_tol must be > 0")
```

---

### ISSUE-005: Missing kurtosis_threshold Bounds Check
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/config.py` |
| **Lines** | 31, 92-115 |
| **Impact** | Accuracy |

**Description:**
`kurtosis_threshold` (default 2.0) can be any float including negative. Negative values invert the register filter behavior.

**Code:**
```python
kurtosis_threshold: float = 2.0  # Line 31
# No validation
```

**Root Cause:**
No bounds check; unclear if negative values are intentional.

**Impact:**
Register mask formula: `M(t) = sigmoid(-Z(t) + τ)`
- `τ < 0`: Low-kurtosis tokens (sinks) get HIGH mask values instead of low
- Inverts the intended filtering behavior

**Suggested Remediation:**
Document allowed range or add validation. If negative is invalid:
```python
# At minimum, document the expected behavior for various ranges
```

---

### ISSUE-006: Lambda Roughness Scale Mismatch
| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **File** | `src/ag_sar/config.py` |
| **Lines** | 39 |
| **Impact** | Accuracy |

**Description:**
`lambda_roughness=10.0` is designed for spectral roughness (Dirichlet energy, unbounded) but also used with MLP divergence which is bounded [0, 2].

**Code:**
```python
lambda_roughness: float = 10.0  # Line 39
# Used in: A_final = A / (1 + λ × roughness)
```

**Root Cause:**
Single parameter used for two different roughness metrics with vastly different scales:
- Spectral roughness: 0 to ~100+ (unbounded Dirichlet energy)
- MLP divergence: 0 to 2 (cosine dissimilarity)

**Impact:**
When using MLP divergence (v3.2):
- `λ=10, roughness=2 → A/(1+20) = A/21` (extremely aggressive penalty)
- When using spectral roughness:
- `λ=10, roughness=50 → A/(1+500) = A/501` (severe penalty but may be appropriate)

**Suggested Remediation:**
Either:
1. Separate parameters: `lambda_spectral_roughness`, `lambda_mlp_divergence`
2. Auto-scale based on roughness type
3. Document expected roughness ranges for each type

---

### ISSUE-007: Float16 Warning Not Enforced
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/config.py` |
| **Lines** | 101-106 |
| **Impact** | Stability |

**Description:**
Float16 causes NaN overflow with GPT-2, but config only warns without preventing use.

**Code:**
```python
if self.preferred_dtype == torch.float16:
    import warnings
    warnings.warn(
        "float16 may cause NaN overflow with GPT-2. Use bfloat16 instead.",
        UserWarning
    )
```

**Root Cause:**
Design choice to warn rather than error. Users may miss warning.

**Impact:**
- GPT-2 + float16 → NaN in attention softmax
- Silent model failure if warnings are suppressed

**Suggested Remediation:**
For GPT-2 specifically, could be an error or auto-switch to bfloat16:
```python
# At minimum, warnings.warn with stacklevel=2 for better visibility
```

---

### ISSUE-008: Cross-Parameter Consistency Not Validated
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/config.py` |
| **Lines** | 92-115 |
| **Impact** | Usability |

**Description:**
Metric-specific parameters are validated even when their metric is not enabled.

**Examples:**
- `mcss_beta`, `mcss_hebbian_tau`, `mcss_penalty_weight` validated but unused when `uncertainty_metric != "mcss"`
- `steering_alpha`, `steering_beta` exist but `head_scores_path` not validated when `use_spectral_steering=True`

**Root Cause:**
All fields validated uniformly without considering which are active.

**Impact:**
- Users may set invalid values for disabled features without errors
- When feature is enabled, invalid values cause runtime failures

**Suggested Remediation:**
Conditional validation based on active features:
```python
if self.use_spectral_steering:
    if self.head_scores_path is None:
        raise ValueError("head_scores_path required when use_spectral_steering=True")
```

---

### ISSUE-009: Semantic Layers Validation Deferred
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/config.py` |
| **Lines** | 42, 98-99 |
| **Impact** | Usability |

**Description:**
`semantic_layers >= 1` is validated but `semantic_layers <= num_hidden_layers` is not checked until engine initialization.

**Code:**
```python
semantic_layers: int = 4  # Line 42

if self.semantic_layers < 1:  # Line 98-99
    raise ValueError(f"semantic_layers must be >= 1")
# No check against num_hidden_layers
```

**Root Cause:**
Model architecture not known at config creation time.

**Impact:**
- Config appears valid but fails at AGSAR initialization
- Error message context may be unclear

**Suggested Remediation:**
Document that validation happens at engine init, or require num_hidden_layers in config:
```python
# Option: Add to docstring
# Note: semantic_layers must be <= num_hidden_layers; validated at engine init
```

---

### ISSUE-010: get_optimal_dtype() Hardware Detection Gaps
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/utils/tensor.py` |
| **Lines** | 86-102 |
| **Impact** | Stability |

**Description:**
`get_optimal_dtype()` assumes compute capability >= 8 means bfloat16 support without explicit verification.

**Code:**
```python
major, _ = torch.cuda.get_device_capability()
if major >= 8:  # Ampere (A100) or newer (H100)
    return torch.bfloat16
return torch.float32
```

**Root Cause:**
Uses compute capability as proxy for dtype support rather than explicit check.

**Impact:**
- Edge cases where bfloat16 is not supported despite high compute capability
- No fallback mechanism

**Suggested Remediation:**
Add explicit dtype support check:
```python
if major >= 8 and torch.cuda.is_bf16_supported():
    return torch.bfloat16
```

---

## Summary: Stage 1 Issues

| Issue ID | Severity | File | Brief Description |
|----------|----------|------|-------------------|
| ISSUE-001 | HIGH | `__init__.py` | TF32 auto-enabled at import |
| ISSUE-002 | MEDIUM | `config.py` | hallucination_threshold not validated |
| ISSUE-003 | MEDIUM | `config.py` | sink_token_count not validated |
| ISSUE-004 | LOW | `config.py` | power_iteration_tol not validated |
| ISSUE-005 | MEDIUM | `config.py` | kurtosis_threshold no bounds |
| ISSUE-006 | HIGH | `config.py` | lambda_roughness scale mismatch |
| ISSUE-007 | MEDIUM | `config.py` | float16 warning not enforced |
| ISSUE-008 | LOW | `config.py` | cross-parameter consistency |
| ISSUE-009 | MEDIUM | `config.py` | semantic_layers validation deferred |
| ISSUE-010 | LOW | `tensor.py` | get_optimal_dtype() detection gaps |

**Stage 1 Totals:**
- CRITICAL: 0
- HIGH: 2
- MEDIUM: 5
- LOW: 3

---

---

## Stage 2: Model Adaptation & Hook Layer (Llama-Focused)

### ISSUE-011: cleanup() Uses Wrong Attribute Path for GPT-2
| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **File** | `src/ag_sar/modeling/hooks.py` |
| **Lines** | 241-242 |
| **Impact** | Stability |

**Description:**
`cleanup()` assumes Llama architecture with `backbone.layers[layer_idx]` but GPT-2 uses `backbone.h[layer_idx]`. This causes AttributeError on GPT-2 cleanup.

**Code:**
```python
for layer_idx, original in self._original_forwards.items():
    self.backbone.layers[layer_idx].self_attn.forward = original  # GPT-2 doesn't have .layers!
```

**Root Cause:**
Cleanup code written for Llama but not adapted for GPT-2 architecture.

**Impact:**
- `cleanup()` crashes on GPT-2 models
- Resources not properly released
- Potential memory leaks in repeated usage

**Suggested Remediation:**
```python
if self.architecture == "gpt2":
    # GPT-2 uses hooks, not monkey-patches, so _original_forwards is empty
    pass
else:
    for layer_idx, original in self._original_forwards.items():
        self.backbone.layers[layer_idx].self_attn.forward = original
```

---

### ISSUE-012: Architecture Detection Lacks Bounds Checking
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/modeling/hooks.py` |
| **Lines** | 174, 179 |
| **Impact** | Stability |

**Description:**
Architecture detection accesses `.h[0]` and `.layers[0]` without checking if the model has any layers.

**Code:**
```python
if hasattr(self.model.transformer.h[0].attn, 'c_attn'):  # Line 174
if hasattr(self.model.model.layers[0].self_attn, 'q_proj'):  # Line 179
```

**Root Cause:**
Assumes model always has at least one layer after checking for list attribute.

**Impact:**
- Empty or malformed models cause IndexError
- Error message is cryptic, not informative

**Suggested Remediation:**
```python
if hasattr(self.model.transformer, 'h') and len(self.model.transformer.h) > 0:
    if hasattr(self.model.transformer.h[0].attn, 'c_attn'):
```

---

### ISSUE-013: Hard Imports from Transformers Internals
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/modeling/hooks.py` |
| **Lines** | 324, 398, 478 |
| **Impact** | Stability |

**Description:**
Llama/Qwen/Mistral patches import internal transformers functions that may change between versions.

**Code:**
```python
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv  # Line 324
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv  # Line 398
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv  # Line 478
```

**Root Cause:**
Need post-RoPE Q/K which requires using internal RoPE application functions.

**Impact:**
- Breaking changes in transformers >= 4.45 (documented in CLAUDE.md)
- Import failures cause cryptic errors at runtime (not import time)
- No fallback mechanism

**Suggested Remediation:**
Wrap imports with version checks and provide fallback implementations:
```python
try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
except ImportError:
    # Fallback implementation or clear error message
```

---

### ISSUE-014: Device Handling in Multi-GPU Llama Patches
| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **File** | `src/ag_sar/modeling/hooks.py` |
| **Lines** | 355-357, 434-436, 513-515 |
| **Impact** | Stability |

**Description:**
Llama patches store tensors with `.to(adapter.dtype)` but not `.to(device)`. In multi-GPU with `device_map="balanced"`, layers reside on different GPUs.

**Code:**
```python
adapter.capture.query_states[layer_idx] = query_states.detach().to(adapter.dtype)
adapter.capture.key_states[layer_idx] = key_states.detach().to(adapter.dtype)
adapter.capture.value_norms[layer_idx] = torch.norm(value_states, dim=-1, p=2).detach().to(adapter.dtype)
```

**Root Cause:**
`adapter.dtype` conversion doesn't change device. Tensors stay on layer's GPU but `adapter.device` (from first parameter) may differ.

**Impact:**
- Tensors scattered across GPUs
- When aggregating in `extract()`, implicit device transfers occur
- NVLink bandwidth bottleneck, potential CUDA errors

**Suggested Remediation:**
Either explicitly manage devices or document that all layers should be on same GPU:
```python
# Option 1: Keep on native device (current implicit behavior)
# Document: Tensors on layer-native devices, aggregation moves to primary
# Option 2: Move to primary immediately
.to(device=adapter.device, dtype=adapter.dtype)
```

---

### ISSUE-015: value_norms Shape Inconsistency
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/modeling/hooks.py` |
| **Lines** | 295, 357, 436, 515 |
| **Impact** | Accuracy |

**Description:**
`value_norms` computed as `torch.norm(v_heads/value_states, dim=-1, p=2)` produces shape `(B, H, S)` but some downstream code may expect `(B, S)`.

**Code:**
```python
# GPT-2 (line 295):
self.capture.value_norms[layer_idx] = torch.norm(v_heads, dim=-1, p=2)  # (B, H, S)

# Llama (line 357):
adapter.capture.value_norms[layer_idx] = torch.norm(value_states, dim=-1, p=2)  # (B, num_kv_heads, S)
```

**Root Cause:**
Computes L2 norm over head_dim, preserving head dimension.

**Impact:**
- Downstream `aggregate_value_norms()` expects dict[layer, (B, H, S)] and handles correctly
- But if code directly uses value_norms expecting (B, S), shape mismatch occurs

**Suggested Remediation:**
Document expected shape clearly:
```python
# Returns: (B, num_heads, S) for GPT-2, (B, num_kv_heads, S) for Llama
```

---

### ISSUE-016: GQA Expansion Memory Impact (Llama-3.x)
| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **File** | `src/ag_sar/modeling/hooks.py` |
| **Lines** | 621-629 |
| **Impact** | Performance |

**Description:**
GQA expansion via `repeat_interleave` creates memory copies. For Llama-3.1-70B (64 Q heads, 8 KV heads), this is 8x memory expansion per layer.

**Code:**
```python
if self.heads_per_group > 1:
    K = K.repeat_interleave(self.heads_per_group, dim=1)  # 8 → 64 heads
    if layer_idx in self.capture.value_norms:
        v_norms = self.capture.value_norms[layer_idx]
        V_norms_expanded[layer_idx] = v_norms.repeat_interleave(
            self.heads_per_group, dim=1
        )
```

**Root Cause:**
GQA requires matching Q and K head counts for centrality computation.

**Impact for Llama-3.1-70B:**
- K: (B, 8, S, 128) → (B, 64, S, 128) = 8x memory
- Per layer: 8 * S * 128 * 2 bytes (bf16) * 8 = 16KB * S per layer
- For S=4096, 80 layers: ~5GB additional memory

**Suggested Remediation:**
Consider lazy expansion or streaming to reduce peak memory:
```python
# Option: Expand only needed layers, not all at once
# Option: Use view tricks where possible (not applicable here due to interleave)
```

---

### ISSUE-017: Monkey-Patch State Overwrite Risk
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/modeling/hooks.py` |
| **Lines** | 167, 221-233, 328, 402, 482 |
| **Impact** | Stability |

**Description:**
If `register()` called twice, `_original_forwards[layer_idx]` is overwritten with already-patched forward, losing the original.

**Code:**
```python
def register(self) -> None:
    if self._is_registered:
        return  # Early exit prevents double-patching...

    self.cleanup()  # But cleanup() restores from _original_forwards which may be wrong
```

**Root Cause:**
Guard at line 223-224 prevents double-patching, but if guard fails (e.g., `_is_registered` incorrectly set), original is lost.

**Impact:**
- Subsequent `cleanup()` restores patched version, not original
- Forward method permanently modified
- Affects model behavior outside AG-SAR usage

**Suggested Remediation:**
Add explicit check before overwriting:
```python
if layer_idx not in self._original_forwards:
    self._original_forwards[layer_idx] = attn.forward
# Only patch if not already patched
```

---

### ISSUE-018: AttentionCapture.get_device() Unused
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/modeling/hooks.py` |
| **Lines** | 115-119 |
| **Impact** | Usability |

**Description:**
`get_device()` method is implemented but never called anywhere in the codebase.

**Code:**
```python
def get_device(self, layer_idx: int) -> Optional[torch.device]:
    """Get the device where a layer's tensors are stored."""
    if layer_idx in self.query_states:
        return self.query_states[layer_idx].device
    return None
```

**Root Cause:**
Method added for multi-GPU support but integration not completed.

**Impact:**
- Dead code confuses maintainers
- Multi-GPU device-aware aggregation not actually implemented
- Docstring claims (lines 92-95) are misleading

**Suggested Remediation:**
Either implement device-aware aggregation or remove dead code.

---

### ISSUE-019: Flash Attention Context Defeated by Manual Attention Capture
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/modeling/hooks.py` |
| **Lines** | 372-379, 586-609 |
| **Impact** | Performance |

**Description:**
The patched forward methods compute attention weights manually (`torch.matmul` + `softmax`) and capture them, which defeats Flash Attention's O(1) memory benefit.

**Code in patch (lines 372-379):**
```python
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
adapter.capture.attention_weights[layer_idx] = attn_weights.detach()  # O(S²) memory!
```

**Root Cause:**
AG-SAR needs attention weights for authority flow, but Flash Attention doesn't expose them.

**Impact:**
- Creates O(S²) attention matrix per layer despite using Flash Attention context
- For S=4096, H=64: 4096² * 64 * 2 bytes = 2GB per layer
- Flash Attention speedup partially lost (still get faster forward, but memory spike)

**Suggested Remediation:**
Document trade-off:
```python
# NOTE: Capturing attention_weights creates O(S²) memory per layer
# Flash Attention context still provides speedup for the attention output computation
# For long sequences, consider using matrix-free centrality only (no authority flow)
```

---

### ISSUE-020: heads_per_group Divisibility Not Validated
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/modeling/hooks.py` |
| **Lines** | 219 |
| **Impact** | Accuracy |

**Description:**
`heads_per_group = num_heads // num_kv_heads` assumes exact divisibility without validation.

**Code:**
```python
self.heads_per_group = self.num_heads // self.num_kv_heads
```

**Root Cause:**
Integer division silently truncates if not evenly divisible.

**Impact:**
- If `num_heads=33, num_kv_heads=8`: `heads_per_group=4` (truncated from 4.125)
- GQA expansion creates 32 heads instead of 33
- Q/K shape mismatch in centrality computation

**Suggested Remediation:**
```python
if self.num_heads % self.num_kv_heads != 0:
    raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")
self.heads_per_group = self.num_heads // self.num_kv_heads
```

---

## Summary: Stage 2 Issues (Llama-Focused)

| Issue ID | Severity | File | Brief Description |
|----------|----------|------|-------------------|
| ISSUE-011 | HIGH | `hooks.py` | cleanup() uses wrong attribute for GPT-2 |
| ISSUE-012 | LOW | `hooks.py` | No bounds check in architecture detection |
| ISSUE-013 | MEDIUM | `hooks.py` | Hard imports from transformers internals |
| ISSUE-014 | HIGH | `hooks.py` | Device handling in multi-GPU Llama patches |
| ISSUE-015 | MEDIUM | `hooks.py` | value_norms shape inconsistency |
| ISSUE-016 | HIGH | `hooks.py` | GQA expansion 8x memory for Llama-70B |
| ISSUE-017 | MEDIUM | `hooks.py` | Monkey-patch state overwrite risk |
| ISSUE-018 | LOW | `hooks.py` | get_device() method unused |
| ISSUE-019 | MEDIUM | `hooks.py` | Flash Attention defeated by manual capture |
| ISSUE-020 | LOW | `hooks.py` | heads_per_group divisibility not validated |

**Stage 2 Totals:**
- CRITICAL: 0
- HIGH: 3
- MEDIUM: 4
- LOW: 3

---

---

## Stage 3: Operations Layer (Kernels & Functional)

### CLARIFICATION: Causal Mask Consistency
| Field | Value |
|-------|-------|
| **Status** | NOT AN ISSUE |
| **File** | `triton_kernels.py`, `torch_functional.py` |
| **Lines** | 138-139, 886-890 |

**Description:**
Initial exploration suggested causal mask direction differs between Triton and PyTorch. Upon detailed review, both implementations are **CONSISTENT**:

**Triton (lines 138-139):**
```python
causal_mask = offs_n[None, :] <= offs_m[:, None]  # True when j <= i
scores = tl.where(causal_mask, scores, float('-inf'))  # Keep scores when j <= i
```

**PyTorch (lines 886-890):**
```python
causal_mask = torch.triu(..., diagonal=1)  # True when j > i
attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))  # Mask when j > i
```

Both allow attending to positions where j <= i (current and past positions). **No inconsistency exists.**

---

### ISSUE-021: Silent Backend Fallback Without Logging
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/ops/__init__.py` |
| **Lines** | 16-25 |
| **Impact** | Usability |

**Description:**
Backend selection silently falls back to PyTorch without any logging or indication to users.

**Code:**
```python
if sys.platform == "linux" and not _FORCE_TORCH:
    try:
        import triton
        from .triton_kernels import centrality_kernel
        _TRITON_AVAILABLE = True
    except ImportError:
        pass  # Silent fallback

if not _TRITON_AVAILABLE:
    from .torch_functional import centrality_kernel_fallback as centrality_kernel
```

**Root Cause:**
No logging infrastructure in the fallback path.

**Impact:**
- Users unaware if Triton import failed on Linux
- Performance degradation without explanation
- Debugging difficulty when kernel performance differs

**Suggested Remediation:**
```python
import logging
logger = logging.getLogger(__name__)

if sys.platform == "linux" and not _FORCE_TORCH:
    try:
        from .triton_kernels import centrality_kernel
        _TRITON_AVAILABLE = True
        logger.info("Using Triton backend for centrality kernel")
    except ImportError as e:
        logger.warning(f"Triton import failed, using PyTorch fallback: {e}")
```

---

### ISSUE-022: Grid Calculation Hardcoded Block Size Mismatch
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/ops/triton_kernels.py` |
| **Lines** | 198 |
| **Impact** | Performance |

**Description:**
Triton grid calculation uses hardcoded block size of 64, but autotune configs include BLOCK_M values from 32 to 128.

**Code:**
```python
grid = (B, H, triton.cdiv(S, 64))  # Hardcoded 64

# But autotune configs include:
triton.Config({'BLOCK_M': 32, ...})  # Line 54
triton.Config({'BLOCK_M': 128, ...})  # Line 70
```

**Root Cause:**
Grid must be set before kernel launch, but BLOCK_M is selected at autotune time.

**Impact:**
- For BLOCK_M=32: Grid over-allocates (2x more blocks than needed)
- For BLOCK_M=128: Grid under-allocates (only half the blocks launch)
- May cause incorrect results for BLOCK_M != 64

**Suggested Remediation:**
Use 32 (minimum BLOCK_M) for safe grid calculation, or use multiple grid configurations:
```python
# Conservative: use minimum BLOCK_M
grid = (B, H, triton.cdiv(S, 32))
# Kernel checks bounds with mask_m = offs_m < S
```

---

### ISSUE-023: Autotune Key Only on Head Dimension
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/ops/triton_kernels.py` |
| **Lines** | 82 |
| **Impact** | Performance |

**Description:**
Autotune key only includes D (head dimension), not S (sequence length). Same config used for S=128 and S=4096.

**Code:**
```python
@triton.autotune(configs=_get_autotune_configs(), key=['D'])
```

**Root Cause:**
Avoiding recompilation for different sequence lengths.

**Impact:**
- Optimal config for S=4096 may be suboptimal for S=128
- Larger blocks better for long sequences, smaller for short
- Trade-off: avoiding recompile overhead vs per-sequence optimization

**Suggested Remediation:**
Could bucket sequence lengths:
```python
# key=['D', 'S_bucket'] where S_bucket = min(4096, 2**ceil(log2(S)))
# But increases compilation cache size
```

---

### ISSUE-024: Authority Flow Uses Python Loop (Non-Vectorizable)
| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **File** | `src/ag_sar/ops/torch_functional.py` |
| **Lines** | 589-608 |
| **Impact** | Performance |

**Description:**
`compute_authority_flow()` uses Python for loop over tokens, which cannot be optimized by torch.compile.

**Code:**
```python
for t in range(prompt_length, S):  # Python loop!
    prompt_attn = attn[:, t, :prompt_length].sum(dim=-1)
    if t > prompt_length:
        gen_attn = attn[:, t, prompt_length:t]
        gen_auth = authority[:, prompt_length:t]
        gen_flow = (gen_attn * gen_auth).sum(dim=-1)
    ...
    authority[:, t] = raw_authority
```

**Root Cause:**
Recursive dependency: authority[t] depends on authority[prompt_length:t], requiring sequential computation.

**Impact:**
- O(S) Python loop iterations
- Each iteration has tensor ops but loop overhead dominates for long sequences
- `@_compile_if_available` decorator on `compute_authority_flow_vectorized` (line 773) shows awareness, but that version is an approximation

**Suggested Remediation:**
Document trade-off between accuracy (sequential) and speed (vectorized approximation):
```python
# Use compute_authority_flow_vectorized() for ~3x speedup with slight accuracy loss
# Use compute_authority_flow() for exact recursive computation
```

---

### ISSUE-025: torch.compile No Error Handling
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/ops/torch_functional.py` |
| **Lines** | 38-44 |
| **Impact** | Stability |

**Description:**
`_compile_if_available` decorator applies `torch.compile` with `fullgraph=True` which fails on unsupported ops.

**Code:**
```python
def _compile_if_available(mode: str = "reduce-overhead"):
    def decorator(func):
        if _should_compile():
            return torch.compile(func, mode=mode, fullgraph=True)  # No try/except
        return func
    return decorator
```

**Root Cause:**
`fullgraph=True` requires entire function graph to be compilable. Dynamic control flow, certain ops, or graph breaks cause compilation failure.

**Impact:**
- Hard crash if any decorated function can't compile
- No fallback to uncompiled version
- May fail on certain input shapes or edge cases

**Suggested Remediation:**
```python
def _compile_if_available(mode: str = "reduce-overhead"):
    def decorator(func):
        if _should_compile():
            try:
                compiled = torch.compile(func, mode=mode, fullgraph=True)
                # Test compilation with dummy input
                return compiled
            except Exception as e:
                import logging
                logging.warning(f"torch.compile failed for {func.__name__}: {e}")
                return func
        return func
    return decorator
```

---

### ISSUE-026: S×S Causal Mask Created Per Forward Pass
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/ops/torch_functional.py` |
| **Lines** | 886-890 |
| **Impact** | Performance |

**Description:**
`centrality_kernel_fallback` creates (S, S) causal mask tensor on every forward pass.

**Code:**
```python
causal_mask = torch.triu(
    torch.ones(S, S, device=device, dtype=torch.bool),
    diagonal=1
)
```

**Root Cause:**
No caching mechanism for masks of same size.

**Impact:**
- S×S tensor allocation per forward
- For S=4096: 16MB boolean tensor created each time
- Adds latency and memory pressure

**Suggested Remediation:**
Cache masks by sequence length:
```python
_CAUSAL_MASK_CACHE = {}

def _get_causal_mask(S, device):
    key = (S, device)
    if key not in _CAUSAL_MASK_CACHE:
        _CAUSAL_MASK_CACHE[key] = torch.triu(
            torch.ones(S, S, device=device, dtype=torch.bool),
            diagonal=1
        )
    return _CAUSAL_MASK_CACHE[key]
```

---

### ISSUE-027: EMA State Shape Mismatch on Cold Start
| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **File** | `src/ag_sar/ops/torch_functional.py` |
| **Lines** | 286-293 |
| **Impact** | Accuracy |

**Description:**
When `ema_state=None`, the initialization creates a (1,) shape tensor that broadcasts incorrectly with (B, S) kurtosis values.

**Code:**
```python
if ema_state is None:
    ema_state = EMAState(
        mean=kurt.mean().unsqueeze(0),  # (1,) shape!
        var=kurt.var().unsqueeze(0).clamp(min=1e-6),  # (1,) shape!
        count=1,
    )
    z_score = torch.zeros_like(kurt)  # First batch: no Z-score filtering!
```

**Root Cause:**
`kurt.mean()` returns 0-d tensor, `.unsqueeze(0)` makes it (1,). This is meant to be a single global statistic, but the code at line 297 does:
```python
z_score = (kurt - ema_state.mean) / sigma  # (B, S) - (1,) / (1,)
```
This broadcasts correctly, so the immediate calculation is fine.

**The real issue:** On first call, `z_score = torch.zeros_like(kurt)` means the register filter is effectively disabled (all tokens pass).

**Impact:**
- First batch: Register filter completely inactive
- `mask = sigmoid(-0 + 2.0) ≈ 0.88` for all tokens
- Sinks/registers not filtered on first inference

**Suggested Remediation:**
Either initialize with reasonable defaults or document the warm-up behavior:
```python
# Option 1: Use reasonable initial statistics
if ema_state is None:
    ema_state = EMAState(
        mean=torch.tensor([0.0], device=device, dtype=dtype),  # Mean kurtosis ~0
        var=torch.tensor([4.0], device=device, dtype=dtype),   # Var from calibration
        count=0,
    )
    # Now Z-scores are computed normally
```

---

### ISSUE-028: SGSS Application Only to Response Tokens
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/ops/torch_functional.py` |
| **Lines** | 924-927 |
| **Impact** | Accuracy |

**Description:**
SGSS steering is only applied to response tokens (positions >= response_start), but if response_start=0, all tokens get SGSS.

**Code:**
```python
if response_start > 0:
    mask = torch.zeros(S, device=device, dtype=dtype)
    mask[response_start:] = 1.0
    w_sgss = 1.0 + (w_sgss - 1.0) * mask.unsqueeze(0).unsqueeze(0)
# If response_start == 0, w_sgss applied to ALL tokens
```

**Root Cause:**
Edge case not handled: response_start=0 means "entire sequence is response" which may not be the intent.

**Impact:**
- If caller passes response_start=0 by mistake, prompt tokens get steering
- Could destabilize prompt token centrality

**Suggested Remediation:**
Document expected behavior or add validation:
```python
# NOTE: response_start=0 applies SGSS to entire sequence
# For typical use, response_start should be > 0 (prompt length)
```

---

## Summary: Stage 3 Issues

| Issue ID | Severity | File | Brief Description |
|----------|----------|------|-------------------|
| CLARIFY | N/A | Both | Causal mask IS consistent (j <= i) |
| ISSUE-021 | MEDIUM | `__init__.py` | Silent backend fallback |
| ISSUE-022 | MEDIUM | `triton_kernels.py` | Grid hardcoded block size |
| ISSUE-023 | LOW | `triton_kernels.py` | Autotune key only on D |
| ISSUE-024 | HIGH | `torch_functional.py` | Authority flow Python loop |
| ISSUE-025 | MEDIUM | `torch_functional.py` | torch.compile no error handling |
| ISSUE-026 | LOW | `torch_functional.py` | S×S mask per forward |
| ISSUE-027 | CRITICAL | `torch_functional.py` | EMA state cold start |
| ISSUE-028 | LOW | `torch_functional.py` | SGSS response_start=0 edge case |

**Stage 3 Totals:**
- CRITICAL: 1
- HIGH: 1
- MEDIUM: 3
- LOW: 3

---

---

## Stage 4: Centrality & Graph Measures

### ISSUE-029: Epsilon Inconsistency Across Graph Operations
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/measures/graph.py` |
| **Lines** | 78, 84, 93, 120, 267 |
| **Impact** | Accuracy |

**Description:**
Graph operations use different epsilon values without documented rationale:

**Code:**
```python
v = v / v.sum(dim=-1, keepdim=True).clamp(min=1e-10)  # Line 78, 93, 120
signal_filter = value_norms / (value_norms.max(...)[0] + 1e-6)  # Line 84
weights = weights / (weights.max(...).values + 1e-6)  # Line 267
```

**Root Cause:**
Ad-hoc choices without standardization.

**Impact:**
- 1e-10 is very small, may cause numerical issues on float16
- 1e-6 is larger, safer but less precise
- Inconsistency complicates maintenance and debugging

**Suggested Remediation:**
Define module-level constant:
```python
EPS_SAFE = 1e-8  # Standard epsilon for numerical stability
```

---

### ISSUE-030: Hebbian Weights Silent Edge Case Handling
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/measures/graph.py` |
| **Lines** | 244 |
| **Impact** | Usability |

**Description:**
`compute_hebbian_weights` silently clamps `prompt_end_idx` without warning.

**Code:**
```python
prompt_end_idx = max(1, min(prompt_end_idx, S))
```

**Root Cause:**
Edge case handling for invalid indices.

**Impact:**
- Caller may pass 0 and get silently corrected to 1
- Could hide bugs in prompt boundary calculation
- No logging/warning when clamping occurs

**Suggested Remediation:**
Add logging for edge cases:
```python
if prompt_end_idx < 1:
    logger.debug(f"prompt_end_idx clamped from {prompt_end_idx} to 1")
```

---

### ISSUE-031: Power Iteration Convergence Not Reported
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/measures/graph.py` |
| **Lines** | 89-127 |
| **Impact** | Usability |

**Description:**
Power iteration has early stopping but doesn't report how many iterations were actually used.

**Code:**
```python
for _ in range(num_iterations):
    ...
    if (v_next - v).abs().max() < tol:
        v = v_next
        break  # No indication of early exit
```

**Root Cause:**
Function returns centrality but not convergence metadata.

**Impact:**
- Users can't verify if algorithm converged
- Debugging performance issues harder
- May want to adjust num_iterations based on typical convergence

**Suggested Remediation:**
Optionally return convergence info:
```python
# Return (centrality, iterations_used, converged_flag)
```

---

## Stage 5: Entropy & Surprisal Measures

### ISSUE-032: Entropy Alignment Shift Makes First Token Zero
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/measures/entropy.py` |
| **Lines** | 57-59 |
| **Impact** | Accuracy |

**Description:**
Entropy alignment shift sets first position to zero, which affects GSE calculation.

**Code:**
```python
entropy = torch.zeros_like(raw_entropy)
entropy[:, 1:] = raw_entropy[:, :-1]  # First position = 0
```

**Root Cause:**
In autoregressive models, `logits[i]` predicts `token[i+1]`. Shifting aligns entropy[i] with token[i].

**Impact:**
- First token always has entropy=0 (no prediction available)
- If first token is important for semantics, its contribution is lost
- Affects BOS token handling differently across models

**Suggested Remediation:**
Document the design decision:
```python
# NOTE: First position entropy is always 0 because there's no prediction for token[0]
# This is correct for autoregressive models where BOS token is structural
```

---

### ISSUE-033: detect_hallucination Name Collision
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/measures/entropy.py`, `src/ag_sar/engine.py` |
| **Lines** | entropy.py:122-141, engine.py (method) |
| **Impact** | Usability |

**Description:**
`detect_hallucination` exists as both a module-level function and an AGSAR class method with different signatures.

**Code in entropy.py:**
```python
def detect_hallucination(gse: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
```

**Code in engine.py:**
```python
def detect_hallucination(self, prompt: str, response: str, threshold: Optional[float] = None) -> Tuple[bool, float, Dict]:
```

**Root Cause:**
Different abstraction levels - tensor-level vs high-level API.

**Impact:**
- Import confusion: `from ag_sar import detect_hallucination` vs `agsar.detect_hallucination()`
- IDE autocomplete shows both
- Maintenance risk if one changes without the other

**Suggested Remediation:**
Rename one:
```python
# entropy.py: detect_hallucination_from_score()
# Or: _detect_hallucination_impl() (internal)
```

---

### ISSUE-034: MC-SS MAX Normalization Design Decision
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/measures/spectral.py` |
| **Lines** | 129-131 |
| **Impact** | Documentation |

**Description:**
MC-SS uses MAX-normalization while GSE uses L1-normalization. This is documented but not explained in the code.

**Code:**
```python
# MAX-normalize centrality (NOT L1!)
v_max = centrality.max(dim=-1, keepdim=True).values + 1e-10
v_norm = centrality / v_max
```

**Root Cause:**
Intentional design choice for discriminative power.

**Impact:**
- Confusion about why metrics use different normalizations
- May affect comparability between GSE and MC-SS scores
- Users may not understand when to use which metric

**Suggested Remediation:**
Add detailed docstring explanation:
```python
# MC-SS uses MAX-normalization to preserve token-level discriminability.
# Unlike L1 which distributes weight, MAX keeps high-centrality tokens at 1.0
# and low-centrality tokens proportionally lower.
# This is critical for detecting "confident lies" where a fabricated token
# may have low centrality despite low surprisal.
```

---

### ISSUE-035: Surprisal Padding Adds Zero At Wrong End
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/measures/spectral.py` |
| **Lines** | 54-57 |
| **Impact** | Documentation |

**Description:**
Surprisal padding adds zeros at the beginning (correct) but differs from entropy's shift logic.

**Code:**
```python
padding = torch.zeros((batch_size, 1), device=logits.device, dtype=logits.dtype)
surprisal = torch.cat([padding, nll], dim=1)  # [0, s_1, s_2, ..., s_{n-1}]
```

**Root Cause:**
Both entropy and surprisal need alignment, implemented slightly differently.

**Impact:**
- Entropy: `entropy[:, 1:] = raw_entropy[:, :-1]` (in-place shift)
- Surprisal: `cat([padding, nll])` (concat shift)
- Same result but different implementation patterns
- Minor maintenance inconsistency

**Suggested Remediation:**
Use consistent implementation pattern across both.

---

## Summary: Stages 4-5 Issues

| Issue ID | Severity | File | Brief Description |
|----------|----------|------|-------------------|
| ISSUE-029 | MEDIUM | `graph.py` | Epsilon inconsistency |
| ISSUE-030 | LOW | `graph.py` | Hebbian silent clamping |
| ISSUE-031 | LOW | `graph.py` | Convergence not reported |
| ISSUE-032 | MEDIUM | `entropy.py` | First token entropy always 0 |
| ISSUE-033 | MEDIUM | `entropy.py` | detect_hallucination name collision |
| ISSUE-034 | LOW | `spectral.py` | MAX vs L1 normalization underdocumented |
| ISSUE-035 | LOW | `spectral.py` | Surprisal padding implementation differs |

**Stages 4-5 Totals:**
- CRITICAL: 0
- HIGH: 0
- MEDIUM: 3
- LOW: 4

---

## Cumulative Issue Summary (Stages 1-5)

### By Severity
| Severity | Count | Issues |
|----------|-------|--------|
| **CRITICAL** | 2 | ISSUE-027 (EMA state), ISSUE-027 duplicate in Stage 3 |
| **HIGH** | 6 | ISSUE-001, -006, -011, -014, -016, -024 |
| **MEDIUM** | 15 | Multiple across all stages |
| **LOW** | 12 | Multiple across all stages |

### By Impact Category
| Category | Count | Key Issues |
|----------|-------|------------|
| **Accuracy** | 10 | EMA state, epsilon handling, normalization |
| **Stability** | 8 | Device handling, cleanup crash, torch.compile |
| **Performance** | 6 | GQA memory, Python loop, mask creation |
| **Usability** | 11 | Silent behaviors, missing logging, name collisions |

### Critical Path Issues (Must Address)
1. **ISSUE-027**: EMA state cold start disables register filter on first batch
2. **ISSUE-024**: Authority flow Python loop prevents optimization
3. **ISSUE-014**: Multi-GPU device handling causes implicit transfers
4. **ISSUE-016**: GQA expansion causes 8x memory for Llama-70B

### Clarifications Made
- **Causal Mask**: Triton and PyTorch are CONSISTENT (both use j <= i)
- **MC-SS MAX-norm**: Intentional design for discriminative power

---

## Stage 6: Authority Flow & Register Filter

### ISSUE-036: Authority Flow May Exceed 1.0 Before Clamping
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/ops/torch_functional.py` |
| **Lines** | 602 |
| **Impact** | Accuracy |

**Description:**
Authority flow formula `raw_authority = prompt_attn + gen_flow` can exceed 1.0 since both components are sums of attention weights multiplied by authority.

**Code:**
```python
raw_authority = prompt_attn + gen_flow  # Can exceed 1.0
# ...
authority = authority.clamp(0.0, 1.0)  # Clamped at end
```

**Root Cause:**
Mathematical property: if all attended tokens have authority=1.0, and attention sums to 1.0, then authority should be ≤1.0. But intermediate calculations can produce values slightly >1 due to numerical imprecision.

**Impact:**
- Final clamping fixes the issue, but intermediate states may be incorrect
- Debug logging would show unexpected values

**Suggested Remediation:**
Document this as expected behavior:
```python
# NOTE: raw_authority is clamped to [0, 1] at the end
# Intermediate values may exceed 1.0 due to accumulation
```

---

### ISSUE-037: Spectral Roughness Normalizes by D But Not by Attention Sum
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/ops/torch_functional.py` |
| **Lines** | 374-377 |
| **Impact** | Accuracy |

**Description:**
Spectral roughness normalizes by hidden dimension D but attention weights already sum to 1 due to softmax. The scaling rationale is unclear.

**Code:**
```python
roughness = (attn * pairwise_dist_sq).sum(dim=-1)  # (B, S)
roughness = roughness / D  # Divide by hidden dim
```

**Root Cause:**
Normalization choice for scale invariance - dividing by D makes roughness comparable across different model sizes.

**Impact:**
- Roughness values depend on model size
- `lambda_roughness=10.0` calibrated for specific D (768? 4096?)
- Cross-model comparison may be inconsistent

**Suggested Remediation:**
Document the expected roughness scale for calibration:
```python
# Roughness normalized by D for scale invariance
# Expected range: [0, ~10] for typical models
# lambda_roughness=10.0 calibrated for D=4096 (Llama-3.1-8B)
```

---

### ISSUE-038: MLP Divergence Range Actually [0, 2] Not [0, 1]
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/ops/torch_functional.py`, `src/ag_sar/measures/authority.py` |
| **Lines** | 426, 132 |
| **Impact** | Accuracy |

**Description:**
MLP divergence uses `1 - cosine_similarity`, producing range [0, 2]. However, documentation and lambda calibration may assume [0, 1].

**Code:**
```python
# torch_functional.py:426
divergence = 1.0 - cos_sim  # Range [0, 2]

# authority.py:132
#     - 0 = perfect alignment
#     - 2 = opposite directions
```

**Root Cause:**
Cosine similarity ranges from -1 to 1, so `1 - cos_sim` ranges from 0 to 2.

**Impact:**
- `lambda_roughness=10.0` with `divergence=2.0` gives:
  - `authority / (1 + 10 * 2) = authority / 21` (very aggressive penalty)
- The [0, 2] range is correctly documented in docstring but may surprise users

**Suggested Remediation:**
Already correctly documented; consider scaling divergence to [0, 1] for consistency:
```python
# Option: divergence = (1.0 - cos_sim) / 2  # Scale to [0, 1]
```

---

### ISSUE-039: Register Mask Sigmoid Saturation Undocumented
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/ops/torch_functional.py` |
| **Lines** | 314 |
| **Impact** | Usability |

**Description:**
Register mask sigmoid `sigmoid(-Z + τ)` saturates for extreme Z-scores, but the behavior is not documented.

**Code:**
```python
mask = torch.sigmoid(-z_score + kurtosis_threshold)
```

**Root Cause:**
Sigmoid saturates at 0 for input << 0 and at 1 for input >> 0.

**Impact:**
- For Z >> τ: mask ≈ 0 (token completely filtered)
- For Z << τ: mask ≈ 1 (token fully passes)
- Default τ=2.0 means Z=5 gives mask≈0.05, Z=-1 gives mask≈0.95
- Users may not understand the sigmoid dynamics

**Suggested Remediation:**
Add documentation:
```python
# Sigmoid dynamics with τ=2.0:
# Z = 0: mask ≈ 0.88 (neutral)
# Z = 2: mask ≈ 0.50 (threshold)
# Z = 5: mask ≈ 0.05 (strongly filtered)
# Z = -1: mask ≈ 0.95 (passes)
```

---

## Summary: Stage 6 Issues

| Issue ID | Severity | File | Brief Description |
|----------|----------|------|-------------------|
| ISSUE-024 | HIGH | `torch_functional.py` | Authority flow Python loop (from Stage 3) |
| ISSUE-027 | CRITICAL | `torch_functional.py` | EMA state cold start (from Stage 3) |
| ISSUE-036 | LOW | `torch_functional.py` | Authority may exceed 1.0 |
| ISSUE-037 | LOW | `torch_functional.py` | Spectral roughness normalization |
| ISSUE-038 | MEDIUM | `torch_functional.py` | MLP divergence range [0,2] |
| ISSUE-039 | LOW | `torch_functional.py` | Sigmoid saturation undocumented |

**Stage 6 New Issues:**
- CRITICAL: 0 (1 existing from Stage 3)
- HIGH: 0 (1 existing from Stage 3)
- MEDIUM: 1
- LOW: 3

---

---

## Stage 7: Engine Integration Layer

### ISSUE-040: compute_uncertainty Returns Any Type
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/engine.py` |
| **Lines** | 143, 148 |
| **Impact** | Usability |

**Description:**
`compute_uncertainty` return type is `Any` instead of proper Union type, reducing IDE support.

**Code:**
```python
def compute_uncertainty(
    self, prompt: str, response: str, return_details: bool = False
) -> Any:  # Should be Union[float, Dict[str, Any]]
```

**Root Cause:**
Type annotation not updated to match actual return values.

**Impact:**
- IDE cannot provide proper type hints
- Static type checkers (mypy) cannot validate usage

**Suggested Remediation:**
```python
from typing import Union

def compute_uncertainty(...) -> Union[float, Dict[str, Any]]:
```

---

### ISSUE-041: Streaming State Fields Never Used
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/engine.py` |
| **Lines** | 107-110 |
| **Impact** | Usability |

**Description:**
Several streaming state fields are initialized but never used in the codebase.

**Code:**
```python
self._authority_history: Optional[torch.Tensor] = None
self._prompt_length: int = 0  # Only used in init
self._value_history: Optional[torch.Tensor] = None
self._h_attn_history: Optional[torch.Tensor] = None
```

**Root Cause:**
Streaming/incremental uncertainty computation was planned but not implemented.

**Impact:**
- Dead code confuses maintainers
- Memory reserved for unused tensors
- API suggests streaming support that doesn't exist

**Suggested Remediation:**
Either implement streaming or remove dead code:
```python
# Option 1: Document as TODO
# TODO: Implement streaming uncertainty computation
# Option 2: Remove unused fields
```

---

### ISSUE-042: _tokenize Missing Empty Input Validation
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/engine.py` |
| **Lines** | 124-141 |
| **Impact** | Stability |

**Description:**
`_tokenize` doesn't validate empty prompt or response, which can cause downstream issues.

**Code:**
```python
def _tokenize(self, prompt: str, response: str) -> Tuple[...]:
    prompt_enc = self.tokenizer(prompt, ...)  # What if prompt == ""?
    response_enc = self.tokenizer(response, ...)  # What if response == ""?
```

**Root Cause:**
No input validation at entry point.

**Impact:**
- Empty prompt → response_start = 0 or 1 (BOS only)
- Empty response → sequence length equals prompt length
- May cause index errors or nonsensical scores

**Suggested Remediation:**
```python
if not prompt or not response:
    raise ValueError("prompt and response must be non-empty strings")
```

---

### ISSUE-043: Default num_layers Fallback Without Warning
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/engine.py` |
| **Lines** | 87 |
| **Impact** | Accuracy |

**Description:**
If model layer count detection fails, defaults to 12 layers without warning.

**Code:**
```python
else:
    num_layers = 12  # Default - no logging!
```

**Root Cause:**
Fallback for unknown model architectures.

**Impact:**
- User unaware that layer detection failed
- semantic_layers=4 may index non-existent layers
- Hooks may not capture correct layers

**Suggested Remediation:**
```python
else:
    import warnings
    warnings.warn(f"Could not detect num_hidden_layers, defaulting to 12")
    num_layers = 12
```

---

### ISSUE-044: detect_hallucination Redundant Tensor Conversion
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `src/ag_sar/engine.py` |
| **Lines** | 357 |
| **Impact** | Performance |

**Description:**
`detect_hallucination` converts float score to tensor, calls function, then extracts `.item()`.

**Code:**
```python
score = details['score']  # Already float
is_hall, conf = detect_hallucination(torch.tensor([score]), threshold)  # Convert to tensor
return is_hall.item(), conf.item(), details  # Back to Python scalars
```

**Root Cause:**
`detect_hallucination` from measures expects tensor input.

**Impact:**
- Unnecessary tensor allocation and copy
- Minor CPU overhead per call

**Suggested Remediation:**
Add scalar variant:
```python
# In entropy.py
def detect_hallucination_scalar(score: float, threshold: float) -> Tuple[bool, float]:
    is_hall = score > threshold
    confidence = abs(score - threshold) / max(threshold, 1 - threshold)
    return is_hall, confidence
```

---

### ISSUE-045: v31 Path Falls Back to GSE Silently
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `src/ag_sar/engine.py` |
| **Lines** | 294-296 |
| **Impact** | Accuracy |

**Description:**
`compute_uncertainty_v31` silently falls back to GSE if attention weights not captured.

**Code:**
```python
attn = attention_weights.get(last_layer)
if attn is None:
    # Fall back to GSE if no attention weights
    return self.compute_uncertainty(prompt, response, return_details)
```

**Root Cause:**
Some architectures may not capture attention weights.

**Impact:**
- User requests v31 but gets GSE without notification
- Score interpretation changes without warning
- Debug difficulty

**Suggested Remediation:**
```python
if attn is None:
    import warnings
    warnings.warn("Attention weights not available for v31; falling back to GSE")
    return self.compute_uncertainty(prompt, response, return_details)
```

---

### ISSUE-046: capture.value_states May Not Be Populated
| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **File** | `src/ag_sar/engine.py` |
| **Lines** | 273, 280 |
| **Impact** | Stability |

**Description:**
`compute_uncertainty_v31` accesses `capture.value_states` which is only populated for certain architectures.

**Code:**
```python
value_states = self._adapter.capture.value_states
# ...
v_states = value_states.get(last_layer)  # May be empty dict
```

**Root Cause:**
GPT-2 uses hook-based capture that populates value_norms but not value_states directly.

**Impact:**
- GPT-2 + v31 metric: v_states = None
- register_mask computation skipped
- Authority flow runs without register filtering

**Suggested Remediation:**
Add explicit check:
```python
if not value_states:
    # Value states not available - skip register filter
    import warnings
    warnings.warn("value_states not captured; register filter disabled")
```

---

## Summary: Stage 7 Issues

| Issue ID | Severity | File | Brief Description |
|----------|----------|------|-------------------|
| ISSUE-040 | LOW | `engine.py` | Return type is Any |
| ISSUE-041 | MEDIUM | `engine.py` | Streaming state unused |
| ISSUE-042 | MEDIUM | `engine.py` | No empty input validation |
| ISSUE-043 | MEDIUM | `engine.py` | Default num_layers no warning |
| ISSUE-044 | LOW | `engine.py` | Redundant tensor conversion |
| ISSUE-045 | MEDIUM | `engine.py` | v31 fallback silent |
| ISSUE-046 | HIGH | `engine.py` | value_states may not be populated |

**Stage 7 Totals:**
- CRITICAL: 0
- HIGH: 1
- MEDIUM: 4
- LOW: 2

---

---

## Stage 8: Testing & Validation Layer

### ISSUE-047: No Llama-Specific Integration Tests
| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **File** | `tests/integration/` |
| **Lines** | N/A |
| **Impact** | Accuracy |

**Description:**
Integration tests use GPT-2 as test model, but the review focus is Llama architecture. No Llama-specific integration tests exist.

**Files Checked:**
- `test_behavior.py` - Uses GPT-2
- `test_e2e_pipeline.py` - Uses GPT-2
- `test_pipeline.py` - Uses GPT-2

**Root Cause:**
Llama models too large for typical CI environments.

**Impact:**
- GQA head expansion not tested in integration
- Post-RoPE capture timing not validated
- Multi-GPU device handling not tested

**Suggested Remediation:**
Add Llama-specific tests with small model or mocks:
```python
@pytest.fixture
def small_llama_model():
    # Use a tiny Llama-like model (TinyLlama or mock)
    ...
```

---

### ISSUE-048: EMA State Cold Start Not Tested
| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **File** | `tests/unit/test_v31_ops.py` |
| **Lines** | N/A |
| **Impact** | Accuracy |

**Description:**
ISSUE-027 (EMA state cold start) is a CRITICAL issue, but there's no specific test verifying the cold start behavior.

**Root Cause:**
Tests pass EMA state or don't check first-batch behavior.

**Impact:**
- First-batch behavior (register filter inactive) not validated
- Regression could be introduced without detection

**Suggested Remediation:**
Add cold start test:
```python
def test_register_mask_cold_start():
    """Verify register mask behavior on first batch."""
    v = torch.randn(2, 64, 768)
    mask, ema = compute_register_mask(v, ema_state=None)
    # First batch: Z-scores should be zero, mask should be ~0.88
    assert mask.mean() > 0.8, "Cold start should not filter aggressively"
```

---

### ISSUE-049: No Multi-GPU Tests
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `tests/` |
| **Lines** | N/A |
| **Impact** | Stability |

**Description:**
No tests verify multi-GPU device handling, despite ISSUE-014 and ISSUE-016 identifying multi-GPU concerns.

**Root Cause:**
Multi-GPU testing requires special hardware and CI setup.

**Impact:**
- Device mismatches not caught
- NVLink transfer issues not detected
- Llama-70B deployment problems undetected

**Suggested Remediation:**
Add skippable multi-GPU tests:
```python
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2+ GPUs")
def test_multi_gpu_tensor_placement():
    ...
```

---

### ISSUE-050: Test Fixtures Only Use Default Dimensions
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `tests/conftest.py` |
| **Lines** | 31-54 |
| **Impact** | Accuracy |

**Description:**
Test fixtures use fixed dimensions (batch=2, seq=64, hidden=768, heads=12) that don't match Llama-3.1 dimensions.

**Code:**
```python
@pytest.fixture
def num_heads():
    return 12  # Llama-3.1-8B has 32 Q-heads, 8 KV-heads
```

**Root Cause:**
Fixtures designed for GPT-2-like architectures.

**Impact:**
- GQA scenarios (32 Q / 8 KV) not tested with proper dimensions
- Large sequence lengths (4096+) not tested

**Suggested Remediation:**
Add Llama-specific fixtures:
```python
@pytest.fixture
def llama_31_8b_dims():
    return {
        'num_q_heads': 32,
        'num_kv_heads': 8,
        'hidden_size': 4096,
        'head_dim': 128,
    }
```

---

### ISSUE-051: No Triton vs PyTorch Consistency Tests
| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **File** | `tests/unit/test_triton_centrality.py` |
| **Lines** | N/A |
| **Impact** | Accuracy |

**Description:**
While `test_triton_centrality.py` exists (2697 bytes), it's relatively small and may not comprehensively test Triton/PyTorch consistency.

**Root Cause:**
Triton only available on Linux, so cross-platform consistency testing is challenging.

**Impact:**
- Triton and PyTorch may produce different results
- Platform-specific bugs undetected

**Suggested Remediation:**
Add explicit consistency tests:
```python
@pytest.mark.skipif(not _TRITON_AVAILABLE, reason="Requires Triton")
def test_triton_pytorch_consistency():
    """Verify Triton and PyTorch produce same results."""
    from ag_sar.ops.triton_kernels import centrality_kernel as triton_kernel
    from ag_sar.ops.torch_functional import centrality_kernel_fallback

    # Run both, compare outputs
    assert torch.allclose(triton_out, torch_out, atol=1e-4)
```

---

## Summary: Stage 8 Issues

| Issue ID | Severity | File | Brief Description |
|----------|----------|------|-------------------|
| ISSUE-047 | HIGH | `tests/integration/` | No Llama-specific tests |
| ISSUE-048 | HIGH | `tests/unit/` | EMA cold start not tested |
| ISSUE-049 | MEDIUM | `tests/` | No multi-GPU tests |
| ISSUE-050 | LOW | `conftest.py` | Fixtures only default dims |
| ISSUE-051 | HIGH | `test_triton_centrality.py` | Triton/PyTorch consistency |

**Stage 8 Totals:**
- CRITICAL: 0
- HIGH: 3
- MEDIUM: 1
- LOW: 1

---

---

## Stage 9: Benchmark & Performance Validation

### ISSUE-052: Benchmark Uses Synthetic Data Only
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `benchmarks/benchmark_latency.py` |
| **Lines** | 39-51, 208 |
| **Impact** | Accuracy |

**Description:**
Benchmark uses synthetic random token IDs, which may not reflect real-world attention patterns.

**Code:**
```python
class SyntheticBenchmarkDataset(Dataset):
    def __getitem__(self, idx):
        return torch.randint(2, self.vocab_size, (self.seq_len,))

# Later:
input_ids = torch.randint(2, 1000, (batch_size, seq_len), device=device)
```

**Root Cause:**
Simplicity and reproducibility of synthetic data.

**Impact:**
- Random tokens produce uniform attention patterns
- Real text has localized attention (nearby words, named entities)
- Benchmark may underestimate memory pressure from attention spikes

**Suggested Remediation:**
Add realistic benchmark option:
```python
# Option: Use real text from validation set
# benchmark_realistic_data.py with wikitext or similar
```

---

### ISSUE-053: Zero-Latency Claim Not Validated for Llama
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `benchmarks/benchmark_latency.py` |
| **Lines** | 17-18 |
| **Impact** | Documentation |

**Description:**
Benchmark claims <10% overhead but testing primarily with GPT-2, not Llama architectures.

**Code:**
```python
# Pass Criteria:
# - < 10% overhead relative to baseline model forward pass
```

**Root Cause:**
Llama models too large for quick benchmark runs.

**Impact:**
- GQA head expansion (8x memory) may increase overhead significantly
- Multi-GPU scenarios not tested
- Zero-latency claim may not hold for Llama-70B

**Suggested Remediation:**
Add Llama-specific benchmarks:
```python
# benchmark_llama.py with TinyLlama or Llama-3.2-1B
# Or: Document that <10% claim is architecture-specific
```

---

### ISSUE-054: Memory Benchmark Missing
| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **File** | `benchmarks/benchmark_latency.py` |
| **Lines** | 95-99 |
| **Impact** | Performance |

**Description:**
`get_memory_usage()` exists but peak memory isn't comprehensively tracked during benchmarks.

**Code:**
```python
def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0
```

**Root Cause:**
Memory tracking is basic, doesn't capture peak or GQA expansion impact.

**Impact:**
- GQA expansion memory not profiled
- Peak memory during S×S mask creation not captured
- OOM risks undetected

**Suggested Remediation:**
Add peak memory tracking:
```python
torch.cuda.reset_peak_memory_stats()
# Run benchmark
peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
```

---

### ISSUE-055: Benchmark Doesn't Test torch.compile
| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **File** | `benchmarks/benchmark_latency.py` |
| **Lines** | N/A |
| **Impact** | Performance |

**Description:**
Benchmark doesn't specifically measure torch.compile overhead or speedup.

**Root Cause:**
torch.compile is enabled implicitly via `_compile_if_available` decorator.

**Impact:**
- First-run compilation overhead not isolated
- Compiled vs uncompiled speedup not quantified
- Users can't assess torch.compile benefit

**Suggested Remediation:**
Add compile-specific benchmark:
```python
# Test with AG_SAR_SKIP_COMPILE=1 vs default
# Measure first-run compilation time vs steady-state
```

---

## Summary: Stage 9 Issues

| Issue ID | Severity | File | Brief Description |
|----------|----------|------|-------------------|
| ISSUE-052 | LOW | `benchmark_latency.py` | Synthetic data only |
| ISSUE-053 | MEDIUM | `benchmark_latency.py` | Zero-latency not validated for Llama |
| ISSUE-054 | MEDIUM | `benchmark_latency.py` | Memory benchmark missing |
| ISSUE-055 | LOW | `benchmark_latency.py` | torch.compile not tested |

**Stage 9 Totals:**
- CRITICAL: 0
- HIGH: 0
- MEDIUM: 2
- LOW: 2

---

---

## Final Cumulative Issue Summary

### By Severity

| Severity | Count | Issues |
|----------|-------|--------|
| **CRITICAL** | 1 | ISSUE-027 |
| **HIGH** | 10 | ISSUE-001, -006, -011, -014, -016, -024, -046, -047, -048, -051 |
| **MEDIUM** | 20 | Multiple across all stages |
| **LOW** | 24 | Multiple across all stages |

**Total Issues: 55**

### By Impact Category

| Category | Count | Key Issues |
|----------|-------|------------|
| **Accuracy** | 15 | EMA cold start, epsilon handling, normalization, GQA |
| **Stability** | 12 | Device handling, cleanup, torch.compile, validation |
| **Performance** | 10 | Python loop, GQA memory, mask creation, benchmarks |
| **Usability** | 18 | Silent behaviors, logging, type hints, docs |

### Critical Path Issues (Priority 1 - Must Address)

1. **ISSUE-027** (CRITICAL): EMA state cold start disables register filter on first batch
2. **ISSUE-024** (HIGH): Authority flow Python loop prevents torch.compile optimization
3. **ISSUE-014** (HIGH): Multi-GPU device handling causes implicit transfers
4. **ISSUE-016** (HIGH): GQA expansion causes 8x memory for Llama-70B
5. **ISSUE-046** (HIGH): value_states may not be populated for all architectures

### Major Issues (Priority 2 - Should Address)

6. **ISSUE-001** (HIGH): TF32 auto-enabled at import without user consent
7. **ISSUE-006** (HIGH): Lambda roughness scale mismatch between spectral and MLP
8. **ISSUE-011** (HIGH): cleanup() crashes on GPT-2
9. **ISSUE-047** (HIGH): No Llama-specific integration tests
10. **ISSUE-048** (HIGH): EMA cold start behavior not tested
11. **ISSUE-051** (HIGH): Triton/PyTorch consistency not tested

### Clarifications Made

- **Causal Mask**: Triton and PyTorch are CONSISTENT (both use j <= i)
- **MC-SS MAX-norm**: Intentional design for discriminative power
- **MLP Divergence Range**: [0, 2] is correct and documented

---

## Recommendations for Publication Readiness

### Immediate Actions (Before Submission)

1. **Fix EMA cold start (ISSUE-027)**: Initialize with reasonable defaults or document warm-up requirement
2. **Add Llama integration tests (ISSUE-047)**: Use TinyLlama or similar small model
3. **Document lambda_roughness calibration (ISSUE-006)**: Specify expected range for spectral vs MLP

### Short-Term Actions (Before Camera-Ready)

4. **Add logging for backend selection (ISSUE-021)**: Users should know which backend is active
5. **Fix cleanup() for GPT-2 (ISSUE-011)**: Architecture-specific cleanup
6. **Add input validation (ISSUE-042, -043)**: Empty strings, layer count fallback

### Long-Term Actions (Post-Publication)

7. **Vectorize authority flow (ISSUE-024)**: Enable torch.compile optimization
8. **Add multi-GPU tests (ISSUE-049)**: Critical for Llama-70B deployment
9. **Implement streaming state (ISSUE-041)**: Or remove dead code

---

*Review completed: 2026-01-03*
*Total issues documented: 55*
*Stages reviewed: 9/9*
