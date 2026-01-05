# AG-SAR v9.0: Universal Hallucination Detection Proposal

## Executive Summary

AG-SAR v8.0 achieves strong performance on QA tasks (AUROC 0.903, TPR@5%FPR 0.422) but degrades significantly on summarization (AUROC 0.643) and factual attribution (AUROC 0.618). This document proposes architectural improvements for AG-SAR v9.0 to achieve robust, task-agnostic hallucination detection.

**Key Proposals:**
1. **Task-Adaptive Calibration** - Per-task temperature and aggregation learned from data
2. **Multi-Head Uncertainty Architecture** - Specialized uncertainty heads for different error types
3. **LLM-Specific Feature Enrichment** - Replace/augment MLP divergence with truthfulness probes
4. **Cross-Layer Attention Probing** - Capture layer-wise semantic dynamics

---

## 1. Problem Analysis: Why Performance Varies by Task

### Current Results (v8.0 with percentile_10 + T=1.8)

| Dataset | Task Type | AUROC | TPR@5%FPR | ECE | Notes |
|---------|-----------|-------|-----------|-----|-------|
| HaluEval QA | Question Answering | 0.903 | 0.422 | 0.260 | Strong |
| RAGTruth | QA + Context | 0.704 | 0.144 | 0.334 | Moderate |
| HaluEval Summ | Summarization | 0.643 | 0.104 | 0.017 | Weak discrimination |
| FAVA | Factual Attribution | 0.618 | 0.076 | 0.029 | Weak discrimination |

### Root Cause Analysis

**1. QA vs Summarization - Structural Differences:**
- **QA**: Clear factoid answer, single token focus, high attention concentration
- **Summarization**: Distributed information, multi-sentence synthesis, diffuse attention
- AG-SAR's Authority Flow assumes prompt→response provenance tracking works best with focused attention

**2. Weak Supervision Signal:**
- **FAVA**: Fine-grained attribution spans are harder to detect than binary QA correctness
- Summarization errors are often subtle paraphrasing issues, not explicit fabrications

**3. Fixed Hyperparameters:**
- Same `stability_sensitivity`, `dispersion_k`, `aggregation_method` for all tasks
- Temperature calibration is global, not task-specific

---

## 2. Proposed Improvements

### 2.1 Task-Adaptive Calibration (TAC)

**Concept:** Learn per-task calibration parameters from a small held-out set.

**Architecture:**
```
                            ┌─────────────────┐
                            │  Task Classifier │
                            │  (lightweight)   │
                            └────────┬─────────┘
                                     │
                                     ▼
           ┌─────────────────────────────────────────────┐
           │           Calibration Router                 │
           │  task_type → {temperature, aggregation}      │
           └─────────────────────────────────────────────┘
                                     │
          ┌────────────┬─────────────┼────────────┬────────────┐
          ▼            ▼             ▼            ▼            ▼
      QA Task    RAG Context   Summarization    FAVA      Free Gen
      T=1.2       T=1.5          T=2.5         T=2.0       T=1.0
      agg=p10    agg=mean       agg=p25       agg=p25    agg=mean
```

**Implementation in `AGSARConfig`:**
```python
@dataclass
class TaskAdaptiveConfig:
    """Per-task calibration parameters."""
    task_type: Literal["qa", "rag", "summarization", "attribution", "free_gen", "auto"]

    # Learned calibration parameters per task
    calibration_params: Dict[str, TaskCalibration] = field(default_factory=lambda: {
        "qa": TaskCalibration(temperature=1.2, aggregation="percentile_10", dispersion_k=5),
        "rag": TaskCalibration(temperature=1.5, aggregation="mean", dispersion_k=7),
        "summarization": TaskCalibration(temperature=2.5, aggregation="percentile_25", dispersion_k=10),
        "attribution": TaskCalibration(temperature=2.0, aggregation="percentile_25", dispersion_k=8),
        "free_gen": TaskCalibration(temperature=1.0, aggregation="mean", dispersion_k=5),
    })

    # Auto-detection from prompt structure
    auto_detect_task: bool = True
```

**Task Detection Heuristics:**
1. **QA**: Short prompt ending with "?", context contains factoid cues
2. **RAG**: Explicit context section (e.g., "Context:", "Document:")
3. **Summarization**: Keywords like "summarize", "summary", long context
4. **Attribution**: Citation markers, source references

**References:**
- [Thermometer (ICML 2024)](https://research.ibm.com/publications/thermometer-towards-universal-calibration-for-large-language-models): Task-conditioned auxiliary calibration models

---

### 2.2 Multi-Head Uncertainty Architecture (MHUD)

**Concept:** Different types of hallucinations manifest in different internal patterns. Use specialized "uncertainty heads" to detect each type.

**Hallucination Taxonomy:**
| Type | Signal Source | Detection Head |
|------|---------------|----------------|
| **Fabrication** (invented facts) | Low authority flow from context | Authority Head |
| **Confusion** (wrong entity) | High semantic dispersion | Dispersion Head |
| **Overconfidence** (wrong but certain) | Low MLP divergence + high logit | Confidence Head |
| **Abstention failure** (should refuse) | Pattern in early layers | Refusal Head |

**Architecture:**
```
                    ┌──────────────────────────────┐
                    │      AG-SAR v9.0 Engine       │
                    └──────────────────────────────┘
                                   │
           ┌───────────────────────┴───────────────────────┐
           │                  Feature Bank                  │
           │  • attention_weights (per layer)               │
           │  • h_attn, h_block (MLP divergence)            │
           │  • logits → top-k probs                        │
           │  • hidden_states[layer_idx] (for probes)       │
           └───────────────────────────────────────────────┘
                                   │
        ┌──────────┬──────────┬────┴─────┬──────────┐
        ▼          ▼          ▼          ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │Authority│ │Dispersion│ │Confidence│ │ Refusal │ │Coherence│
   │  Head   │ │   Head   │ │   Head   │ │  Head   │ │  Head   │
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
        │          │          │          │          │
        └──────────┴──────────┴────┬─────┴──────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │     Uncertainty Aggregator    │
                    │  u = Σ w_i × head_i(features) │
                    │  (learned weights or gating)  │
                    └──────────────────────────────┘
```

**Implementation Sketch:**
```python
class MultiHeadUncertainty(nn.Module):
    """Multi-head uncertainty detection with specialized heads."""

    def __init__(self, hidden_size: int, num_heads: int = 5):
        super().__init__()

        # Specialized uncertainty heads
        self.authority_head = AuthorityHead()      # Existing AG-SAR flow
        self.dispersion_head = DispersionHead()    # Semantic consistency
        self.confidence_head = ConfidenceHead()    # Logit-based overconfidence
        self.refusal_head = RefusalHead()          # Early-layer abstention signals
        self.coherence_head = CoherenceHead()      # Cross-layer trajectory

        # Learned aggregation (or task-conditioned gating)
        self.aggregator = nn.Linear(num_heads, 1)

    def forward(self, features: FeatureBank) -> torch.Tensor:
        scores = torch.stack([
            self.authority_head(features.attention_weights, features.prompt_length),
            self.dispersion_head(features.logits, features.embed_matrix),
            self.confidence_head(features.logits, features.h_attn, features.h_block),
            self.refusal_head(features.hidden_states[:8]),  # Early layers
            self.coherence_head(features.hidden_states),    # All layers
        ], dim=-1)

        return self.aggregator(scores).squeeze(-1)
```

**Training Strategy:**
1. **Stage 1**: Train individual heads on their respective signals (unsupervised)
2. **Stage 2**: Train aggregator weights on labeled hallucination data
3. **Stage 3**: Fine-tune end-to-end with task-specific weighting

**References:**
- [HaluNet (2025)](https://arxiv.org/html/2512.24562): Multi-granular uncertainty with probabilistic confidence + distributional uncertainty
- [MHAD (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/0929.pdf): Multi-layer hallucination awareness via neuron selection
- [Critical Hallucination Heads](https://www.emergentmind.com/topics/critical-hallucination-heads): Identifying specific attention heads linked to hallucination

---

### 2.3 LLM-Specific Feature Enrichment

**Current Limitation:** MLP divergence (cosine similarity between h_attn and h_block) is a coarse signal. It captures "MLP override" but not the semantic nature of that override.

**Proposed Replacements/Augmentations:**

#### 2.3.1 Truthfulness Direction Probe

**Concept:** Extract the "truthfulness direction" from hidden states using contrastive learning.

```
     Truthful Examples          Hallucinated Examples
     ──────────────────         ─────────────────────
     h(correct_response)        h(hallucinated_response)
              │                          │
              └──────────┬───────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Truthfulness Probe  │
              │  (linear classifier) │
              │  v_truth = h · w     │
              └─────────────────────┘
                         │
                         ▼
              truthfulness_score ∈ [0, 1]
```

**Implementation:**
```python
class TruthfulnessProbe(nn.Module):
    """Linear probe trained on truthful vs hallucinated hidden states."""

    def __init__(self, hidden_size: int, layer_idx: int = -1):
        super().__init__()
        self.layer_idx = layer_idx  # Which layer to probe (default: last)
        self.probe = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: Dict[int, torch.Tensor]) -> torch.Tensor:
        h = hidden_states[self.layer_idx]  # (B, S, D)
        return torch.sigmoid(self.probe(h)).squeeze(-1)  # (B, S)
```

**Training Data Sources:**
- TruthfulQA for general truthfulness
- Task-specific contrastive pairs from HaluEval/RAGTruth

**References:**
- [SAPLMA (Azaria & Mitchell, 2023)](https://arxiv.org/abs/2304.13734): Predicting correctness from hidden states
- [Truthfulness Direction (Marks & Tegmark, 2024)](https://arxiv.org/abs/2310.06824): Linear representation of truth in LLMs
- [TSV Framework](https://arxiv.org/html/2503.01917): Truthfulness Separator Vector for enhanced separation

#### 2.3.2 Layer-wise Semantic Dynamics (LSD)

**Concept:** Track how hidden state semantics evolve across layers. Hallucinations show different "trajectories" than factual generations.

```
Layer 0 → Layer 8 → Layer 16 → Layer 24 → Layer 31
   │         │          │           │          │
   h₀        h₈        h₁₆         h₂₄        h₃₁
   │         │          │           │          │
   └─────────┴──────────┴─────┬─────┴──────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Trajectory Encoder │
                    │ (1D CNN or LSTM)   │
                    └─────────┬─────────┘
                              │
                              ▼
                    semantic_trajectory_score
```

**Implementation:**
```python
class LayerSemanticDynamics(nn.Module):
    """Analyze hidden state trajectory across layers."""

    def __init__(self, hidden_size: int, num_layers: int, sample_layers: int = 8):
        super().__init__()
        # Sample every N layers
        self.layer_indices = torch.linspace(0, num_layers-1, sample_layers).long()

        # Trajectory encoder (1D CNN over layer dimension)
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_size, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, hidden_states: Dict[int, torch.Tensor]) -> torch.Tensor:
        # Stack hidden states at sampled layers: (B, S, num_layers, D)
        stacked = torch.stack([hidden_states[i] for i in self.layer_indices.tolist()], dim=2)
        B, S, L, D = stacked.shape

        # Reshape for 1D CNN: (B*S, D, L)
        x = stacked.view(B * S, L, D).transpose(1, 2)

        # Encode trajectory
        encoded = self.encoder(x).squeeze(-1)  # (B*S, 128)

        # Classify
        score = torch.sigmoid(self.classifier(encoded))  # (B*S, 1)
        return score.view(B, S)
```

**References:**
- [LSD (2025)](https://arxiv.org/html/2510.04933v1): Achieves F1=0.92, AUROC=0.96 on TruthfulQA
- [Layer-wise Information Deficiency (2024)](https://arxiv.org/abs/2412.10246v2): Tracking cross-layer information dynamics

#### 2.3.3 Spectral Attention Features

**Concept:** Treat attention maps as graph Laplacians and extract spectral features.

```python
class SpectralAttentionFeatures(nn.Module):
    """Extract spectral features from attention graphs."""

    def __init__(self, num_eigenvalues: int = 10):
        self.k = num_eigenvalues

    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        # attention_weights: (B, H, S, S)
        B, H, S, _ = attention_weights.shape

        # Average over heads
        attn = attention_weights.mean(dim=1)  # (B, S, S)

        # Compute graph Laplacian: L = D - A
        degree = attn.sum(dim=-1, keepdim=True)  # (B, S, 1)
        laplacian = torch.diag_embed(degree.squeeze(-1)) - attn

        # Compute eigenvalues (top-k smallest)
        eigenvalues = torch.linalg.eigvalsh(laplacian)  # (B, S)

        # Use first k eigenvalues as features
        return eigenvalues[:, :self.k]  # (B, k)
```

**References:**
- [LapEigvals](https://arxiv.org/html/2502.17598): AUCROC 88.9% on TriviaQA using spectral features
- [HSAD (2025)](https://arxiv.org/html/2509.13154): FFT-based spectral analysis of hidden layer signals

---

### 2.4 Cross-Layer Attention Probing (CLAP Integration)

**Concept:** Instead of only using the last 4 layers, process activations across the entire residual stream as a joint sequence.

```
Layer 0   Layer 8   Layer 16   Layer 24   Layer 31
  │         │          │          │          │
  h₀        h₈        h₁₆        h₂₄        h₃₁
  │         │          │          │          │
  └─────────┴──────────┴──────────┴──────────┘
                       │
           ┌───────────▼───────────┐
           │ Cross-Layer Attention  │
           │ (Transformer Encoder)  │
           │ Query: response tokens │
           │ Keys: all layer states │
           └───────────┬───────────┘
                       │
                       ▼
              cross_layer_score
```

**Key Insight:** Different layers contribute differently to different tasks:
- **Early layers (0-8)**: Perception, basic semantics
- **Middle layers (8-20)**: Factual retrieval
- **Late layers (20-32)**: Decision consolidation

**Implementation:**
```python
class CrossLayerProbe(nn.Module):
    """CLAP-style cross-layer attention probing."""

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int = 8):
        super().__init__()
        self.layer_embed = nn.Embedding(num_layers, hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: Dict[int, torch.Tensor],  # layer_idx -> (B, S, D)
        response_mask: torch.Tensor,              # (B, S) response token positions
    ) -> torch.Tensor:
        B, S, D = list(hidden_states.values())[0].shape
        num_layers = len(hidden_states)

        # Stack all layer states: (B, S, L, D)
        stacked = torch.stack([hidden_states[i] for i in sorted(hidden_states.keys())], dim=2)

        # Add layer embeddings
        layer_ids = torch.arange(num_layers, device=stacked.device)
        stacked = stacked + self.layer_embed(layer_ids).unsqueeze(0).unsqueeze(0)

        # Reshape for cross-attention: (B*S, L, D)
        stacked = stacked.view(B * S, num_layers, D)

        # Query = mean of response tokens across layers
        query = stacked.mean(dim=1, keepdim=True)  # (B*S, 1, D)

        # Cross-attention: query attends to all layer representations
        out, _ = self.cross_attn(query, stacked, stacked)  # (B*S, 1, D)

        # Classify
        score = torch.sigmoid(self.classifier(out.squeeze(1)))  # (B*S, 1)
        return score.view(B, S)
```

**References:**
- [CLAP (2024)](https://arxiv.org/abs/2509.09700): Cross-Layer Attention Probing for fine-grained hallucination detection

---

### 2.5 Semantic Entropy Proxy (SEP Integration)

**Concept:** Replace expensive multi-sample semantic entropy with a learned probe on hidden states.

**Current:** Semantic Dispersion approximates this but uses embedding distance, not true semantic equivalence.

**Improved:** Train a probe to predict semantic entropy labels:

```python
class SemanticEntropyProxy(nn.Module):
    """Learned proxy for semantic entropy without sampling."""

    def __init__(self, hidden_size: int, num_layers: int = 4):
        super().__init__()
        # Use last N layers
        self.probe = nn.Sequential(
            nn.Linear(hidden_size * num_layers, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # Concatenate last N layers
        concat = torch.cat(hidden_states[-4:], dim=-1)  # (B, S, D*4)
        return torch.sigmoid(self.probe(concat)).squeeze(-1)  # (B, S)
```

**Training:**
1. Generate semantic entropy labels via expensive multi-sample method
2. Train probe to predict these labels from single-pass hidden states
3. Deploy probe for zero-latency inference

**References:**
- [Semantic Entropy Probes (2024)](https://arxiv.org/abs/2406.15927): Cheap approximation of SE via linear probes

---

## 3. Unified v9.0 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AG-SAR v9.0 Engine                               │
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │  Task Detector   │───▶│ Task-Adaptive    │───▶│  Calibration     │   │
│  │  (auto/manual)   │    │ Config Router    │    │  (per-task T, α) │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                       Feature Extraction                          │   │
│  │  • Attention Weights (all layers)                                 │   │
│  │  • Hidden States (all layers)                                     │   │
│  │  • h_attn, h_block (MLP outputs)                                  │   │
│  │  • Logits, Embeddings                                             │   │
│  └────────────────────────────────┬─────────────────────────────────┘   │
│                                   │                                      │
│  ┌────────────────────────────────▼─────────────────────────────────┐   │
│  │                    Multi-Head Uncertainty                         │   │
│  │                                                                   │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │   │
│  │  │Authority│ │ Truth   │ │Semantic │ │  Layer  │ │ Cross-  │     │   │
│  │  │  Flow   │ │ Probe   │ │Entropy  │ │Dynamics │ │ Layer   │     │   │
│  │  │ (v8.0)  │ │ (new)   │ │ Proxy   │ │  (LSD)  │ │  Probe  │     │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘     │   │
│  │       │           │           │           │           │          │   │
│  │       └───────────┴───────────┴─────┬─────┴───────────┘          │   │
│  │                                     │                             │   │
│  │                    ┌────────────────▼────────────────┐            │   │
│  │                    │    Learned Aggregator (MLP)     │            │   │
│  │                    │    + Task-Conditioned Gating    │            │   │
│  │                    └────────────────┬────────────────┘            │   │
│  └─────────────────────────────────────┼─────────────────────────────┘   │
│                                        │                                 │
│  ┌─────────────────────────────────────▼─────────────────────────────┐   │
│  │                     Conservative Aggregation                       │   │
│  │   u_final = percentile(u_tokens, α_task)  # α varies by task       │   │
│  └─────────────────────────────────────┬─────────────────────────────┘   │
│                                        │                                 │
│  ┌─────────────────────────────────────▼─────────────────────────────┐   │
│  │                    Post-hoc Temperature Scaling                    │   │
│  │   calibrated = sigmoid(logit(u_final) / T_task)                    │   │
│  └─────────────────────────────────────┬─────────────────────────────┘   │
│                                        │                                 │
│                                        ▼                                 │
│                               uncertainty_score                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Roadmap

### Phase 1: Feature Enrichment (Week 1-2)
1. Extend `ModelAdapter` to capture all layer hidden states
2. Implement `TruthfulnessProbe` with linear classifier
3. Implement `LayerSemanticDynamics` with 1D CNN
4. Add to `engine.py` as optional components

### Phase 2: Multi-Head Architecture (Week 2-3)
1. Define `FeatureBank` dataclass for feature collection
2. Implement individual uncertainty heads
3. Create `MultiHeadUncertainty` module with learned aggregator
4. Train aggregator weights on labeled data

### Phase 3: Task-Adaptive Calibration (Week 3-4)
1. Implement `TaskDetector` with heuristic rules
2. Create `TaskCalibrationRouter` with per-task parameters
3. Learn optimal parameters via grid search on dev sets
4. Integrate with existing temperature scaling

### Phase 4: Integration & Evaluation (Week 4-5)
1. Unify all components into `AGSARv9Engine`
2. Benchmark on all 4 datasets with ablations
3. Analyze per-component contribution
4. Tune hyperparameters

---

## 5. Expected Improvements

### Baseline (v8.0)
| Dataset | AUROC | TPR@5%FPR |
|---------|-------|-----------|
| HaluEval QA | 0.903 | 0.422 |
| RAGTruth | 0.704 | 0.144 |
| HaluEval Summ | 0.643 | 0.104 |
| FAVA | 0.618 | 0.076 |

### Target (v9.0)
| Dataset | AUROC (target) | TPR@5%FPR (target) |
|---------|----------------|---------------------|
| HaluEval QA | 0.92+ | 0.50+ |
| RAGTruth | 0.80+ | 0.25+ |
| HaluEval Summ | 0.75+ | 0.20+ |
| FAVA | 0.70+ | 0.15+ |

**Justification:**
- Multi-head uncertainty captures diverse error types
- Task-adaptive calibration prevents hyperparameter mismatch
- LSD trajectory analysis improves summarization (cross-layer coherence)
- Truthfulness probes add direct supervision signal

---

## 6. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Increased latency from multi-head | High | Parallelize heads; prune low-impact ones |
| Overfitting on training data | Medium | Cross-validation; held-out calibration sets |
| Task detector errors | Medium | Allow manual override; confidence thresholds |
| OOD generalization | High | Include diverse training data; regularization |

---

## 7. References

### Core Papers
1. [Semantic Entropy Probes (2024)](https://arxiv.org/abs/2406.15927) - Cheap SE approximation
2. [CLAP (2024)](https://arxiv.org/abs/2509.09700) - Cross-layer attention probing
3. [LSD (2025)](https://arxiv.org/html/2510.04933v1) - Layer-wise semantic dynamics
4. [HaluNet (2025)](https://arxiv.org/html/2512.24562) - Multi-granular uncertainty
5. [LLM-Check (NeurIPS 2024)](https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection) - Attention + hidden score
6. [MHAD (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/0929.pdf) - Multi-layer hallucination awareness
7. [Thermometer (ICML 2024)](https://research.ibm.com/publications/thermometer-towards-universal-calibration-for-large-language-models) - Universal calibration

### Truthfulness Direction
8. [SAPLMA (Azaria & Mitchell, 2023)](https://arxiv.org/abs/2304.13734) - Hidden state correctness prediction
9. [TSV Framework (2025)](https://arxiv.org/html/2503.01917) - Truthfulness separator vector
10. [Critical Hallucination Heads](https://www.emergentmind.com/topics/critical-hallucination-heads) - Attention head analysis

### Spectral Methods
11. [LapEigvals (2025)](https://arxiv.org/html/2502.17598) - Spectral attention features
12. [HSAD (2025)](https://arxiv.org/html/2509.13154) - FFT-based spectral analysis

---

## 8. Conclusion

AG-SAR v9.0 proposes a significant architectural evolution:

1. **Task-Adaptive Calibration** addresses the hyperparameter mismatch problem
2. **Multi-Head Uncertainty** captures diverse hallucination types
3. **LLM-Specific Features** add truthfulness probes and layer dynamics
4. **Cross-Layer Probing** enables layer-wise importance weighting

These improvements target the fundamental limitation of v8.0: a single Authority Flow mechanism cannot capture all hallucination patterns across diverse tasks. By combining multiple complementary signals with task-aware aggregation, v9.0 can become a truly universal hallucination detector.
