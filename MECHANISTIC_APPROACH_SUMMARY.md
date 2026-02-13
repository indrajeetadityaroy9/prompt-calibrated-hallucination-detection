# Mechanistic Hallucination Detection: Research Summary

## Key Research (from transformer-circuits.pub)

### 1. RAGLens: SAE-Based Detection
- **Paper**: [Toward Faithful RAG with Sparse Autoencoders](https://arxiv.org/abs/2512.08892)
- **Key Finding**: Mid-layer SAE features (layer ~19 for Llama-3.1-8B) are most informative
- **Method**: Train GAM on SAE features with mutual information feature selection
- **Limitation**: Requires labeled training data

### 2. INSIDE/EigenScore
- **Paper**: [LLMs' Internal States Retain the Power of Hallucination Detection](https://arxiv.org/abs/2402.03744)
- **Key Finding**: Eigenvalues of response embedding covariance measure consistency
- **Method**: Sample K responses, compute covariance, use eigenvalues
- **Limitation**: Requires multiple samples (K forward passes)

### 3. Pre-trained SAEs Available
- **Llama Scope**: [fnlp/Llama-Scope](https://huggingface.co/fnlp/Llama-Scope) - 256 SAEs for all layers
- **EleutherAI**: [EleutherAI/sae-llama-3-8b-32x](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x)
- **Goodfire**: [Goodfire/Llama-3.1-8B-SAE](https://huggingface.co/Goodfire)

## Experimental Results

| Approach | QA AUROC | Summarization AUROC | Task-Agnostic? |
|----------|----------|---------------------|----------------|
| Uniform Noisy-OR | 0.87 | 0.50 | Yes |
| Task-Specific | 0.93 | 0.74 | No |
| Self-Adaptive (SA-NOR) | 0.93 | 0.57 | Yes |
| Fisher Score | 0.85 | 0.62 | Yes |
| Mechanistic (mid-layer) | 0.73 | 0.56 | Yes |
| Unified (all signals) | 0.85 | 0.57 | Yes |

## Key Insight

For **zero-shot, single-pass (forced decoding) detection**, we face fundamental limitations:

1. **No labeled data** → Can't learn which SAE features correlate with hallucination
2. **Single pass** → Can't use self-consistency methods (EigenScore, Semantic Entropy)
3. **Task difference** → Summarization involves legitimate creative processing

The research clearly shows that the highest-performing methods require either:
- Labeled training data (RAGLens achieves ~0.85+ across tasks)
- Multiple samples (EigenScore, Semantic Entropy)

## Paths Forward

### Option 1: SAE Feature Probing (Recommended)
```python
# Install: pip install sparsify
from sparsify import Sae

# Load pre-trained SAE for layer 19
sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.19")

# Extract features
features = sae.encode(hidden_states)  # Sparse features

# Train lightweight probe on features (requires labeled data)
# RAGLens shows ~1000 features selected via MI are sufficient
```

### Option 2: Multi-Sample Detection
```python
# Generate K responses
responses = [model.generate(prompt) for _ in range(K)]

# Compute EigenScore
embeddings = [get_embedding(r) for r in responses]
cov = np.cov(embeddings)
eigenvalues = np.linalg.eigvalsh(cov)
risk = np.mean(np.log(eigenvalues))  # High = inconsistent = risky
```

### Option 3: Accept Task-Specific Weights
For production use, task-specific weights achieve better results:
- QA: Weight probability signals (entropy, margin) highly
- Summarization: Weight hidden-state signals (norms) highly

## Implementation Status

Created files:
- `ag_sar/signals/sae_features.py` - SAE feature extraction framework
- `ag_sar/aggregation/fisher_aggregator.py` - Data-driven signal weighting
- `scripts/evaluate_mechanistic.py` - Mid-layer feature evaluation
- `scripts/evaluate_unified.py` - Unified signal evaluation
- `scripts/evaluate_fisher_aggregator.py` - Fisher score evaluation

## Conclusion

True task-agnostic, zero-shot, single-pass detection that matches task-specific performance (0.93 QA, 0.74 Summarization) is not achievable with current methods. The transformer-circuits research points to SAE features as the key, but extracting the relevant features requires either:

1. **Labeled data** to learn feature-hallucination correlations
2. **Multiple samples** to measure self-consistency
3. **Pre-computed feature annotations** (interpretable SAE feature labels)

For ICML, recommend:
- Present Fisher Score + adaptive weighting as principled baseline
- Acknowledge the fundamental limitation of zero-shot detection
- Propose SAE-based probing as future work with labeled data
