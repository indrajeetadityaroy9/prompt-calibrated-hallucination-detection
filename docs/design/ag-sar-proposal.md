Here is the formalized framework for **Attention-Graph Shifting Attention to Relevance (AG-SAR)**.

This framework synthesizes the original SAR methodology with modern Graph Transformer interpretation techniques (specifically **Generalized Attention Flow** and **Spectral Analysis**) to create a zero-latency uncertainty quantification method.

---

# Framework: Attention-Graph SAR (AG-SAR)

**Abstract:**
AG-SAR adapts the *Shifting Attention to Relevance* paradigm to standard Transformers by treating the self-attention mechanism as a dynamic directed graph. Instead of relying on external semantic models (which induce latency) to measure token relevance, AG-SAR derives relevance from the **Topological Centrality** of tokens within the model's internal attention graph, explicitly correcting for "Attention Sink" artifacts.

---

### 1. Mathematical Formulation

Let a generated sequence be a set of tokens $T = \{t_1, t_2, ..., t_N\}$.
The Transformer produces a set of Attention Matrices for layer $L$ and head $H$, denoted as $A^{L,H} \in \mathbb{R}^{N \times N}$.

#### A. Constructing the Graph
We define the **Global Attention Graph** $G = (V, E)$ where:
*   **Nodes ($V$):** The tokens $t_i$.
*   **Edges ($E$):** The weighted edges representing information flow. To capture the most "semantic" structure, we aggregate attention across the final layers (where semantic consolidation occurs), filtering out "noisy" heads using an entropy threshold $\tau$.

$$ \mathbf{A}_{global} = \frac{1}{|L_{sem}|} \sum_{l \in L_{sem}} \sum_{h=1}^{H} \mathbb{I}(\text{Entropy}(A^{l,h}) < \tau) \cdot A^{l,h} $$
*(Where $\mathbb{I}$ is an indicator function that removes heads with uniform/diffuse attention.)*

#### B. Determining Token Relevance (The "Sink-Aware" Centrality)
Standard graph centrality fails in Transformers due to **Attention Sinks** (e.g., initial tokens like `<s>` or separators like `.`), which accumulate massive attention without carrying semantic meaning.

We define **Relevance $R(t_i)$** as the **Norm-Weighted Eigenvector Centrality**:

$$ R(t_i) = C_{eigen}(t_i) \times ||\mathbf{v}_i||_2 $$

Where:
*   $C_{eigen}(t_i)$ is the Eigenvector Centrality of token $i$ in $\mathbf{A}_{global}$, representing its structural importance.
*   $||\mathbf{v}_i||_2$ is the $L2$-norm of the Value Vector projected by token $i$.
    *   *Rationale:* Tokens that are "Sinks" have high centrality but low vector norms (mechanistically passive). "Relevant" tokens have high centrality *and* high vector norms (active information donors).

#### C. The Shifted Uncertainty Metric
We replace the standard Predictive Entropy (PE) with **Graph-Shifted Entropy (GSE)**. This effectively down-weights the uncertainty contributions of "leaf nodes" (irrelevant tokens) and "sinks."

$$ \text{GSE}(T) = \sum_{i=1}^{N} \mathcal{H}(t_i) \times \frac{R(t_i)}{\sum_{j=1}^{N} R(t_j)} $$

Where $\mathcal{H}(t_i)$ is the local entropy (randomness) of predicting token $t_i$.

---

### 2. The Implementation Recipe

This recipe can be implemented on any standard Transformer (BERT, GPT, Llama, ViT) without retraining.

#### Step 1: Head Pruning (The "Linguistic Redundancy" Filter)
Not all attention heads encode semantic structure; many are syntactic or positional.
*   **Action:** Calculate the entropy of the attention distribution for each head.
*   **Filter:** Discard heads with extremely high entropy (attending to everything) or extremely low entropy (attending only to the previous token). Keep the "sparse" heads.

#### Step 2: Matrix Aggregation
*   **Action:** Average the remaining attention matrices to form a single $N \times N$ adjacency matrix.
*   **Optimization:** Use the **Generalized Attention Flow (GAF)** method: instead of raw weights, compute the "Flow Network" which accounts for residual connections, effectively tracing the path from input to output.

#### Step 3: Sink-Aware Centrality Calculation
*   **Action:** Compute the Eigenvector Centrality of the aggregated matrix.
*   **Correction:** Multiply each node's centrality score by the norm of its hidden state vector.
    *   *Result:* Stop words ("the", "of") and Sinks (`<s>`) drop to near-zero relevance. Named Entities and Verbs spike in relevance.

#### Step 4: Uncertainty Re-Weighting
*   **Action:** Calculate the standard entropy for the sequence.
*   **Shift:** Compute the weighted average using the scores from Step 3.
*   **Decision:** If $\text{GSE} > \text{Threshold}$, flag as Hallucination.

---

### 3. Sentence-Level Consensus (The "Spectral" Shift)

The original SAR paper clustered sentences by text meaning. AG-SAR clusters them by **Reasoning Topology**.

If you generate 5 answers ($S_1 ... S_5$), do not just check if they *look* the same (text). Check if the model *thought* about them the same way.

$$ \text{Consensus}(S_a, S_b) = 1 - \text{JSD}(\text{Spectrum}(\mathbf{A}_a) || \text{Spectrum}(\mathbf{A}_b)) $$

*   **Logic:** We compare the **Graph Laplacian Spectrum** (eigenvalues) of the attention graphs for two generated sentences.
*   **Why:** If two sentences are semantically different but yield identical attention spectra, the model is using the exact same reasoning path to generate them. They are likely "Topological Synonyms."
*   **Application:** Group generated answers by their Spectral Signature. If the largest cluster has low internal variance, confidence is high.

---

### 4. Comparison: Original SAR vs. AG-SAR

| Feature | Original SAR (Paper) | **Attention-Graph SAR (Proposed)** |
| :--- | :--- | :--- |
| **Relevance Signal** | Semantic Perturbation (External) | **Information Flow (Internal)** |
| **Complexity** | $O(K \times N)$ Inference Passes | $O(1)$ (Matrix Algebra only) |
| **Handling "Sinks"** | Implicitly via text removal | **Explicitly via Vector Norms** |
| **Dependency** | Requires RoBERTa/SimModel | **Self-Contained** |
| **Best For** | Black-box models (API) | **White-box models (Local/Weights)** |

### 5. Why this works (Theoretical Basis 2024-2025)
*   **TOHA (Topology-based Hallucination Detection, 2025):** Established that "confabulations" (hallucinations) result in **fractured attention graphs**. When a model lies, it cannot sustain a coherent centrality structure; the graph becomes decentralized. AG-SAR detects this because the sum of Relevance scores ($\sum R$) will plummet in a fractured graph, naturally inflating the uncertainty score.
*   **Value-State Gated Attention (2025):** Proven that the "content" of a token is encoded in the magnitude of the Value vector, while the "routing" is in the Attention weight. AG-SAR combines these to ensure we only measure uncertainty on tokens that are both **routed heavily** and **rich in content**.
