# Tiny Aya Under The Hood: Cross-Lingual Embedding Alignment Analysis

**A Mechanistic Interpretability Study of Multilingual Representation Emergence in Tiny Aya Global**

---

## Abstract

This document provides a comprehensive account of the eight-notebook analysis pipeline implemented in this repository. The goal is to identify *which layers* in Tiny Aya Global (3.35B parameters, 4 transformer layers) produce **language-agnostic representations** -- layers where the model transitions from encoding surface-level language identity ("this is Spanish") to encoding abstract semantic content ("this is the concept of X"). This is a **mechanistic interpretability** task, not a training task. We employ Centered Kernel Alignment (CKA), hierarchical clustering, anisotropy-corrected similarity, translation retrieval metrics, script-based decomposition, and regional model comparison across 13 languages spanning 5 language families and 6 writing scripts.

This work is directly aligned with the Linear issue for cross-lingual embedding alignment analysis (TIN-7) and builds upon the foundation laid by the [Wayy-Research/project-aya](https://github.com/Wayy-Research/project-aya/tree/dev/notebooks) notebooks -- specifically `02_cka_analysis.ipynb` (CKA tutorial and noise sensitivity) and `07_teacher_representations.ipynb` (layer-wise activation extraction and self-similarity heatmaps). Our analysis is the natural next step: going from *self-similarity* (how a model's own layers relate to each other) to *cross-lingual similarity* (how different languages relate to each other at each layer).

---

## Table of Contents

1. [Motivation and Research Question](#1-motivation-and-research-question)
2. [Background: The Three-Stage Hypothesis](#2-background-the-three-stage-hypothesis)
3. [Model and Data](#3-model-and-data)
4. [Notebook 01: Data Preparation](#4-notebook-01-data-preparation)
5. [Notebook 02: Activation Extraction](#5-notebook-02-activation-extraction)
6. [Notebook 03: Cross-Lingual CKA (Core Analysis)](#6-notebook-03-cross-lingual-cka)
7. [Notebook 04: Language Family Clustering (Novel Technique 1)](#7-notebook-04-language-family-clustering)
8. [Notebook 05: Anisotropy and Whitened CKA (Novel Technique 2)](#8-notebook-05-anisotropy-and-whitened-cka)
9. [Notebook 06: Retrieval Alignment (Novel Technique 3)](#9-notebook-06-retrieval-alignment)
10. [Notebook 07: Script Decomposition (Novel Technique 4)](#10-notebook-07-script-decomposition)
11. [Notebook 08: Regional Comparison (Novel Technique 5)](#11-notebook-08-regional-comparison)
12. [Synthesis: What the Full Pipeline Reveals](#12-synthesis)
13. [Alignment with the Linear Issue (TIN-7)](#13-alignment-with-tin-7)
14. [References](#14-references)

---

## 1. Motivation and Research Question

Recent mechanistic interpretability work has converged on a striking hypothesis about multilingual language models: they process input through three distinct stages (Wendler et al., 2024; Dumas et al., 2025; Harrasse et al., 2025):

1. **Language-specific encoding** (early layers): Input tokens are mapped into initial representations that retain strong language identity -- tokenization artifacts, script-specific patterns, and morphological structure.
2. **Language-agnostic processing** (middle layers): Representations converge into a shared semantic space where the same concept expressed in different languages occupies nearby positions. The model "thinks" in an abstract conceptual language.
3. **Language-specific decoding** (late layers): The model maps from the shared space back to target-language-specific token predictions.

**Our central question**: *Where does stage 2 emerge in Tiny Aya Global, and how complete is the convergence?*

Tiny Aya is particularly interesting because:
- It has only **4 transformer layers** (compared to 32-36 in typical LLMs), compressing the entire representational pipeline into a very shallow architecture.
- It covers **70+ languages** with only 3.35B parameters, meaning cross-lingual sharing is not optional -- it is structurally necessary.
- It has **regional variants** (South Asia, Africa, Europe), enabling direct comparison of how specialization affects universality.

The practical implications are significant: identifying universal vs. specialized layers enables targeted interventions such as representation steering, informed adapter placement, safe compression/pruning decisions, and efficient parameter sharing across regional models.

---

## 2. Background: The Three-Stage Hypothesis

### 2.1 Evidence from Prior Work

Wendler et al. (2024) ("Do Llamas Work in English?") used the **logit lens** -- a technique that projects intermediate hidden states through the language model head to see what tokens the model would predict at each layer -- and found that Llama-2 predicts English tokens at intermediate layers regardless of the input language, before switching to target-language tokens in the final layers. This suggests an English-centric (or at minimum language-agnostic) internal representation.

Dumas et al. (2025) ("Separating Tongue from Thought") went further with **activation patching**: they swapped hidden states between parallel prompts in different languages at specific layers and measured whether the model still produced correct output. They found that mid-layer activations are interchangeable across languages, providing **causal** evidence for language-agnostic concept representations.

Harrasse et al. (2025) used **Cross-Layer Transcoders (CLTs)** and attribution graphs to trace multilingual processing, confirming that all languages converge to a shared representation in middle layers while language-specific decoding emerges in later layers.

### 2.2 The CKA Framework

Our primary tool is **Centered Kernel Alignment (CKA)** (Kornblith et al., 2019), which measures representational similarity between two sets of neural network activations. Unlike Canonical Correlation Analysis (CCA), CKA is:
- Invariant to orthogonal transformations (rotation of the representation space)
- Invariant to isotropic scaling
- Reliable even when the feature dimension exceeds the number of data points

This makes it ideal for comparing representations across languages, where the representations live in the same dimensional space but may be rotated or scaled versions of each other.

### 2.3 Connection to the Wayy-Research Notebooks

The [project-aya](https://github.com/Wayy-Research/project-aya) repository established:
- **Notebook 02** (`02_cka_analysis.ipynb`): A CKA tutorial establishing that CKA = 1.0 means identical representations, CKA near 0 means uncorrelated. It introduced `MinibatchCKAAccumulator` and `compute_layerwise_cka` utilities and established the 0.75 threshold for "acceptable alignment."
- **Notebook 07** (`07_teacher_representations.ipynb`): Used `ActivationStore` + `register_teacher_hooks` to extract layer-wise activations, sampled every 3rd layer across 10 languages, and produced self-similarity CKA heatmaps. The key question asked was: "do languages cluster?"

Our cross-lingual module is the **next step** after notebook 07 -- transitioning from *self-similarity* (which layers of the same model are similar to each other) to *cross-language similarity* (are Hindi and English processed similarly at layer L?). The project-aya results showed cross-lingual similarity of 0.878 at the teacher level, with attention-to-SSM CKA of 0.604 -- below the 0.75 threshold, indicating that refinement is needed after weight mapping.

---

## 3. Model and Data

### 3.1 Tiny Aya Global

| Property | Value |
|---|---|
| Model ID | `CohereLabs/tiny-aya-global` |
| Parameters | 3.35B |
| Architecture | CohereForCausalLM (decoder-only) |
| Transformer Layers | 4 (3 sliding-window attention + 1 global attention) |
| Hidden Dimension | 3072 |
| Languages | 70+ |
| Tokenizer | CohereTokenizer (BPE) |

The shallow depth (4 layers) is both a challenge and an opportunity: the three-stage hypothesis must compress into very few layers, making transitions between stages more abrupt and potentially easier to detect.

### 3.2 FLORES+ Parallel Corpus

We use the **FLORES+** benchmark (`openlanguagedata/flores_plus`) -- the actively maintained successor to FLORES-200, containing 1,012 professionally translated sentences across 228+ language varieties. The `devtest` split provides guaranteed semantic equivalence across all languages, which is essential for controlled cross-lingual analysis.

### 3.3 Language Selection

We analyze 13 languages chosen to maximize diversity along three axes:

| Language | ISO | Script | Family | Resource Level |
|---|---|---|---|---|
| English | en | Latin | Indo-European | High |
| Spanish | es | Latin | Indo-European | High |
| French | fr | Latin | Indo-European | High |
| German | de | Latin | Indo-European | High |
| Arabic | ar | Arabic | Afro-Asiatic | High |
| Hindi | hi | Devanagari | Indo-European | Mid |
| Bengali | bn | Bengali | Indo-European | Mid |
| Tamil | ta | Tamil | Dravidian | Mid |
| Turkish | tr | Latin | Turkic | Mid |
| Persian | fa | Arabic | Indo-European | Mid |
| Swahili | sw | Latin | Niger-Congo | Low |
| Amharic | am | Ge'ez | Afro-Asiatic | Low |
| Yoruba | yo | Latin | Niger-Congo | Low |

This selection spans **5 language families**, **6 writing scripts**, and **3 resource tiers**, enabling us to disentangle multiple confounds: are similar representations due to shared vocabulary (same script), shared grammar (same family), or true semantic convergence?

---

## 4. Notebook 01: Data Preparation

### 4.1 Purpose

Establish the data foundation by loading the FLORES+ parallel corpus, validating alignment, computing corpus statistics, and visualizing language metadata.

### 4.2 Procedure

1. Load FLORES+ `devtest` split (1,012 sentences) for all 13 languages via the HuggingFace `datasets` library with gated authentication.
2. Validate parallel alignment: all languages must have exactly 1,012 sentences, with sentence `i` in language A being the translation of sentence `i` in language B.
3. Compute corpus statistics: average character length, word count, and their distributions.
4. Visualize language metadata: family and script distributions.

### 4.3 Key Observations

The corpus statistics reveal **tokenizer fertility disparities**:
- French, Spanish, and German have the highest average character lengths (~152-156 characters), reflecting analytical morphology and longer word forms.
- Amharic has the lowest average character length (~86 characters), reflecting the compact Ge'ez script.
- Tamil has relatively few words (~16.6 per sentence) but long character lengths (~152), reflecting agglutinative morphology where single words carry multiple morphemes.

These disparities matter because BPE tokenization interacts differently with each script: Latin-script languages share more subword tokens, which could inflate similarity scores at early layers through token-surface matching rather than semantic alignment.

### 4.4 Mechanistic Interpretability Perspective

The corpus statistics foreshadow a critical confound: if two languages use the same script (and thus share BPE subwords), their early-layer representations may be similar for purely tokenization-related reasons, not because the model has learned to represent meaning similarly. This motivates the script decomposition analysis in Notebook 07.

---

## 5. Notebook 02: Activation Extraction

### 5.1 Purpose

Extract sentence-level embeddings from every transformer layer for every language, producing the raw representational data that all subsequent analyses consume.

### 5.2 Mathematical Framework: Mean-Pooled Embeddings

For a sentence $s$ in language $\ell$ consisting of tokens $t_1, \ldots, t_T$ (after tokenization and padding), the hidden state at layer $l$ is:

$$h_l^{(i)} = \text{TransformerLayer}_l(h_{l-1}^{(i)}) \quad \text{for } i = 1, \ldots, T$$

The sentence embedding is obtained by **mean pooling** over non-padding tokens:

$$e_l^\ell(s) = \frac{1}{\sum_{i=1}^{T} m_i} \sum_{i=1}^{T} m_i \cdot h_l^{(i)}$$

where $m_i \in \{0, 1\}$ is the attention mask (1 for real tokens, 0 for padding).

Mean pooling is preferred over CLS-token extraction for decoder-only models because:
- Decoder-only models (like Tiny Aya/Cohere) do not have a dedicated [CLS] token.
- Mean pooling distributes information across all positions, making the embedding more robust to positional artifacts.
- It is the standard approach in multilingual alignment analysis (Artetxe & Schwenk, 2019).

### 5.3 Implementation via Forward Hooks

The `ActivationStore` class uses PyTorch's `register_forward_hook` API to non-invasively capture hidden states:

```python
store = ActivationStore(detach=True, device="cpu")
register_model_hooks(model, store, layer_indices=[0, 1, 2, 3])

with torch.no_grad():
    model(**inputs)

activations = store.collect_mean_pooled()
# Result: {"layer_0": tensor(1012, 3072), "layer_1": ..., ...}
```

Activations are **detached** from the computation graph (no gradient tracking) and moved to **CPU** immediately to keep GPU memory free for the next batch. This design, inspired by `register_teacher_hooks` in the project-aya notebook 07, enables processing the full 1,012-sentence corpus per language without out-of-memory errors.

### 5.4 Output Structure

After extraction, we have a tensor of shape `(13 languages x 4 layers x 1012 sentences x 3072 hidden_dim)`. Each `layer_{idx}_{language}.pt` file stores one `(1012, 3072)` matrix, totaling 52 files (13 x 4).

### 5.5 Dimensionality Reduction Visualizations

The notebook produces PCA and t-SNE plots at each layer, colored by language. In a model with strong cross-lingual alignment:
- **Early layers**: Languages form distinct clusters (each language occupies a separate region of the embedding space).
- **Later layers**: Clusters merge (translations of the same sentence, regardless of language, are near each other).

PCA variance analysis tracks how many principal components are needed to explain 90% of variance at each layer. A decrease in this number across layers indicates that representations are being compressed into a lower-dimensional, more structured space -- consistent with the emergence of a shared conceptual representation.

---

## 6. Notebook 03: Cross-Lingual CKA (Core Analysis)

This is the **central notebook** of the pipeline. It computes the primary metric -- pairwise CKA between all language pairs at each layer -- and identifies the convergence layer.

### 6.1 Mathematical Foundation: Centered Kernel Alignment

Given two activation matrices $X \in \mathbb{R}^{n \times d_x}$ and $Y \in \mathbb{R}^{n \times d_y}$ (where $n$ = number of aligned sentences, $d$ = hidden dimension), CKA is defined as:

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

where $K = XX^\top$ and $L = YY^\top$ are kernel (Gram) matrices, and **HSIC** is the Hilbert-Schmidt Independence Criterion:

$$\text{HSIC}(K, L) = \frac{1}{n^2} \text{tr}(K_c L_c)$$

with $K_c = HKH$ being the centered Gram matrix and $H = I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ being the centering matrix.

**Linear CKA** uses linear kernels ($K = XX^\top$) and simplifies to:

$$\text{CKA}_\text{linear}(X, Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \cdot \|Y^\top Y\|_F}$$

This has $O(n \cdot d^2)$ complexity, making it efficient for our setting (n=1012, d=3072).

**RBF CKA** uses Gaussian kernels $K_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$ with the median heuristic for bandwidth selection. It captures nonlinear representational relationships at $O(n^2 \cdot d)$ cost.

**Interpretation**:
- CKA = 1.0: The two representation spaces are identical (up to orthogonal transformation and isotropic scaling).
- CKA near 0: The representations are unrelated.
- CKA > 0.75: "Acceptable alignment" -- the threshold established in notebook 02 of project-aya.

### 6.2 The Convergence Layer

For each layer $l$, we compute the **average cross-lingual CKA** -- the mean of all off-diagonal entries in the $13 \times 13$ CKA matrix:

$$\overline{\text{CKA}}_l = \frac{2}{n_\text{langs}(n_\text{langs} - 1)} \sum_{i < j} \text{CKA}(X_l^{(i)}, X_l^{(j)})$$

where $X_l^{(i)}$ is the activation matrix for language $i$ at layer $l$.

The **convergence layer** is defined as the first layer where $\overline{\text{CKA}}_l \geq 0.75$. This is the layer where the model transitions from language-specific to language-agnostic processing.

We also compute a 95% confidence interval:

$$\text{CI}_l = \overline{\text{CKA}}_l \pm 1.96 \cdot \frac{\sigma_l}{\sqrt{n_\text{pairs}}}$$

where $\sigma_l$ is the standard deviation of off-diagonal CKA scores and $n_\text{pairs} = \binom{13}{2} = 78$.

### 6.3 Statistical Significance via Permutation Tests

To verify that observed CKA scores are significantly above chance, we run **permutation tests** (Kornblith et al., 2019):

1. Compute the observed CKA score on aligned data: $\text{CKA}_\text{obs}(X, Y)$.
2. For $B$ permutations, randomly shuffle the rows of $Y$ (breaking alignment) and compute $\text{CKA}_b(X, Y_{\pi_b})$.
3. The p-value is $p = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}[\text{CKA}_b \geq \text{CKA}_\text{obs}]$.

A p-value < 0.05 confirms that the observed cross-lingual similarity is not an artifact of random alignment.

### 6.4 Visualizations

- **CKA Heatmaps**: One $13 \times 13$ heatmap per layer, showing pairwise similarities. Look for the off-diagonal becoming uniformly warm (high CKA) in later layers.
- **Convergence Curve**: Average CKA vs. layer depth, with confidence bands and the 0.75 threshold line. This is the **primary deliverable** -- the inflection point where cross-lingual convergence occurs.
- **Spaghetti Plot**: Per-pair CKA trajectories across layers. Highlights specific pairs (e.g., English-Hindi, English-Arabic, French-Spanish) to reveal whether convergence is uniform or pair-dependent.

### 6.5 Mechanistic Interpretability Perspective

The convergence curve directly visualizes the three-stage hypothesis for Tiny Aya. With only 4 layers:
- If convergence occurs at layer 0 or 1: The model achieves language-agnostic representations very early, suggesting that the embedding layer itself provides substantial cross-lingual alignment (possibly from shared BPE tokens).
- If convergence occurs at layer 2 or 3: Deeper processing is required to abstract away from surface features.
- If the threshold is never reached: The model may rely on partial overlap rather than a fully shared space, or the 4-layer architecture may be too shallow for complete convergence.

Recent literature (Nakai et al., 2025, "TRepLiNa") found that mid-layer alignment (roughly layers 10-15 of a 36-layer model) is most effective for cross-lingual transfer. For Tiny Aya with 4 layers, the "mid-layer" equivalent would be layers 1-2. This is the expected location of the convergence point.

---

## 7. Notebook 04: Language Family Clustering (Novel Technique 1)

### 7.1 Hypothesis

In early layers, languages from the same genetic family (e.g., Indo-European: English, Hindi, Bengali, Persian, German, French, Spanish) should cluster together because surface-level features dominate -- shared vocabulary roots, similar morphological patterns, and common grammatical structures. In deeper layers, if the model has achieved language-agnostic representations, **family-based clusters should dissolve**.

### 7.2 Mathematical Framework

#### Hierarchical Clustering

We convert the CKA similarity matrix at each layer into a distance matrix:

$$D_{ij}^{(l)} = 1 - \text{CKA}(X_l^{(i)}, X_l^{(j)})$$

and apply **Ward's method** for agglomerative hierarchical clustering. Ward's method minimizes the total within-cluster variance at each merge step:

$$\Delta(A, B) = \frac{n_A \cdot n_B}{n_A + n_B} \|\bar{d}_A - \bar{d}_B\|^2$$

where $\bar{d}_A$ and $\bar{d}_B$ are the centroids of clusters $A$ and $B$.

#### Cophenetic Correlation

Measures how faithfully the dendrogram preserves the original pairwise distances:

$$r_c = \text{corr}(D_{ij}, C_{ij})$$

where $C_{ij}$ is the cophenetic distance (the height at which languages $i$ and $j$ first merge in the dendrogram). Values close to 1.0 indicate the dendrogram is a faithful representation of the distance structure.

#### Adjusted Rand Index (ARI)

Compares the clusters discovered by Ward's method against the ground-truth family labels:

$$\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$$

where RI is the Rand Index (fraction of pairs that are either in the same cluster in both partitions or in different clusters in both partitions). ARI = 1.0 means perfect agreement with true families; ARI = 0 means no better than random.

#### Family Gap

The most intuitive metric -- the difference between intra-family and inter-family CKA:

$$\text{Gap}_l = \overline{\text{CKA}}_\text{intra}^{(l)} - \overline{\text{CKA}}_\text{inter}^{(l)}$$

A positive gap means the model treats same-family languages more similarly than different-family languages. **Convergence to a language-agnostic space implies Gap -> 0**: the model stops distinguishing families.

### 7.3 Mechanistic Interpretability Perspective

The dissolution of language family clusters is direct evidence for the transition from stage 1 (language-specific encoding) to stage 2 (language-agnostic processing). Tracking **which families dissolve first** reveals the model's learning priorities:
- If Indo-European dissolves first (high-resource languages become indistinguishable before low-resource ones), it suggests that training data volume drives representational sharing.
- If Niger-Congo (Swahili, Yoruba) remains clustered in late layers while Indo-European has dissolved, it indicates inequitable representational quality across resource tiers.

The ARI declining across layers is the quantitative signature of the "dissolution of linguistic identity" -- a key mechanistic finding.

---

## 8. Notebook 05: Anisotropy and Whitened CKA (Novel Technique 2)

### 8.1 The Anisotropy Problem

**Representation anisotropy** is a well-documented phenomenon in transformer models (Ethayarajh, 2019): all hidden-state vectors tend to cluster in a narrow cone of the embedding space, leading to high pairwise cosine similarity even between semantically unrelated inputs. This can inflate CKA scores, making cross-lingual alignment appear stronger than it actually is.

### 8.2 Measuring Anisotropy

Anisotropy is quantified as the average cosine similarity between random pairs of sentence embeddings within a single language:

$$\text{Aniso}(\ell, l) = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \frac{e_l^\ell(s_i) \cdot e_l^\ell(s_j)}{\|e_l^\ell(s_i)\| \cdot \|e_l^\ell(s_j)\|}$$

where $\mathcal{P}$ is a set of random pairs from the same language. Values close to 1.0 indicate severe anisotropy (all vectors point in similar directions); values close to 0 indicate isotropic representations.

### 8.3 Eigenvalue Spectrum Analysis

The eigenvalue spectrum of the representation covariance matrix reveals the **intrinsic dimensionality** of the representation space:

$$\Sigma_l = \frac{1}{n} \tilde{X}_l^\top \tilde{X}_l$$

where $\tilde{X}_l = X_l - \bar{X}_l$ is the mean-centered activation matrix. A sharply decaying spectrum (few dominant eigenvalues) indicates that representations live in a low-dimensional subspace -- high anisotropy. A flat spectrum indicates isotropic representations.

### 8.4 ZCA Whitening

**Zero-phase Component Analysis (ZCA) whitening** transforms the data so that its covariance matrix becomes the identity, removing anisotropy while staying as close as possible to the original data:

$$X_w = \tilde{X} \cdot W, \quad W = V \cdot \text{diag}\left(\frac{1}{\sqrt{\lambda_i + \epsilon}}\right) \cdot V^\top$$

where $\lambda_i, V$ are the eigenvalues and eigenvectors of $\Sigma$, and $\epsilon$ is a regularization constant.

**Whitened CKA** applies ZCA whitening to both activation matrices before computing standard linear CKA:

$$\text{CKA}_\text{whitened}(X, Y) = \text{CKA}_\text{linear}(X_w, Y_w)$$

### 8.5 Interpretation

If whitened CKA scores are **substantially lower** than standard CKA scores, the observed alignment was partly an artifact of anisotropy -- the representations were high-CKA simply because all vectors pointed in similar directions, not because the model genuinely mapped different languages to the same semantic regions.

If whitened CKA scores **remain high** (close to standard CKA), the alignment is genuine and robust to geometric confounds.

### 8.6 Mechanistic Interpretability Perspective

Anisotropy analysis reveals whether the "shared multilingual space" reported in prior work (Wendler et al., 2024; Harrasse et al., 2025) is a genuine semantic alignment or a geometric artifact. This is a critical distinction:
- **Genuine alignment**: The model has learned to represent "cat" and "gato" and "billi" near each other in embedding space because they share semantic features.
- **Anisotropy artifact**: All representations are crammed into a narrow cone, so everything is near everything else, regardless of semantic content.

The eigenvalue spectrum further reveals the **effective dimensionality** of the multilingual space at each layer. If the model compresses 3072-dimensional representations into a much lower-dimensional subspace in later layers, this is evidence of abstract, compressed semantic processing -- consistent with the "concept bottleneck" hypothesis.

---

## 9. Notebook 06: Retrieval Alignment (Novel Technique 3)

### 9.1 From Geometry to Function

CKA measures **geometric** similarity between representation spaces. But geometry can be misleading: two spaces can be geometrically similar (high CKA) without being functionally useful for cross-lingual tasks. Conversely, modest geometric differences might not impair function.

This notebook bridges the gap by measuring **functional alignment**: can the embedding space at each layer actually support a practical cross-lingual task?

### 9.2 The Task: Parallel Sentence Retrieval

Given an English sentence embedding, find its translation in another language by nearest-neighbor search in the shared representation space:

1. Compute the cosine similarity matrix between all English sentences and all target-language sentences:

$$S_{ij} = \frac{e_l^\text{en}(s_i) \cdot e_l^\text{tgt}(s_j)}{\|e_l^\text{en}(s_i)\| \cdot \|e_l^\text{tgt}(s_j)\|}$$

2. For each English sentence $s_i$, rank all target sentences by descending similarity.
3. The correct translation is $s_i$ in the target language (same index, since FLORES+ is aligned).

### 9.3 Metrics

**Mean Reciprocal Rank (MRR)**:

$$\text{MRR} = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{\text{rank}_i}$$

where $\text{rank}_i$ is the position of the correct translation in the ranked list. MRR = 1.0 means every translation is the nearest neighbor; MRR close to 0 means translations are effectively random.

**Recall@k**:

$$\text{Recall@}k = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}[\text{rank}_i \leq k]$$

Recall@1 is the strictest test (is the correct translation the single nearest neighbor?); Recall@10 is more lenient.

### 9.4 Mechanistic Interpretability Perspective

Retrieval metrics provide a **task-grounded** measure of the quality of the shared multilingual space. While CKA tells us "these representations are geometrically similar," MRR tells us "this similarity is sufficient for a real cross-lingual task."

The layer with the highest MRR is the layer where the model's representations are most **functionally multilingual** -- it has learned to position translations as nearest neighbors in embedding space. This may or may not coincide with the CKA convergence layer: if MRR peaks at a different layer than CKA, it suggests that geometric similarity does not perfectly predict functional utility.

This analysis is inspired by Artetxe & Schwenk (2019), who demonstrated that cross-lingual sentence embeddings can serve as the basis for zero-shot transfer, and by the XTREME benchmark (Hu et al., 2020), which uses retrieval-based evaluation for multilingual models.

---

## 10. Notebook 07: Script Decomposition (Novel Technique 4)

### 10.1 The Script Confound

A critical threat to validity in cross-lingual alignment studies is the **script confound**: languages that share a writing system (e.g., English, Spanish, French, German, Swahili, Turkish, Yoruba -- all Latin script) will share many BPE subword tokens. This token-level overlap could produce high CKA scores at early layers *even if the model has not learned any semantic alignment*.

### 10.2 Decomposition

We split the CKA scores into two categories at each layer:

**Intra-script CKA**: Average CKA between languages that share the same writing system:

$$\overline{\text{CKA}}_\text{intra-script}^{(l)} = \text{mean}\{\text{CKA}(X_l^{(i)}, X_l^{(j)}) : \text{script}(i) = \text{script}(j), i \neq j\}$$

**Inter-script CKA**: Average CKA between languages with different writing systems:

$$\overline{\text{CKA}}_\text{inter-script}^{(l)} = \text{mean}\{\text{CKA}(X_l^{(i)}, X_l^{(j)}) : \text{script}(i) \neq \text{script}(j)\}$$

**Script Gap**:

$$\text{ScriptGap}_l = \overline{\text{CKA}}_\text{intra-script}^{(l)} - \overline{\text{CKA}}_\text{inter-script}^{(l)}$$

### 10.3 Script Groups

| Script | Languages | Count |
|---|---|---|
| Latin | en, es, fr, de, sw, tr, yo | 7 |
| Arabic | ar, fa | 2 |
| Devanagari | hi | 1 |
| Bengali | bn | 1 |
| Tamil | ta | 1 |
| Ge'ez | am | 1 |

The Latin script group is dominant (7 of 13 languages), providing 21 intra-script pairs. The Arabic script group (ar, fa) provides 1 intra-script pair. Scripts with a single language (Devanagari, Bengali, Tamil, Ge'ez) have no intra-script pairs.

### 10.4 Mechanistic Interpretability Perspective

The **evolution of the script gap across layers** tells a clear mechanistic story:

- **Large script gap in early layers**: The model's representations are dominated by token-surface features. Languages that share a script (and thus share BPE tokens) are represented similarly because they literally share the same input tokens, not because the model understands their semantic equivalence.
- **Script gap approaching zero in later layers**: The model has learned to abstract away from surface-level script features and represent meaning regardless of writing system. An Arabic sentence and its English translation are now positioned similarly because they mean the same thing, not because they share tokens.

This is the most direct test of the "tongue from thought" distinction (Dumas et al., 2025): are we seeing genuine semantic convergence, or just BPE-level similarity?

The **Latin-script deep dive** further tests whether Western European languages (en, es, fr, de) are more similar to each other than to African Latin-script languages (sw, yo), which would indicate a confound from shared vocabulary roots within Indo-European rather than just shared script.

---

## 11. Notebook 08: Regional Comparison (Novel Technique 5)

### 11.1 Research Question

Tiny Aya comes in multiple variants: Global (balanced across 70+ languages), South Asia (optimized for South Asian languages), Africa (optimized for African languages), and others. The question is: **does regional specialization trade off cross-lingual universality?**

### 11.2 Delta-CKA

For each layer, compute the difference in CKA matrices between the global and regional models:

$$\Delta\text{CKA}_{ij}^{(l)} = \text{CKA}_\text{global}^{(l)}(i, j) - \text{CKA}_\text{regional}^{(l)}(i, j)$$

The average off-diagonal delta:

$$\overline{\Delta\text{CKA}}_l = \frac{2}{n(n-1)} \sum_{i < j} \Delta\text{CKA}_{ij}^{(l)}$$

- **Positive delta**: The global model has higher cross-lingual alignment at this layer than the regional model.
- **Negative delta**: The regional model has developed stronger cross-lingual alignment (potentially for its target languages).

### 11.3 Expected Findings

Based on the Tiny Aya paper (arXiv:2603.11510):
- Regional models should show **higher intra-region CKA** (e.g., South Asia model should have higher Hindi-Bengali CKA than the global model).
- Regional models should show **lower inter-region CKA** (e.g., South Asia model should have lower Hindi-Yoruba CKA than the global model).
- The **delta should be most pronounced in late layers**, where regional fine-tuning has the strongest effect, while early layers (which capture more universal linguistic features) may be relatively unaffected.

### 11.4 Mechanistic Interpretability Perspective

Delta-CKA reveals **where in the network regional specialization occurs**:

- If delta is concentrated in **early layers**: Regional fine-tuning has modified the low-level feature extraction, potentially altering tokenization-level representations. This is a deep structural change.
- If delta is concentrated in **late layers**: Regional fine-tuning has primarily modified the language-specific decoding stage, leaving the shared multilingual core intact. This is the safer and more expected outcome.
- If delta is **uniform across layers**: The model has been thoroughly restructured for the target region, suggesting that global and regional models use fundamentally different internal representations.

This analysis directly addresses the project's original research question: "Which parts of Tiny Aya's network learn language-agnostic representations, and which parts become specialized for specific languages or regions?"

---

## 12. Synthesis: What the Full Pipeline Reveals

The eight notebooks form a coherent pipeline that progressively deepens our understanding:

### Layer 0: The Embedding Layer
- **Expected**: High intra-script CKA (Latin-script languages are similar), large script gap, strong family clustering, low inter-script CKA. Representations dominated by tokenization artifacts.
- **Metrics**: High ARI (clusters match families), large script gap, low MRR for cross-script pairs.

### Layers 1-2: The Convergence Zone
- **Expected**: CKA scores rising, script gap narrowing, family clusters beginning to dissolve, MRR improving.
- **Mechanistic interpretation**: The model is building shared semantic representations. The transition from "this is Devanagari text" to "this is about a cat" happens here.

### Layer 3: The Output Layer
- **Expected**: Either continued convergence (full language-agnostic space) or slight divergence (the model beginning to specialize for output language prediction, consistent with stage 3 of the three-stage hypothesis).
- **Key question**: Does the final layer show a slight decrease in cross-lingual CKA as the model begins to prepare language-specific outputs?

### Cross-cutting Analysis

| Metric | Early Layers | Late Layers | Significance |
|---|---|---|---|
| Avg Cross-Lingual CKA | Lower | Higher (ideally > 0.75) | Core convergence signal |
| Family Gap | Large positive | Near zero | Family dissolution |
| Script Gap | Large positive | Near zero | Genuine vs. surface alignment |
| ARI | High (matches families) | Low (random clusters) | Quantitative dissolution |
| Cophenetic Correlation | High (strong structure) | Lower (flatter dendrogram) | Structure dissolution |
| MRR | Low | High at best layer | Functional alignment |
| Whitened CKA | Lower than standard | Similar to standard (if genuine) | Anisotropy control |
| Delta-CKA | Near zero | Positive or negative | Regional specialization |

---

## 13. Alignment with the Linear Issue (TIN-7)

This analysis directly addresses every requirement of the TIN-7 specification:

### Data Preparation (Step 1)
- FLORES+ parallel corpus with 1,012 semantically aligned sentences across 13 languages.
- Language metadata (family, script, resource level) tracked via the `Language` enum.

### Activation Extraction (Step 2)
- Forward hooks on all 4 transformer layers via `ActivationStore` + `register_model_hooks`.
- Mean-pooled sentence embeddings of shape `(1012, 3072)` per language per layer.
- Design directly extends `register_teacher_hooks` from project-aya notebook 07.

### CKA Cross-Language Similarity Matrix (Step 3)
- Full `(13, 13, 4)` similarity tensor computed with `linear_cka` and `rbf_cka`.
- Both linear and RBF kernels implemented, with mini-batch support for memory efficiency.

### Finding the Convergence Layer (Step 4)
- Average cross-lingual CKA plotted vs. layer depth with 95% confidence intervals.
- Convergence layer identified as first layer exceeding the 0.75 threshold.
- Permutation tests confirm statistical significance.

### Novel Techniques (Step 5)

| Technique | Section | Specification Requirement |
|---|---|---|
| Language Family Clustering | Notebook 04 | Technique 1: Hierarchical clustering, family dissolution tracking |
| Anisotropy-Corrected CKA | Notebook 05 | Technique 2: ZCA whitening before CKA, complementary metric |
| Retrieval Scoring (MRR) | Notebook 06 | Technique 3: Task-grounded functional alignment measurement |
| Script-Based Decomposition | Notebook 07 | Technique 4: Intra- vs. inter-script CKA per layer |
| Regional Model Comparison | Notebook 08 | Technique 5: Delta-CKA between Global and regional variants |

### Output Visualizations

| Visualization | Notebook | Status |
|---|---|---|
| Cross-lingual CKA heatmap | 03 | Implemented |
| Convergence curve | 03 | Implemented |
| Language dendrogram | 04 | Implemented |
| Family gap curve | 04 | Implemented |
| Anisotropy heatmap | 05 | Implemented |
| Eigenvalue spectrum | 05 | Implemented |
| Retrieval MRR curve | 06 | Implemented |
| Recall@k bar charts | 06 | Implemented |
| Script decomposition | 07 | Implemented |
| Delta-CKA heatmaps | 08 | Implemented |

### Module Architecture

The `uth/analysis/` package implements the exact module structure specified in TIN-7:

```
uth/analysis/
    cross_lingual_alignment.py   # CrossLingualAlignmentAnalyzer orchestrator
    retrieval_metrics.py         # MRR, Recall@k, MAP computation
    clustering.py                # Hierarchical clustering, family dissolution
    cka.py                       # Linear, RBF, whitened, mini-batch CKA
    hooks.py                     # ActivationStore, model loading
    visualization.py             # Publication-quality plotting functions
```

The core class `CrossLingualAlignmentAnalyzer` provides the exact API from the specification:

```python
class CrossLingualAlignmentAnalyzer:
    def __init__(self, model, tokenizer, parallel_corpus, ...)
    def extract_all_activations(self) -> None
    def compute_cka_matrices(self, kernel="linear") -> dict[int, ndarray]
    def find_convergence_layer(self, threshold=0.75) -> Optional[int]
    def compute_retrieval_scores(self) -> dict
    def save_results(self, output_dir: str) -> None
```

---

## 14. References

1. **Kornblith, S., Norouzi, M., Lee, H., & Hinton, G.** (2019). Similarity of Neural Network Representations Revisited. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, PMLR 97:3519-3529.

2. **Wendler, C., Veselovsky, V., Monea, G., & West, R.** (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers. *Proceedings of the Association for Computational Linguistics*.

3. **Dumas, C., Wendler, C., Veselovsky, V., Monea, G., & West, R.** (2025). Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers. *Proceedings of the 63rd Annual Meeting of the ACL*.

4. **Harrasse, A., Draye, F., Pandey, P.S., & Jin, Z.** (2025). Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders. arXiv:2511.10840.

5. **Nakai, T., Chikkala, R.K., et al.** (2025). TRepLiNa: Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B. arXiv:2510.06249.

6. **Salamanca, A.R., et al. (Cohere Labs).** (2026). Tiny Aya: Bridging Scale and Multilingual Depth. arXiv:2603.11510.

7. **Artetxe, M. & Schwenk, H.** (2019). Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond. *Transactions of the ACL*.

8. **Hu, J., et al.** (2020). XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalisation. *Proceedings of ICML*.

9. **Ethayarajh, K.** (2019). How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. *Proceedings of EMNLP*.

10. **Nguyen, T., Raghu, M., & Kornblith, S.** (2021). Do Wide Neural Networks Really Need to be Wide? A Minibatch CKA Perspective. *Proceedings of AAAI*.

11. **Wu, Z., Yu, X.V., Yogatama, D., Lu, J., & Kim, Y.** (2024). The Semantic Hub Hypothesis: Language Models Share Semantic Representations Across Languages and Modalities. arXiv:2411.04986.

12. **Koerner, F., et al.** (2026). Where Meanings Meet: Investigating the Emergence and Quality of Shared Concept Spaces during Multilingual Language Model Training. *Proceedings of EACL*.
