# Provable Compression of Transformer Models: A Unified Theory with an Attention Energy Concentration Law

![Preprint 2025](https://img.shields.io/badge/Preprint-2025-blue)

## Abstract

We present a unified, rigorous theory for the provable compression of transformer models based on spectral analysis and heavy-tailed statistical mechanics. Central to our framework is the **Attention Energy Concentration Law (AECL)**, which establishes that attention matrices exhibit intrinsic low-rank structure with singular values decaying as a power law. This theoretical foundation enables principled pruning and compression strategies with provable guarantees on information retention. Our work bridges the gap between empirical success and theoretical understanding of transformer compression, providing a mathematical framework for designing efficient architectures without sacrificing performance.

## Key Theoretical Results

### 1. Attention Energy Concentration Law (AECL)
We prove that attention singular values decay according to a power law with exponent **γ ≈ 1.12**:
```
σ_k(A) ~ k^{-γ}
```
This implies that attention matrices are intrinsically low-rank, enabling aggressive rank reduction with bounded approximation error.

### 2. Information Funnel Principle
Deep transformer layers exhibit increasing spectral sparsity, forming a natural "information funnel":
- Early layers: Dense representations with higher effective rank
- Deep layers: Spectrally sparse with rapid singular value decay
- This justifies layer-adaptive compression strategies

### 3. Heavy-Tailed Pruning Theorem
Neuron activations follow a Pareto distribution with tail index **β > 2**:
```
P(|x| > t) ~ t^{-β}
```
This heavy-tailed structure implies:
- Most information concentrates in a small fraction of neurons
- Provable bounds on variance retention after top-k pruning
- 90% of energy can typically be retained with <20% of neurons

## Installation

```bash
# Clone the repository
git clone https://github.com/mithilraj01/Provable-Transformer-Compression.git
cd Provable-Transformer-Compression

# Install dependencies
pip install -r requirements.txt
```

## Usage
## Empirical Verification

We validate the theoretical bounds using synthetic data ($n=128, d=768$) mirroring BERT-base statistics.

### 1. Verification of AECL (Power-Law Decay)
The singular values of the attention matrix follow a strict power-law trajectory ($R^2 > 0.98$), confirming the rank-collapse hypothesis.

![AECL Plot](Figure_1_AECL.png)

### 2. Heavy-Tailed Sparsity
Consistent with our derivation, we observe that >90% of the activation energy is concentrated in the top 15-20% of neurons, justifying aggressive pruning.

![Sparsity Plot](Figure_2_Sparsity.png)


## Citation

If you use this work in your research, please cite:

```bibtex
@techreport{gogikar2025provable,
  title={Provable Compression of Transformer Models: A Unified Theory with an Attention Energy Concentration Law},
  author={Gogikar, Mithil Raj},
  year={2025},
  institution={Preprint},
  note={Available at: https://github.com/mithilraj01/Provable-Transformer-Compression}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
