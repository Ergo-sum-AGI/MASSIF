# Cross-Architectural Study of Persistence Phase Transitions in Latent Transformer Dynamics

**A Multi-Model Statistical Validation (8 Architectures)**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20232960.svg)](https://doi.org/10.5281/zenodo.20232960)

## Related Publications

This paper is the **empirical validation study** accompanying the theoretical framework introduced in:

> Solis, D. (2026). *Latent Dynamics in Autoregressive Transformers: A Mathematical Framework for Measuring Recursive Self-Reference*. Zenodo. https://doi.org/10.5281/zenodo.20208319

## Key Findings

| Finding | Result |
|---------|--------|
| **Cleanest sustained flip** | Gemma-2-2B at n=4 (all temperatures) |
| **Temperature-sensitive flips** | Bloom-560M, GPT-2 XL flip at T=0.5 (n=4) |
| **Never flip** | GPT-2 Small, Medium, Large, TinyLlama, Qwen2 |
| **Anti-persistence** | Universal for non-repetitive inputs ($I_t < 0$) |
| **Tokenization matters** | "I" flips; "I " does not |
| **Peak layer** | Layer 2 (early) for GPT-2 Small |

## Models Tested (8 architectures)

| Model | Parameters | Flip Behavior |
|-------|------------|---------------|
| GPT-2 Small | 124M | No sustained flip |
| GPT-2 Medium | 355M | No sustained flip |
| GPT-2 Large | 774M | No sustained flip |
| GPT-2 XL | 1.5B | Flips at n=4 (T=0.5 only) |
| TinyLlama 1.1B | 1.1B | No sustained flip |
| Qwen2 0.5B | 0.5B | No sustained flip |
| Bloom-560M | 560M | Flips at n=4 (T=0.5,0.7) |
| **Gemma-2-2B** | 2.6B | **Sustained flip at n=4 (all T)** |

## Key Insight

> *"Anti-persistence ($I_t < 0$) is universal. Sustained persistence phase transitions are rare and architecture-specific. Gemma-2-2B is the only model in our study exhibiting a clean, temperature-invariant monotonic flip."*

## Repository Contents

- `paper.pdf` - Main empirical paper
- `code/` - All experimental code
- `data/` - Raw results
- `figures/` - Publication-ready figures

## Citation

```bibtex
@techreport{Solis2026Persistence,
  author = {Solis, Daniel},
  title = {Cross-Architectural Study of Persistence Phase Transitions in Latent Transformer Dynamics},
  institution = {DUBITO Inc. / Ergo Sum AGI Safety Systems},
  year = {2026},
  doi = {10.5281/zenodo.20232960},
  note = {Statistical validation across 8 architectures. Anti-persistence is universal; sustained flips are rare and architecture-specific.}
}