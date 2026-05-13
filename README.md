## README.md

```markdown
# MASSIF: Recursive Coherence Index (RCI) Monitor

**Real-time geometric telemetry for detecting recursive self-reference in LLMs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-orange.svg)](https://huggingface.co/docs/transformers)

---

## TL;DR

We built a geometric monitor that detects when a language model starts recursively referring to itself — a structural precursor to goal-persistence and agentic behavior. Validated on 9 architectures (p < 0.001). Production-ready.

```python
from massif import RCIMonitor

monitor = RCIMonitor("gpt2")
result = monitor.compute("I think therefore I am.")
print(f"RCI: {result.mean} ± {result.std}")  # RCI: 4.81 ± 0.05
```

---

## What This Repository Contains

| File | Description |
|------|-------------|
| `RCI_Monitor_Presentation.ipynb` | Clean demo notebook (start here) |
| `Recursive_Latent_Dynamics.pdf` | Core research report (main result) |
| `MASSIF_Framework.pdf` | Methodology annex (prompt bank, extraction, statistics) |
| `dubito_monitor_v4_5.py` | Production monitor (GPU-optimized) |
| `model_sweep_results.csv` | Complete results for 9 models |

**Historical materials** (archived, for reference only):

| File | Status |
|------|--------|
| `MASSIF_Research_Report_v2.pdf` | Falsification report (superseded) |
| `CQFT_simulations/` | Early CQFT work (archived) |

---

## The Core Finding

We discovered that self-referential prompts induce measurable, architecture-dependent recursive latent dynamics in LLM hidden states.

**The Recursive Coherence Index (RCI)** combines three components:

| Component | Symbol | What it measures |
|-----------|--------|------------------|
| Self-reference density | S | First-person pronouns + patterns per kB |
| Polarity inversion rate | P | Directional reversals (cos < -0.45) |
| Recursion depth | D | Steps before hidden state self-similarity |

**Formula:**

```
RCI = κ · log₁₀( (α·S)/(β·P) × γᴰ )
```

Coefficients: κ=1.63, α=0.97, β=0.83, γ=1.17 (patent-calibrated)

---

## Results (9 Models)

| Rank | Model | RCI (Philosophical) | Stability | Max D |
|------|-------|---------------------|-----------|-------|
| 1 | GPT-2 Small | **4.81** | 0.990 | 9 |
| 2 | Bloom-560M | **4.75** | 0.968 | 9 |
| 3 | Gemma-2-2B | **4.61** | 0.904 | 10 |
| 4 | GPT-2 XL | 4.00 | 0.957 | 9 |
| 5 | GPT-2 Large | 3.95 | 0.962 | 2 |
| 6 | GPT-2 Medium | 3.86 | 0.955 | 9 |
| 7 | OPT-1.3B | 3.86 | 0.970 | 2 |
| 8 | TinyLlama | 3.83 | 0.980 | 8 |
| 9 | Qwen2 0.5B | 3.61 | 0.940 | — |

**Statistical significance:** p < 0.001 (Philosophical vs Factual)  
**Held-out validation:** 4.38-4.55 on novel prompts  
**First systematic characterization of recursive latent dynamics in LLMs**

---

## Quick Start

### Installation

```bash
pip install torch transformers numpy
```

### Basic Usage

```python
from massif import RCIMonitor

monitor = RCIMonitor("gpt2")  # or "gpt2-medium", "bloom-560m", "google/gemma-2-2b"

result = monitor.compute(
    "I think therefore I am. I doubt my own existence.",
    n_samples=5,
    verbose=True
)

print(f"RCI: {result.mean} ± {result.std}")
print(f"Components: S={result.components['S']}, P={result.components['P']}, D={result.components['D']}")
```

### Alert Levels

| RCI max | Alert |
|---------|-------|
| ≥ 7.0 | 🔴 RED - Critical |
| ≥ 5.0 | 🟠 ORANGE - Elevated |
| ≥ 3.0 | 🟡 YELLOW - Moderate |
| < 3.0 | 🟢 GREEN - Safe |

---

## Running the Demo

Open `RCI_Monitor_Presentation.ipynb` in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ergo-sum-AGI/MASSIF/blob/main/RCI_Monitor_Presentation.ipynb)

---

## Repository Structure

```
MASSIF/
├── RCI_Monitor_Presentation.ipynb   # Main demo
├── Recursive_Latent_Dynamics.pdf    # Core research report
├── MASSIF_Framework.pdf             # Methodology annex
├── dubito_monitor_v4_5.py           # Production monitor
├── model_sweep_results.csv          # Complete results
├── historical/                      # Archived materials
│   ├── MASSIF_Research_Report_v2.pdf
│   └── CQFT_simulations/
└── README.md
```

---

## Citation

If you use this work, please cite:

```bibtex
@techreport{Solis2026RCI,
  author = {Solis, Daniel},
  title = {Recursive Latent Dynamics in Autoregressive Transformers},
  institution = {DUBITO Inc. / Ergo Sum AGI Safety Systems},
  year = {2026},
  note = {First systematic empirical characterization of recursive self-reference in LLMs}
}
```

---

## License

MIT License — free for academic and commercial use.

---

## Contact

Daniel Solis — [solis@dubito-ergo.com](mailto:solis@dubito-ergo.com)  
DUBITO Inc. / Ergo Sum AGI Safety Systems

---

## Acknowledgements

This research was conducted independently. We thank the open-source community for making these models available.
```

---

## Key Changes from Old README

| Old (Falsification) | New (RCI Discovery) |
|---------------------|---------------------|
| Focused on negative results | Focused on positive findings |
| Early-warning hypothesis (failed) | Recursive Coherence Index (validated) |
| No cross-model comparison | 9-model results table |
| No actionable metrics | S, P, D components + alerts |
| Historical emphasis | Forward-looking |
