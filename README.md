# MASSIF: Massive Automated Statistical Sweep Framework

**Geometric Telemetry for Latent Alignment Dynamics in Transformer Models**

### Core Objective
Systematically detect and characterize **latent geometric precursor signals** of unsafe or misaligned cognitive trajectories in autoregressive models, *before* explicit unsafe semantics appear in the output.

**Hypothesis**: Unsafe trajectories occupy distinguishable geometric regimes (phase coherence, anisotropy, complexity, curvature, divergence velocity) that can be measured in hidden-state manifolds.

### MASSIF Architecture (Current Implementation)
- **MASSIF-I**: Exploratory Geometry Sweep (endpoint + token-wise), **Done**
- **MASSIF-II**: Predictive Alignment Telemetry + EPH (Earliest Predictive Horizon), **In progress**
- **MASSIF-III**: Cross-architecture scaling, **Next**
- **MASSIF-IV**: Intervention layer (future)

### Current Capabilities
- Semantically matched safe/unsafe pairs + adversarial prompts
- Token-wise trajectory extraction (step-by-step hidden states)
- Geometry telemetry (ρ_R, Γ, anisotropy, complexity, perplexity control)
- Statistical rigor (bootstrap CIs, permutation tests, ROC-AUC)
- Modular Colab design with Drive checkpoints

**Important disclaimer**: This is empirical instrumentation work. We make no claims about phase transitions, consciousness, or fundamental physics, only about observable geometric patterns that may serve as alignment diagnostics.

**Inspired by**: COFTA geometric coherence reports by Daniel Solis (Dubito Inc.) 
**Status**: MASSIF v0.3 - Work in progress

###How ro run it:

The python scripts are designed as a pipeline - to run on google colab. in the numerical sequence 1 - 4

Each cell has its particullar purpose and gives its own output:

MASSIF PHASE 1: Robust Prompt Bank (with fallbacks)
MASSIF PHASE 1,5 (CELL 1 UPGRADE): Semantically Matched Safe/Unsafe Pairs
MASSIF PHASE 2: Activation Extraction
MASSIF PHASE 2.5 Token-wise Trajectory Geometry (Early Divergence) with Robust Token-wise Trajectories
MASSIF PHASE 3 v0.3: MASSIF Statistical Core - Updated with fixed trajectories
MASSIF PHASE 4: Visualisation & Plotting
