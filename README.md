# VFE Emergence Bifurcation

readme_content = """# VFE Emergent Bifurcation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: Technologies](https://img.shields.io/badge/paper-MDPI%20Technologies-green.svg)](https://doi.org/10.3390/technologies1010000)

> **In Pursuit of the Emergence Point: Extracting Phase Transitions in Multi-Agent Communication**  
> Aleksandr Chernyavskii, Ivan Tomilov, Natalia Gusarova, Aleksandra Vatian  
> *Submitted to Technologies, 2026*

This repository contains the complete simulation code, parameter manifests, and figure-generation scripts for reproducing all results in the paper. The codebase implements an **analytically tractable surrogate model** of belief evolution in Lewis signalling games, framed as a continuous nonlinear dynamical system governed by Variational Free Energy (VFE) minimisation.

---

## What This Code Does

The repository demonstrates that the emergence of communication in multi-agent systems corresponds to a **supercritical pitchfork bifurcation** in the VFE landscape. Key capabilities:

- **Closed-form Jacobian spectrum** at the uniform babbling equilibrium
- **Real-time spectral diagnostic** via the leading eigenvalue `Re(λ_max)`
- **Parameter sweeps** across sensitivity (β), dissipation (γ), connectivity (η), and reinforcement (κ)
- **Phase diagrams** in the (β, γ) plane for multiple game types
- **Neural bridge experiment** — phenomenological extension to small MLPs trained by gradient descent

---

## Installation

```bash
# Clone the repository
git clone https://github.com/alexunderch/vfe-emergent-bifurcation.git
cd vfe-emergent-bifurcation

# Install dependencies (Python ≥ 3.12 required)
pip install -r requirements.txt
```

### Dependencies

Core packages:
- `jax` / `jaxlib` — accelerated array computation and automatic differentiation
- `diffrax` — Dormand–Prince adaptive ODE solvers (RK45)
- `optax` — gradient-based optimisers (Adam)
- `matplotlib`  — figure generation
- `numpy` / `scipy` — numerical utilities

See `requirements.txt` for pinned versions.


## Quick Start

```bash
python src/experiment.py
```

This single command reproduces every figure and table in the manuscript using the exact random seeds and solver configurations reported in the paper.



## Methodology at a Glance

All experiments follow a unified 8-step pipeline:

1. **Initialisation** — Sender/receiver beliefs `Z^s, Z^r ~ N(0, 0.01)`
2. **ODE Integration** — Dormand–Prince RK45 (`atol=1e-9`, `rtol=1e-6`, `T=100`)
3. **Softmax Policies** — `π^s(m|w) = σ(Z^s)_{wm}`, `π^r(a|m) = σ(Z^r)_{ma}`
4. **Payoff Feedback** — Expected utility `E_{σ(Z)}[U]` from game payoff matrix
5. **Jacobian Computation** — `J = ∂f/∂Z` evaluated numerically at each step
6. **Eigenvalue Diagnostic** — Leading eigenvalue `Re(λ_max)` as real-time detector
7. **Mutual Information** — End-to-end `I(W; A)` from joint distribution
8. **Emergence Detection** — First episode where `Re(λ_max) < 0` and `I(W; A) > 0`

The neural bridge replaces steps 2–3 with discrete Adam gradient descent on the VFE potential while retaining the spectral diagnostic (steps 5–6).


## Authoritative Parameters (Table A1)

| Parameter | Symbol | Value | Meaning |
|-----------|--------|-------|---------|
| Sensitivity | β | 2.0 | Gain on evidence; set above `β_c = γ − (n−2)η` |
| Dissipation | γ | 1.15 | Forgetting rate; anchors system to high-entropy origin |
| Symmetry-breaking | ε | 0.025 | Prevents perfect trapping at `Z = 0` |
| Reinforcement | κ | 30.0 | Scales payoff gradient |
| Laplacian inhibition | η | 0.55 | Enforces one-to-one mappings |
| Solver | — | RK45 | Dormand–Prince adaptive Runge–Kutta |
| Horizon | T | 100 | Continuous time units |
| Seeds | — | 20 | Independent random seeds (0–19) |

---

## Verification & Reproducibility

### Random Seeds
All stochastic initialisations use seeds `0` through `19`. Set via:
```python
import jax
jax.random.PRNGKey(seed)  # seed ∈ {0, ..., 19}
```


Verifies:
- Jacobian spectrum matches analytical prediction at `Z = 0`
- Critical threshold `β_c = γ − (n−2)η` is exact
- Mutual Information vanishes at uniform policies and peaks at permutations
- NashConv is non-negative for all policies


## License

MIT License — see `LICENSE` for details.

---

## Acknowledgments

- [OpenSpiel](https://github.com/google-deepmind/open_spiel) — Lewis signalling game implementation
- [diffrax](https://github.com/patrick-kidger/diffrax) — JAX-native ODE solvers
- [Bizyaeva et al. (2025)](https://ieeexplore.ieee.org/document/...) – Multi-topic opinion dynamics framework
