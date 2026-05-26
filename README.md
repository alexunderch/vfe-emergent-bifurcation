# VFE Emergence Bifurcation

Code for "In Pursuit of the Emergence Point: Extracting Phase Transitions in Multi-Agent Communication".
Reqiures `python >= 3.12`.

## Quick start

1. Install
```bash
pip install -r requirements.txt
```

2. Run experiments

All ODE simulations use the following fixed settings:

- **Solver:** Dormand–Prince adaptive RK45 (`diffrax.Dopri5()`)
- **Tolerances:** `rtol=1e-6`, `atol=1e-9`
- **Horizon:** `T=100`
- **Initialisation:** `Z ~ N(0, 0.01)`
- **Seeds:** 5 independent seeds (0, 1, 2, 3, 4)
- **Laplacian:** Row-clique + column-clique graph

```bash
python src/experiment.py
```