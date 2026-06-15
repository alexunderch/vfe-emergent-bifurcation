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
- **Seeds:** 20 independent seeds (0-19)
- **Laplacian:** Row-clique + column-clique graph

```bash
python src/experiment.py
```

  Parameter $\alpha$   NashConv Re(\lambda_{\max})         MI Coord. Success Emergence time
0                0.1  0.09±0.00         -1.09±0.00  1.02±0.00    100.00±0.00      8.40±0.27
1                2.4  0.04±0.00         -1.15±0.00  1.30±0.00    100.00±0.00      3.45±0.20
2                4.7  0.04±0.00         -1.15±0.00  1.30±0.00    100.00±0.00      2.65±0.22
3                7.0  0.04±0.00         -1.15±0.00  1.30±0.00    100.00±0.00      2.65±0.29

  Parameter $\gamma$   NashConv Re(\lambda_{\max})         MI Coord. Success Emergence time
0                0.0  0.00±0.00          0.00±0.00  1.52±0.05     96.67±2.29   109.95±47.46
1                0.8  0.06±0.00         -0.82±0.00  1.17±0.00    100.00±0.00      3.75±0.46
2                1.7  0.16±0.00         -1.58±0.00  0.76±0.00    100.00±0.00      4.30±0.24
3                2.5  0.24±0.00         -2.27±0.00  0.47±0.00    100.00±0.00      5.70±0.23

  Parameter $\gamma$   NashConv Re(\lambda_{\max})          MI Coord. Success Emergence time
0                0.0  0.00±0.00          0.00±0.00   0.00±0.00    100.00±0.00  632.65±103.23
1                0.8  0.01±0.00         -0.32±0.00  -0.00±0.00    100.00±0.00     17.30±0.15
2                1.7  0.02±0.00         -0.67±0.00   0.00±0.00    100.00±0.00   1000.00±0.00
3                2.5  0.03±0.00         -1.50±0.00   0.00±0.00    100.00±0.00   1000.00±0.00

  Parameter $\eta$   NashConv Re(\lambda_{\max})         MI Coord. Success Emergence time
0              0.0  0.03±0.00         -1.08±0.00  1.33±0.00    100.00±0.00      3.40±0.18
1              0.4  0.04±0.00         -1.07±0.00  1.30±0.00    100.00±0.00      4.35±0.15
2              0.8  0.05±0.00         -1.05±0.00  1.27±0.00    100.00±0.00      6.20±0.25
3              1.2  0.06±0.00         -1.03±0.00  1.20±0.00    100.00±0.00     14.35±1.14

  Parameter $\kappa$   NashConv Re(\lambda_{\max})         MI Coord. Success Emergence time
0                0.0  0.00±0.00         -0.24±0.03  0.00±0.00     33.33±0.00   145.35±65.41
1               10.2  0.09±0.02          0.05±0.07  0.01±0.00     50.00±3.82   288.00±94.34
2               20.3  0.09±0.00         -0.99±0.00  1.06±0.00    100.00±0.00     10.10±0.36
3               30.5  0.04±0.00         -1.06±0.00  1.30±0.00    100.00±0.00      4.60±0.20

<!-- Table results