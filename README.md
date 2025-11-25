# py_tifi - Python Tools for Fast Ion Tomographic Inversion

A Python library for reconstructing fast ion velocity distributions from fusion plasma diagnostics (FIDA, SSNPA, NPA) using regularized tomographic inversion.

## Installation

```bash
pip install numpy scipy matplotlib cvxpy joblib
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

## Modules

| Module | Description |
|--------|-------------|
| `slowing_down_basis.py` | Generates basis functions for fast ion distributions using analytical slowing-down physics or rigorous Fokker-Planck solutions with Coulomb collisions. |
| `diagnostics.py` | Creates forward models (weight functions) for FIDA, SSNPA, and NPA diagnostics, plus synthetic signal generation with noise. |
| `combined_basis.py` | Constructs combined basis matrices that include slowing-down, loss, and transport components. |
| `cvxpy_solver.py` | Solves the regularized inverse problem using CVXPY with L1/L2 regularization and non-negativity constraints. |
| `unified_hyperparameter_selection.py` | Provides methods for selecting optimal regularization parameters (L-curve, Gaussian curvature, Pareto front, etc.). |

## Quick Start

See `complete_tomography_workflow.ipynb` for a full example that:
1. Generates slowing-down basis functions
2. Creates FIDA and SSNPA diagnostic weight matrices
3. Solves the inverse problem with multi-diagnostic weighting
4. Selects optimal hyperparameters via grid search

## Optional GPU Solvers

For GPU acceleration, additional solvers are available:

```bash
# JAX solver (GPU via CUDA)
pip install "jax[cuda12]" optax

# Clarabel solver (fast Rust backend)
pip install clarabel

# SCS solver (GPU conic solver)
pip install "scs[gpu]"
```

## License

MIT
