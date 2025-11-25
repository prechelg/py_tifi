"""
CLARABEL Solver Module for Fast Ion Distribution Tomography

This module provides convex optimization solvers using the CLARABEL backend,
a modern Rust-based interior point solver. It maintains the same API as
cvxpy_solver.py and jax_solver.py to allow interchangeable use.

CLARABEL features:
- Modern Rust implementation (fast and memory-efficient)
- Interior point method optimized for conic programs
- Excellent numerical stability
- Active development and support
- Works on CPU with excellent performance

The typical inverse problem structure:
    minimize    ||A @ f - y||^2 + lambda * regularization(c)
    subject to  Phi @ c >= 0  (ensures physical non-negativity of distribution)

where:
    - A: Diagnostic weight function matrix (from diagnostics module)
    - Phi: Basis function matrix (from slowing_down_basis module)
    - y: Measured signal vector
    - c: Coefficient vector (optimization variable)
    - f = Phi @ c: Reconstructed distribution

Usage:
    from clarabel_solver import solve_distribution, solve_distribution_parallel

    # Single solve
    result = solve_distribution(A, Phi, signal, lambda_reg=1e-3, regularization='l1')
    coefficients = result['coefficients']
    distribution = result['distribution']

    # Parallel solve for multiple lambda values
    results = solve_distribution_parallel(A, Phi, signal, lambda_values=[1e-4, 1e-3, 1e-2])

Requirements:
    pip install cvxpy
    pip install clarabel
"""

import numpy as np
import cvxpy as cp
from typing import Dict, List, Optional, Union, Tuple
import warnings

# Optional parallelization support
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    def delayed(func):
        return func


# ============================================================================
# CLARABEL Configuration
# ============================================================================

def get_clarabel_kwargs(**user_kwargs) -> Dict:
    """
    Get default CLARABEL solver parameters optimized for tomography problems.

    Parameters:
    -----------
    **user_kwargs : dict
        User-provided overrides for CLARABEL parameters

    Returns:
    --------
    clarabel_kwargs : dict
        Dictionary of CLARABEL parameters
    """
    # Default CLARABEL parameters optimized for tomography
    defaults = {
        'tol_gap_abs': 1e-8,        # Absolute duality gap tolerance
        'tol_gap_rel': 1e-8,        # Relative duality gap tolerance
        'tol_feas': 1e-8,           # Feasibility tolerance
        'tol_infeas_abs': 1e-8,     # Infeasibility tolerance
        'max_iter': 200,            # Maximum iterations (CLARABEL is fast)
        'verbose': False,           # Suppress output unless requested
    }

    # Override with user-provided kwargs
    defaults.update(user_kwargs)

    return defaults


# ============================================================================
# Core Solver Functions
# ============================================================================

def solve_distribution(
    A: np.ndarray,
    Phi: np.ndarray,
    signal: np.ndarray,
    lambda_reg: float = 1e-3,
    regularization: str = 'l1',
    non_negative_constraint: bool = True,
    verbose: bool = False,
    **solver_kwargs
) -> Dict[str, Union[np.ndarray, float, str]]:
    """
    Solve the inverse problem using CLARABEL solver.

    Solves:
        minimize    ||A @ Phi @ c - y||^2 + lambda * regularization(c)
        subject to  Phi @ c >= 0  (optional non-negativity constraint)

    Parameters:
    -----------
    A : np.ndarray
        Diagnostic weight function matrix, shape (n_measurements, n_grid_points)
    Phi : np.ndarray
        Basis function matrix, shape (n_grid_points, n_basis_functions)
    signal : np.ndarray
        Measured signal vector, shape (n_measurements,)
    lambda_reg : float
        Regularization parameter (default: 1e-3)
    regularization : str
        Type of regularization:
        - 'l1': L1 norm (promotes sparsity, default)
        - 'l2': L2 norm (ridge regression)
        - 'none': No regularization
    non_negative_constraint : bool
        If True, enforce Phi @ c >= 0 (default: True)
    verbose : bool
        If True, print solver output (default: False)
    **solver_kwargs : dict
        Additional arguments passed to CLARABEL solver (overrides defaults)

    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'coefficients': Optimized coefficient vector c, shape (n_basis_functions,)
        - 'distribution': Reconstructed distribution Phi @ c, shape (n_grid_points,)
        - 'residual': Residual norm ||A @ Phi @ c - y||
        - 'objective': Final objective function value
        - 'status': Solver status string
        - 'solve_time': Solver time in seconds
        - 'solver': Solver used ('CLARABEL')

    Example:
    --------
    >>> from diagnostics import create_fida_diagnostic, generate_signal
    >>> from slowing_down_basis import generate_simple_basis
    >>>
    >>> # Create basis and diagnostics
    >>> Phi = generate_simple_basis(E_values, p_values, S0=1e20, tau_s=1e-3)
    >>> A = create_fida_diagnostic(E_values, p_values, [5, 25, 45, 65, 85])
    >>>
    >>> # Generate synthetic signal
    >>> true_dist = ...  # Some distribution
    >>> signal = generate_signal(A, true_dist, noise_level=0.05)
    >>>
    >>> # Solve
    >>> result = solve_distribution(A, Phi, signal, lambda_reg=1e-3)
    >>> print(f"Residual: {result['residual']:.2e}")
    >>> print(f"Solver: {result['solver']}")
    """
    # Validate inputs
    if A.shape[0] != len(signal):
        raise ValueError(f"Signal length {len(signal)} does not match A rows {A.shape[0]}")
    if A.shape[1] != Phi.shape[0]:
        raise ValueError(f"A columns {A.shape[1]} does not match Phi rows {Phi.shape[0]}")

    # Compute combined matrix A_Phi
    A_Phi = A @ Phi

    # Normalize for numerical stability (optional but recommended)
    A_Phi_norm = np.linalg.norm(A_Phi)
    A_Phi_normalized = A_Phi / A_Phi_norm if A_Phi_norm > 0 else A_Phi

    # Define optimization variable
    n_basis = Phi.shape[1]
    c = cp.Variable(n_basis)

    # Define objective function
    data_fidelity = cp.sum_squares(A_Phi_normalized @ c - signal)

    if regularization.lower() == 'l1':
        reg_term = lambda_reg * cp.norm1(c)
    elif regularization.lower() == 'l2':
        reg_term = lambda_reg * cp.sum_squares(c)
    elif regularization.lower() == 'none':
        reg_term = 0
    else:
        raise ValueError(f"Unknown regularization type: {regularization}. Use 'l1', 'l2', or 'none'")

    objective = cp.Minimize(data_fidelity + reg_term)

    # Define constraints
    constraints = []
    if non_negative_constraint:
        constraints.append(Phi @ c >= 0)

    # Create and solve problem
    problem = cp.Problem(objective, constraints)

    # Get CLARABEL solver kwargs
    clarabel_kwargs = get_clarabel_kwargs(**solver_kwargs)
    if verbose:
        clarabel_kwargs['verbose'] = True

    try:
        problem.solve(solver=cp.CLARABEL, **clarabel_kwargs)
    except cp.error.SolverError as e:
        warnings.warn(f"CLARABEL solver failed: {e}. Trying default solver.")
        problem.solve(verbose=verbose)

    # Extract results
    if c.value is None:
        raise ValueError(f"Solver failed with status: {problem.status}")

    coefficients = c.value
    distribution = Phi @ coefficients
    residual = np.linalg.norm(A_Phi @ coefficients - signal)

    return {
        'coefficients': coefficients,
        'distribution': distribution,
        'residual': residual,
        'objective': problem.value,
        'status': problem.status,
        'solve_time': problem.solver_stats.solve_time if problem.solver_stats else None,
        'solver': 'CLARABEL'
    }


def solve_distribution_grid(
    A: np.ndarray,
    Phi: np.ndarray,
    signal: np.ndarray,
    E_values: np.ndarray,
    p_values: np.ndarray,
    **kwargs
) -> Dict[str, Union[np.ndarray, float, str]]:
    """
    Solve distribution and return result with 2D grid reshaping.

    This is a convenience wrapper around solve_distribution that automatically
    reshapes the distribution to (n_pitch, n_energy) grid format.

    Parameters:
    -----------
    A, Phi, signal : np.ndarray
        Same as solve_distribution
    E_values : np.ndarray
        Energy grid values (eV)
    p_values : np.ndarray
        Pitch grid values (dimensionless)
    **kwargs : dict
        Additional arguments passed to solve_distribution

    Returns:
    --------
    result : dict
        Same as solve_distribution, plus:
        - 'distribution_2d': Distribution reshaped to (len(p_values), len(E_values))
        - 'coefficients_2d': Coefficients reshaped to (len(p_values), len(E_values))
                            if n_basis == n_grid_points
    """
    result = solve_distribution(A, Phi, signal, **kwargs)

    # Add 2D reshaped versions
    result['distribution_2d'] = result['distribution'].reshape(len(p_values), len(E_values))

    # Reshape coefficients if they match grid size
    if len(result['coefficients']) == len(E_values) * len(p_values):
        result['coefficients_2d'] = result['coefficients'].reshape(len(p_values), len(E_values))

    return result


# ============================================================================
# Parallel Solver Functions
# ============================================================================

def _solve_single_lambda(
    A: np.ndarray,
    Phi: np.ndarray,
    signal: np.ndarray,
    lambda_reg: float,
    **kwargs
) -> Tuple[float, Dict]:
    """Helper function for parallel execution."""
    result = solve_distribution(A, Phi, signal, lambda_reg=lambda_reg, **kwargs)
    return lambda_reg, result


def solve_distribution_parallel(
    A: np.ndarray,
    Phi: np.ndarray,
    signal: np.ndarray,
    lambda_values: Union[List[float], np.ndarray],
    n_jobs: int = -1,
    **kwargs
) -> Dict[float, Dict]:
    """
    Solve distribution for multiple regularization parameters in parallel.

    This function is useful for hyperparameter tuning or L-curve analysis.

    Parameters:
    -----------
    A : np.ndarray
        Diagnostic weight function matrix
    Phi : np.ndarray
        Basis function matrix
    signal : np.ndarray
        Measured signal vector
    lambda_values : list or array
        List of regularization parameters to try
    n_jobs : int
        Number of parallel jobs (default: -1 uses all cores)
        Set to 1 to disable parallelization
    **kwargs : dict
        Additional arguments passed to solve_distribution

    Returns:
    --------
    results : dict
        Dictionary mapping lambda_reg -> result dictionary
        Keys are lambda values, values are result dicts from solve_distribution

    Example:
    --------
    >>> lambda_values = np.logspace(-5, -2, 10)
    >>> results = solve_distribution_parallel(A, Phi, signal, lambda_values, n_jobs=-1)
    >>>
    >>> # Find best lambda by residual
    >>> best_lambda = min(results.keys(), key=lambda l: results[l]['residual'])
    >>> print(f"Best lambda: {best_lambda:.2e}")
    >>> best_result = results[best_lambda]
    """
    if not HAS_JOBLIB and n_jobs != 1:
        warnings.warn("joblib not installed. Running serially. Install with: pip install joblib")
        n_jobs = 1

    if HAS_JOBLIB and n_jobs != 1:
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_solve_single_lambda)(A, Phi, signal, lambda_reg, **kwargs)
            for lambda_reg in lambda_values
        )
    else:
        results_list = [_solve_single_lambda(A, Phi, signal, lambda_reg, **kwargs)
                       for lambda_reg in lambda_values]

    # Convert to dictionary
    return {lambda_reg: result for lambda_reg, result in results_list}


# ============================================================================
# Utility Functions
# ============================================================================

def compute_l_curve(
    results: Dict[float, Dict],
    log_scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute L-curve data from parallel solver results.

    The L-curve plots regularization term vs. residual norm to help
    choose the optimal regularization parameter.

    Parameters:
    -----------
    results : dict
        Results from solve_distribution_parallel
    log_scale : bool
        If True, return log10 values (default: True)

    Returns:
    --------
    lambda_values : np.ndarray
        Array of lambda values
    residuals : np.ndarray
        Array of residual norms
    reg_norms : np.ndarray
        Array of solution norms (approximate regularization term)

    Example:
    --------
    >>> results = solve_distribution_parallel(A, Phi, signal, lambda_values)
    >>> lambdas, residuals, norms = compute_l_curve(results)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.plot(residuals, norms, 'o-')
    >>> plt.xlabel('log10(Residual Norm)')
    >>> plt.ylabel('log10(Solution Norm)')
    >>> plt.title('L-Curve')
    >>> plt.show()
    """
    lambda_values = np.array(sorted(results.keys()))
    residuals = np.array([results[l]['residual'] for l in lambda_values])
    reg_norms = np.array([np.linalg.norm(results[l]['coefficients'])
                          for l in lambda_values])

    if log_scale:
        residuals = np.log10(residuals + 1e-30)
        reg_norms = np.log10(reg_norms + 1e-30)

    return lambda_values, residuals, reg_norms


def find_optimal_lambda(
    results: Dict[float, Dict],
    method: str = 'l_curve'
) -> float:
    """
    Find optimal regularization parameter from parallel solver results.

    Parameters:
    -----------
    results : dict
        Results from solve_distribution_parallel
    method : str
        Method to use:
        - 'l_curve': Find corner of L-curve (default)
        - 'min_residual': Minimum residual
        - 'gcv': Generalized cross-validation (TODO)

    Returns:
    --------
    optimal_lambda : float
        Optimal regularization parameter

    Example:
    --------
    >>> results = solve_distribution_parallel(A, Phi, signal, lambda_values)
    >>> optimal_lambda = find_optimal_lambda(results, method='l_curve')
    >>> best_result = results[optimal_lambda]
    """
    if method == 'min_residual':
        return min(results.keys(), key=lambda l: results[l]['residual'])

    elif method == 'l_curve':
        # Find corner of L-curve using curvature
        lambdas, residuals, norms = compute_l_curve(results, log_scale=True)

        # Compute curvature (simple approximation)
        if len(lambdas) < 3:
            warnings.warn("Need at least 3 lambda values for L-curve. Using min residual.")
            return min(results.keys(), key=lambda l: results[l]['residual'])

        # Compute second derivative (curvature)
        dx = np.gradient(residuals)
        dy = np.gradient(norms)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

        # Find maximum curvature
        max_curv_idx = np.argmax(curvature)
        return lambdas[max_curv_idx]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'l_curve' or 'min_residual'")


def extract_sparse_coefficients(
    coefficients: np.ndarray,
    threshold: float = 1e-6,
    relative: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract non-zero coefficients and their indices.

    Parameters:
    -----------
    coefficients : np.ndarray
        Coefficient vector from solver
    threshold : float
        Threshold for considering coefficient as non-zero (default: 1e-6)
    relative : bool
        If True, threshold is relative to max coefficient (default: True)

    Returns:
    --------
    indices : np.ndarray
        Indices of non-zero coefficients
    values : np.ndarray
        Values of non-zero coefficients

    Example:
    --------
    >>> result = solve_distribution(A, Phi, signal, lambda_reg=1e-2, regularization='l1')
    >>> indices, values = extract_sparse_coefficients(result['coefficients'], threshold=1e-2)
    >>> print(f"Non-zero coefficients: {len(indices)} out of {len(result['coefficients'])}")
    """
    if relative:
        threshold = threshold * np.max(np.abs(coefficients))

    indices = np.where(np.abs(coefficients) > threshold)[0]
    values = coefficients[indices]

    return indices, values


def compare_regularizations(
    A: np.ndarray,
    Phi: np.ndarray,
    signal: np.ndarray,
    lambda_reg: float = 1e-3,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Compare different regularization methods with the same lambda.

    Parameters:
    -----------
    A, Phi, signal : np.ndarray
        Problem data
    lambda_reg : float
        Regularization parameter to use for all methods
    verbose : bool
        If True, print comparison table

    Returns:
    --------
    results : dict
        Dictionary mapping regularization type -> result dictionary

    Example:
    --------
    >>> comparison = compare_regularizations(A, Phi, signal, lambda_reg=1e-3)
    >>> # Compare sparsity
    >>> for reg_type, result in comparison.items():
    >>>     nnz = np.sum(np.abs(result['coefficients']) > 1e-6)
    >>>     print(f"{reg_type}: {nnz} non-zero coefficients")
    """
    reg_types = ['l1', 'l2', 'none']
    results = {}

    for reg_type in reg_types:
        try:
            results[reg_type] = solve_distribution(
                A, Phi, signal, lambda_reg=lambda_reg,
                regularization=reg_type, verbose=False
            )
        except Exception as e:
            warnings.warn(f"Failed to solve with {reg_type}: {e}")
            continue

    if verbose and results:
        print("\nRegularization Comparison (CLARABEL)")
        print("=" * 70)
        print(f"{'Type':<10} {'Residual':<15} {'Objective':<15} {'Sparsity':<15}")
        print("-" * 70)
        for reg_type, result in results.items():
            nnz = np.sum(np.abs(result['coefficients']) > 1e-6)
            sparsity = f"{nnz}/{len(result['coefficients'])}"
            print(f"{reg_type:<10} {result['residual']:<15.2e} "
                  f"{result['objective']:<15.2e} {sparsity:<15}")
        print("=" * 70)

    return results


# ============================================================================
# Advanced Multi-Hyperparameter Optimization
# ============================================================================

def solve_distribution_advanced(
    A_diagnostics: List[np.ndarray],
    Phi: np.ndarray,
    signals: List[np.ndarray],
    num_slowing_down: int,
    alpha: Union[float, List[float]] = 0.5,
    lambda_sd: float = 1e-3,
    lambda_ext: float = 1e-3,
    regularization: str = 'l1',
    non_negative_constraint: bool = True,
    verbose: bool = False,
    **solver_kwargs
) -> Dict:
    """
    Advanced solver with weighted diagnostics AND basis-specific regularization.

    This implements the complete multi-hyperparameter optimization from the notebooks:
    - Different weighting for different diagnostics (e.g., FIDA vs SSNPA vs NPA)
    - Different regularization for slowing-down vs loss/transport basis functions

    Solves:
        minimize  w_1 * ||A_1 @ Phi @ c - y_1||^2 +
                  w_2 * ||A_2 @ Phi @ c - y_2||^2 +
                  w_3 * ||A_3 @ Phi @ c - y_3||^2 +
                  ... +
                  lambda_sd * reg(c_slowing_down) +
                  lambda_ext * reg(c_loss_transport)
        subject to  Phi @ c >= 0, c >= 0

    Parameters:
    -----------
    A_diagnostics : list of ndarray
        List of diagnostic matrices [A_diag1, A_diag2, ...]
        Example: [A_FIDA, A_SSNPA, A_NPA]
    Phi : np.ndarray
        Combined basis matrix (slowing-down + loss + transport)
        Shape: (n_grid_points, n_basis_total)
    signals : list of ndarray
        List of signal vectors [y_1, y_2, ...]
        Example: [y_FIDA, y_SSNPA, y_NPA]
    num_slowing_down : int
        Number of slowing-down basis functions
        Remaining columns are loss/transport: num_ext = n_basis_total - num_slowing_down
    alpha : float or list of float
        Diagnostic weights. Can be:
        - float: For 2 diagnostics, weight of first (second gets 1-alpha)
          - alpha=1.0: Use only first diagnostic
          - alpha=0.5: Equal weighting
          - alpha=0.0: Use only second diagnostic
        - list of float: For N diagnostics, list of N weights [w1, w2, ..., wN]
          - Weights should sum to 1.0 (will be normalized if not)
          - Example: [0.6, 0.3, 0.1] for 3 diagnostics
        (default: 0.5 for 2 diagnostics, equal for others)
    lambda_sd : float
        Regularization for slowing-down coefficients (default: 1e-3)
    lambda_ext : float
        Regularization for loss/transport coefficients (default: 1e-3)
    regularization : str
        Type of regularization: 'l1', 'l2', or 'none' (default: 'l1')
    non_negative_constraint : bool
        Enforce Phi @ c >= 0 and c >= 0 (default: True)
    verbose : bool
        Print solver output
    **solver_kwargs : dict
        Additional solver arguments

    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'coefficients': Full coefficient vector
        - 'coefficients_sd': Slowing-down coefficients
        - 'coefficients_ext': Loss/transport coefficients
        - 'distribution': Reconstructed distribution
        - 'residual': Total residual
        - 'residuals_per_diagnostic': List of residuals for each diagnostic
        - 'objective': Final objective value
        - 'status': Solver status
        - 'solve_time': Time in seconds
        - 'hyperparameters': Dict of alpha, lambda_sd, lambda_ext
        - 'solver': Solver used ('CLARABEL')

    Example:
    --------
    >>> # Two diagnostics
    >>> result = solve_distribution_advanced(
    ...     A_diagnostics=[A_FIDA, A_SSNPA],
    ...     Phi=Phi_combined,
    ...     signals=[y_FIDA, y_SSNPA],
    ...     num_slowing_down=1640,
    ...     alpha=0.99,           # 99% FIDA, 1% SSNPA
    ...     lambda_sd=5e-6,       # Weak regularization on slowing-down
    ...     lambda_ext=2.5e-3,    # Strong regularization on loss
    ...     regularization='l1'
    ... )
    """
    # Validate inputs
    if len(A_diagnostics) < 1:
        raise ValueError("At least one diagnostic required")

    if len(A_diagnostics) != len(signals):
        raise ValueError(f"Number of diagnostics ({len(A_diagnostics)}) must match signals ({len(signals)})")

    n_basis_total = Phi.shape[1]
    if num_slowing_down > n_basis_total:
        raise ValueError(f"num_slowing_down ({num_slowing_down}) > total basis ({n_basis_total})")

    num_extended = n_basis_total - num_slowing_down

    # Compute A @ Phi for each diagnostic
    A_Phi_list = [A @ Phi for A in A_diagnostics]

    # Normalize for numerical stability
    A_Phi_normalized = []
    for A_Phi in A_Phi_list:
        norm = np.linalg.norm(A_Phi)
        A_Phi_normalized.append(A_Phi / norm if norm > 0 else A_Phi)

    # Define optimization variable
    c = cp.Variable(n_basis_total)

    # Split coefficient vector
    c_sd = c[:num_slowing_down]  # Slowing-down coefficients
    c_ext = c[num_slowing_down:]  # Loss/transport coefficients

    # Process diagnostic weights
    if isinstance(alpha, (list, np.ndarray)):
        # List of weights provided
        weights = np.array(alpha, dtype=float)
        if len(weights) != len(A_diagnostics):
            raise ValueError(f"Length of alpha ({len(weights)}) must match number of diagnostics ({len(A_diagnostics)})")

        # Normalize weights to sum to 1
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            raise ValueError("Sum of weights cannot be zero")
        weights = weights / weight_sum

    else:
        # Single float alpha - handle based on number of diagnostics
        if len(A_diagnostics) == 1:
            weights = np.array([1.0])
        elif len(A_diagnostics) == 2:
            # Traditional alpha weighting: alpha for first, (1-alpha) for second
            weights = np.array([alpha, 1.0 - alpha])
        else:
            # Equal weighting for 3+ diagnostics when single alpha given
            weights = np.ones(len(A_diagnostics)) / len(A_diagnostics)

    # Data fidelity terms with weighted diagnostics
    data_fidelity = sum(
        weight * cp.sum_squares(A_Phi_n @ c - sig)
        for weight, A_Phi_n, sig in zip(weights, A_Phi_normalized, signals)
    )

    # Regularization terms
    reg_type = regularization.lower()

    if reg_type == 'l1':
        if num_extended > 0:
            reg_term = lambda_sd * cp.norm1(c_sd) + lambda_ext * cp.norm1(c_ext)
        else:
            reg_term = lambda_sd * cp.norm1(c_sd)
    elif reg_type == 'l2':
        if num_extended > 0:
            reg_term = lambda_sd * cp.sum_squares(c_sd) + lambda_ext * cp.sum_squares(c_ext)
        else:
            reg_term = lambda_sd * cp.sum_squares(c_sd)
    elif reg_type == 'none':
        reg_term = 0
    else:
        raise ValueError(f"Unknown regularization: {reg_type}")

    # Objective
    objective = cp.Minimize(data_fidelity + reg_term)

    # Constraints
    constraints = []
    if non_negative_constraint:
        constraints.append(Phi @ c >= 0)
        constraints.append(c >= 0)

    # Solve
    problem = cp.Problem(objective, constraints)

    # Get CLARABEL solver kwargs
    clarabel_kwargs = get_clarabel_kwargs(**solver_kwargs)
    if verbose:
        clarabel_kwargs['verbose'] = True

    try:
        problem.solve(solver=cp.CLARABEL, **clarabel_kwargs)
    except cp.error.SolverError as e:
        warnings.warn(f"CLARABEL solver failed: {e}. Trying default solver.")
        problem.solve(verbose=verbose)

    if c.value is None:
        raise ValueError(f"Solver failed with status: {problem.status}")

    # Extract results
    coefficients = c.value
    distribution = Phi @ coefficients

    # Compute residuals for each diagnostic
    residuals_per_diagnostic = []
    for A, signal in zip(A_diagnostics, signals):
        residual = np.linalg.norm(A @ distribution - signal)
        residuals_per_diagnostic.append(residual)

    total_residual = np.sqrt(sum(r**2 for r in residuals_per_diagnostic))

    return {
        'coefficients': coefficients,
        'coefficients_sd': coefficients[:num_slowing_down],
        'coefficients_ext': coefficients[num_slowing_down:] if num_extended > 0 else np.array([]),
        'distribution': distribution,
        'residual': total_residual,
        'residuals_per_diagnostic': residuals_per_diagnostic,
        'objective': problem.value,
        'status': problem.status,
        'solve_time': problem.solver_stats.solve_time if problem.solver_stats else None,
        'solver': 'CLARABEL',
        'hyperparameters': {
            'alpha': alpha,
            'diagnostic_weights': weights.tolist(),  # Normalized weights used
            'lambda_sd': lambda_sd,
            'lambda_ext': lambda_ext,
            'regularization': regularization
        }
    }


def grid_search_hyperparameters(
    A_diagnostics: List[np.ndarray],
    Phi: np.ndarray,
    signals: List[np.ndarray],
    num_slowing_down: int,
    alpha_values: List[float],
    lambda_sd_values: List[float],
    lambda_ext_values: List[float],
    n_jobs: int = 1,
    **kwargs
) -> Dict[Tuple[float, float, float], Dict]:
    """
    Parallel grid search over all three hyperparameters using CLARABEL solver.

    This matches the notebook workflow where you search over:
    - alpha: diagnostic weighting
    - lambda_sd: slowing-down regularization
    - lambda_ext: loss/transport regularization

    Parameters:
    -----------
    A_diagnostics, Phi, signals, num_slowing_down :
        Same as solve_distribution_advanced
    alpha_values : list of float
        Alpha values to try (e.g., [0.98, 0.99])
    lambda_sd_values : list of float
        Slowing-down regularization values (e.g., [3e-7, 7e-7, 1e-6, 5e-6])
    lambda_ext_values : list of float
        Loss/transport regularization values (e.g., [1e-3, 1.5e-3, 2e-3, 2.5e-3])
    n_jobs : int
        Number of parallel jobs (default: 1)
    **kwargs : dict
        Additional arguments for solve_distribution_advanced

    Returns:
    --------
    results : dict
        Dictionary mapping (alpha, lambda_sd, lambda_ext) -> result dict
        Keys are tuples of hyperparameter values

    Example:
    --------
    >>> # Match notebook hyperparameter ranges
    >>> results = grid_search_hyperparameters(
    ...     A_diagnostics=[A_FIDA, A_SSNPA],
    ...     Phi=Phi_combined,
    ...     signals=[y_FIDA, y_SSNPA],
    ...     num_slowing_down=1640,
    ...     alpha_values=[0.98, 0.99],
    ...     lambda_sd_values=[3.0e-7, 7.0e-7, 1.0e-6, 5.0e-6],
    ...     lambda_ext_values=[1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3],
    ...     n_jobs=64  # Parallel processing
    ... )
    """
    import itertools

    # Create all combinations
    param_combinations = list(itertools.product(alpha_values, lambda_sd_values, lambda_ext_values))

    if len(param_combinations) == 0:
        raise ValueError("No parameter combinations to search")

    print(f"Grid search (CLARABEL): {len(param_combinations)} combinations")
    print(f"  Alpha values: {len(alpha_values)}")
    print(f"  Lambda_sd values: {len(lambda_sd_values)}")
    print(f"  Lambda_ext values: {len(lambda_ext_values)}")

    # Worker function
    def solve_single_combo(params):
        alpha_val, lambda_sd_val, lambda_ext_val = params
        try:
            # Filter out verbose from kwargs to avoid conflict
            solve_kwargs = {k: v for k, v in kwargs.items() if k != 'verbose'}
            result = solve_distribution_advanced(
                A_diagnostics, Phi, signals, num_slowing_down,
                alpha=alpha_val,
                lambda_sd=lambda_sd_val,
                lambda_ext=lambda_ext_val,
                verbose=False,
                **solve_kwargs
            )
            return params, result
        except Exception as e:
            # Return failed result
            warnings.warn(f"Failed for params {params}: {e}")
            return params, {'status': f'FAILED: {e}', 'residual': np.inf, 'solver': 'CLARABEL'}

    # Parallel or serial execution
    if HAS_JOBLIB and n_jobs != 1:
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(solve_single_combo)(params)
            for params in param_combinations
        )
    else:
        if n_jobs != 1 and not HAS_JOBLIB:
            warnings.warn("joblib not installed. Running serially.")
        results_list = [solve_single_combo(params) for params in param_combinations]

    # Convert to dictionary
    return {params: result for params, result in results_list}


def analyze_grid_search_results(
    results: Dict[Tuple[float, float, float], Dict],
    metric: str = 'residual'
) -> Dict:
    """
    Analyze grid search results and find best hyperparameters.

    Parameters:
    -----------
    results : dict
        Results from grid_search_hyperparameters
    metric : str
        Metric to optimize: 'residual', 'objective', or custom
        (default: 'residual')

    Returns:
    --------
    analysis : dict
        Dictionary containing:
        - 'best_params': Tuple of (alpha, lambda_sd, lambda_ext)
        - 'best_result': Result dict for best parameters
        - 'all_successful': List of successful parameter combinations
        - 'num_successful': Number of successful solves
        - 'num_failed': Number of failed solves

    Example:
    --------
    >>> analysis = analyze_grid_search_results(results, metric='residual')
    >>> print(f"Best params: alpha={analysis['best_params'][0]:.3f}, "
    ...       f"λ_sd={analysis['best_params'][1]:.1e}, "
    ...       f"λ_ext={analysis['best_params'][2]:.1e}")
    >>> print(f"Best residual: {analysis['best_result']['residual']:.2e}")
    """
    # Filter successful results
    successful = {
        params: res for params, res in results.items()
        if isinstance(res.get('status'), str) and 'optimal' in res['status'].lower()
    }

    if len(successful) == 0:
        warnings.warn("No successful results found!")
        return {
            'best_params': None,
            'best_result': None,
            'all_successful': [],
            'num_successful': 0,
            'num_failed': len(results)
        }

    # Find best by metric
    if metric == 'residual':
        best_params = min(successful.keys(), key=lambda k: successful[k]['residual'])
    elif metric == 'objective':
        best_params = min(successful.keys(), key=lambda k: successful[k]['objective'])
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return {
        'best_params': best_params,
        'best_result': successful[best_params],
        'all_successful': list(successful.keys()),
        'num_successful': len(successful),
        'num_failed': len(results) - len(successful)
    }


# ============================================================================
# Convenience Functions
# ============================================================================

def solve_tomography(
    basis_matrix: np.ndarray,
    diagnostic_matrix: np.ndarray,
    signal: np.ndarray,
    lambda_reg: float = 1e-3,
    **kwargs
) -> Dict:
    """
    Convenience wrapper with alternative parameter names.

    This function uses naming that may be more familiar from tomography literature.

    Parameters:
    -----------
    basis_matrix : np.ndarray
        Basis function matrix (Phi)
    diagnostic_matrix : np.ndarray
        Diagnostic weight function matrix (A)
    signal : np.ndarray
        Measured signal vector (y)
    lambda_reg : float
        Regularization parameter
    **kwargs : dict
        Additional arguments passed to solve_distribution

    Returns:
    --------
    result : dict
        Same as solve_distribution
    """
    return solve_distribution(diagnostic_matrix, basis_matrix, signal, lambda_reg, **kwargs)


if __name__ == "__main__":
    # Example usage and demonstration
    print("CLARABEL Tomography Solver Module")
    print("=" * 70)
    print()
    print("This module uses the CLARABEL solver - a modern Rust-based")
    print("interior point solver for convex optimization.")
    print()

    # Check if CLARABEL is available
    try:
        import clarabel
        print(f"✓ CLARABEL installed")
        print(f"  To install: pip install clarabel")
    except ImportError:
        print("✗ CLARABEL not installed")
        print("  Install with: pip install clarabel")

    print()
    print("Example usage:")
    print("  from clarabel_solver import solve_distribution")
    print("  result = solve_distribution(A, Phi, signal, lambda_reg=1e-3)")
    print()
    print("This solver has the exact same API as cvxpy_solver and jax_solver!")
