"""
Unified Hyperparameter Selection Module

This module provides a uniform interface for selecting optimal regularization
hyperparameters using various methods. It consolidates approaches from the
converted Jupyter notebooks and early October implementations.

Supported Methods:
------------------
1D Parameter Selection (single hyperparameter):
    - 'lcurve': L-curve curvature analysis
    - 'corner': L-curve corner detection

2D Parameter Selection (two hyperparameters):
    - 'lcurve_2d': L-curve analysis for each value of first parameter
    - 'gaussian_curvature': Gaussian curvature on 2D parameter surface
    - 'composite_merit': Weighted combination of residual and regularization
    - 'discrepancy': Match expected noise level with maximum sparsity
    - 'pareto': Multi-objective Pareto front analysis

Usage:
------
    from unified_hyperparameter_selection import HyperparameterSelector

    # For 1D search
    selector = HyperparameterSelector(method='lcurve')
    result = selector.select(results_dict)

    # For 2D search
    selector = HyperparameterSelector(method='gaussian_curvature')
    result = selector.select(
        lambda_vals=lambdas,
        lambda2_vals=lambda2s,
        residuals=residuals_2d,
        regularizations=regularizations_2d
    )

    # Compare all methods
    comparison = HyperparameterSelector.compare_all_methods(
        lambda_vals=lambdas,
        lambda2_vals=lambda2s,
        residuals=residuals_2d,
        regularizations=regularizations_2d
    )
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.interpolate import UnivariateSpline, griddata, RectBivariateSpline
from scipy.spatial.distance import cdist
import warnings

# Optional plotting support
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available - plotting functions will not work")


# ============================================================================
# Base Result Class
# ============================================================================

class SelectionResult:
    """
    Unified result object for hyperparameter selection.

    Attributes:
    -----------
    method : str
        Name of selection method used
    optimal_params : dict
        Dictionary of optimal parameter values
        - 1D case: {'lambda': value}
        - 2D case: {'lambda1': value1, 'lambda2': value2}
    optimal_idx : tuple or int
        Index/indices of optimal point in grid
    metrics : dict
        Additional metrics specific to the method
    metadata : dict
        Method-specific data (curvature arrays, surfaces, etc.)
    """

    def __init__(self, method: str, optimal_params: Dict[str, float],
                 optimal_idx: Union[int, Tuple], metrics: Optional[Dict] = None,
                 metadata: Optional[Dict] = None):
        self.method = method
        self.optimal_params = optimal_params
        self.optimal_idx = optimal_idx
        self.metrics = metrics or {}
        self.metadata = metadata or {}

    def __repr__(self):
        params_str = ", ".join(f"{k}={v:.2e}" for k, v in self.optimal_params.items())
        return f"SelectionResult(method='{self.method}', {params_str})"

    def __str__(self):
        lines = [f"Method: {self.method}"]
        lines.append("Optimal Parameters:")
        for name, value in self.optimal_params.items():
            lines.append(f"  {name} = {value:.4e}")
        if self.metrics:
            lines.append("Metrics:")
            for name, value in self.metrics.items():
                if isinstance(value, (int, float, np.number)):
                    lines.append(f"  {name} = {value:.4e}")
                else:
                    lines.append(f"  {name} = {value}")
        return "\n".join(lines)

    def get_parameter(self, name: str, default=None):
        """Get a specific optimal parameter value."""
        return self.optimal_params.get(name, default)

    def get_metric(self, name: str, default=None):
        """Get a specific metric value."""
        return self.metrics.get(name, default)


# ============================================================================
# Main Hyperparameter Selector Class
# ============================================================================

class HyperparameterSelector:
    """
    Unified interface for hyperparameter selection methods.

    Parameters:
    -----------
    method : str
        Selection method to use. See module docstring for available methods.
    **kwargs : dict
        Method-specific parameters (e.g., noise_level, weights, etc.)

    Examples:
    ---------
    >>> # 1D selection
    >>> selector = HyperparameterSelector(method='lcurve')
    >>> result = selector.select(results_dict)
    >>> print(f"Optimal lambda: {result.optimal_params['lambda']:.2e}")

    >>> # 2D selection with Gaussian curvature
    >>> selector = HyperparameterSelector(method='gaussian_curvature')
    >>> result = selector.select(
    ...     lambda_vals=lambdas, lambda2_vals=lambda2s,
    ...     residuals=res_2d, regularizations=reg_2d
    ... )

    >>> # 2D selection with discrepancy principle
    >>> selector = HyperparameterSelector(
    ...     method='discrepancy',
    ...     noise_level=0.01,
    ...     n_measurements=100
    ... )
    >>> result = selector.select(...)
    """

    # Available methods for each dimensionality
    METHODS_1D = ['lcurve', 'corner']
    METHODS_2D = ['lcurve_2d', 'gaussian_curvature', 'composite_merit',
                  'discrepancy', 'pareto']

    def __init__(self, method: str, **kwargs):
        """Initialize selector with specified method and parameters."""
        all_methods = self.METHODS_1D + self.METHODS_2D
        if method not in all_methods:
            raise ValueError(
                f"Unknown method '{method}'. Available methods:\n"
                f"  1D: {self.METHODS_1D}\n"
                f"  2D: {self.METHODS_2D}"
            )

        self.method = method
        self.params = kwargs

    def select(self,
               results: Optional[Dict] = None,
               lambda_vals: Optional[np.ndarray] = None,
               lambda2_vals: Optional[np.ndarray] = None,
               residuals: Optional[np.ndarray] = None,
               regularizations: Optional[np.ndarray] = None,
               results_list: Optional[List[Tuple]] = None,
               **kwargs) -> SelectionResult:
        """
        Select optimal hyperparameters using configured method.

        Parameters:
        -----------
        results : dict, optional
            For 1D methods: dictionary with lambda as keys, results as values
        lambda_vals : ndarray, optional
            For 2D methods: 1D array of first parameter values
        lambda2_vals : ndarray, optional
            For 2D methods: 1D array of second parameter values
        residuals : ndarray, optional
            For 2D methods: 2D array of residual values
        regularizations : ndarray, optional
            For 2D methods: 2D array of regularization values
        results_list : list of tuples, optional
            Alternative format: [(param1, param2, log_res, log_reg, coeffs, status), ...]
        **kwargs : dict
            Additional method-specific arguments

        Returns:
        --------
        SelectionResult
            Object containing optimal parameters and metadata
        """
        # Merge instance params with call-time kwargs
        method_kwargs = {**self.params, **kwargs}

        # Route to appropriate method
        if self.method in self.METHODS_1D:
            return self._select_1d(results, method_kwargs)
        elif self.method in self.METHODS_2D:
            if results_list is not None:
                return self._select_2d_from_list(results_list, method_kwargs)
            else:
                return self._select_2d(
                    lambda_vals, lambda2_vals, residuals,
                    regularizations, method_kwargs
                )
        else:
            raise RuntimeError(f"Method '{self.method}' not implemented")

    # ========================================================================
    # 1D Selection Methods
    # ========================================================================

    def _select_1d(self, results: Dict, params: Dict) -> SelectionResult:
        """Select optimal parameter from 1D grid search."""
        if results is None or len(results) < 3:
            raise ValueError("Need at least 3 parameter values for 1D selection")

        # Extract data from results dict
        lambda_values = np.array(sorted(results.keys()))
        residuals = np.array([results[l]['residual'] for l in lambda_values])
        reg_norms = np.array([np.linalg.norm(results[l]['coefficients'])
                              for l in lambda_values])

        # Convert to log scale for L-curve
        log_residuals = np.log10(residuals + 1e-30)
        log_reg_norms = np.log10(reg_norms + 1e-30)

        if self.method == 'lcurve':
            # Compute curvature along L-curve
            lambda_sorted, curvature = self._compute_lcurve_curvature(
                lambda_values, log_residuals, log_reg_norms,
                spline_order=params.get('spline_order', 4)
            )

            # Find maximum curvature
            optimal_idx = np.argmax(curvature)
            optimal_lambda = lambda_sorted[optimal_idx]

            return SelectionResult(
                method='L-curve Curvature',
                optimal_params={'lambda': optimal_lambda},
                optimal_idx=optimal_idx,
                metrics={
                    'max_curvature': curvature[optimal_idx],
                    'residual': residuals[optimal_idx],
                    'regularization': reg_norms[optimal_idx]
                },
                metadata={
                    'lambda_values': lambda_sorted,
                    'curvature': curvature,
                    'log_residuals': log_residuals,
                    'log_regularizations': log_reg_norms
                }
            )

        elif self.method == 'corner':
            # Simple corner detection using distance from line connecting endpoints
            # L-curve endpoints in (log_res, log_reg) space
            p1 = np.array([log_residuals[0], log_reg_norms[0]])
            p2 = np.array([log_residuals[-1], log_reg_norms[-1]])

            # Distance from each point to line p1-p2
            distances = []
            for i in range(len(lambda_values)):
                p = np.array([log_residuals[i], log_reg_norms[i]])
                # Distance from point to line
                d = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
                distances.append(d)

            distances = np.array(distances)
            optimal_idx = np.argmax(distances)
            optimal_lambda = lambda_values[optimal_idx]

            return SelectionResult(
                method='L-curve Corner',
                optimal_params={'lambda': optimal_lambda},
                optimal_idx=optimal_idx,
                metrics={
                    'corner_distance': distances[optimal_idx],
                    'residual': residuals[optimal_idx],
                    'regularization': reg_norms[optimal_idx]
                },
                metadata={
                    'lambda_values': lambda_values,
                    'distances': distances
                }
            )

    # ========================================================================
    # 2D Selection Methods
    # ========================================================================

    def _select_2d(self, lambda_vals: np.ndarray, lambda2_vals: np.ndarray,
                   residuals: np.ndarray, regularizations: np.ndarray,
                   params: Dict) -> SelectionResult:
        """Select optimal parameters from 2D grid search."""
        if lambda_vals is None or lambda2_vals is None:
            raise ValueError("lambda_vals and lambda2_vals required for 2D methods")
        if residuals is None or regularizations is None:
            raise ValueError("residuals and regularizations required for 2D methods")

        if self.method == 'gaussian_curvature':
            return self._gaussian_curvature_2d(
                lambda_vals, lambda2_vals, residuals, regularizations, params
            )

        elif self.method == 'composite_merit':
            return self._composite_merit_2d(
                lambda_vals, lambda2_vals, residuals, regularizations, params
            )

        elif self.method == 'discrepancy':
            return self._discrepancy_2d(
                lambda_vals, lambda2_vals, residuals, regularizations, params
            )

        elif self.method == 'pareto':
            return self._pareto_2d(
                lambda_vals, lambda2_vals, residuals, regularizations, params
            )

        elif self.method == 'lcurve_2d':
            return self._lcurve_2d(
                lambda_vals, lambda2_vals, residuals, regularizations, params
            )

        else:
            raise NotImplementedError(f"2D method '{self.method}' not yet implemented")

    def _select_2d_from_list(self, results_list: List[Tuple],
                             params: Dict) -> SelectionResult:
        """Select from list format: [(p1, p2, log_res, log_reg, coeffs, status), ...]."""
        # Filter successful results
        results_list = [r for r in results_list
                       if r[5] in ['optimal', 'optimal_inaccurate',
                                   'OPTIMAL', 'OPTIMAL_INACCURATE']]

        if len(results_list) < 3:
            raise ValueError("Need at least 3 successful results")

        # Extract data
        param1_data = np.array([r[0] for r in results_list])
        param2_data = np.array([r[1] for r in results_list])
        log_res_data = np.array([r[2] for r in results_list])
        log_reg_data = np.array([r[3] for r in results_list])

        # Get unique parameter values
        param1_vals = np.unique(param1_data)
        param2_vals = np.unique(param2_data)

        # Create 2D grids
        residuals_2d = np.full((len(param1_vals), len(param2_vals)), np.nan)
        regularizations_2d = np.full((len(param1_vals), len(param2_vals)), np.nan)

        for r in results_list:
            i = np.where(param1_vals == r[0])[0][0]
            j = np.where(param2_vals == r[1])[0][0]
            residuals_2d[i, j] = 10**r[2]  # Convert from log
            regularizations_2d[i, j] = 10**r[3]

        # Call appropriate 2D method
        return self._select_2d(
            param1_vals, param2_vals, residuals_2d, regularizations_2d, params
        )

    # ========================================================================
    # 2D Method Implementations
    # ========================================================================

    def _gaussian_curvature_2d(self, lambda_vals, lambda2_vals, residuals,
                               regularizations, params) -> SelectionResult:
        """Gaussian curvature on parameter surface."""
        metric_name = params.get('metric', 'residual')
        grid_resolution = params.get('grid_resolution', 100)

        # Choose which surface to analyze
        if metric_name == 'residual':
            metric_data = residuals
        elif metric_name == 'regularization':
            metric_data = regularizations
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        # Work in log space
        log_metric = np.log10(np.clip(metric_data, 1e-30, None))

        # Compute Gaussian curvature
        grid_p1, grid_p2, grid_metric, K = self._compute_gaussian_curvature(
            lambda_vals, lambda2_vals, log_metric, grid_resolution
        )

        # Find maximum curvature
        max_index = np.nanargmax(np.abs(K))
        optimal_lambda = grid_p1.ravel()[max_index]
        optimal_lambda2 = grid_p2.ravel()[max_index]

        # Find nearest original grid point
        i = np.argmin(np.abs(lambda_vals - optimal_lambda))
        j = np.argmin(np.abs(lambda2_vals - optimal_lambda2))
        optimal_lambda = lambda_vals[i]
        optimal_lambda2 = lambda2_vals[j]

        return SelectionResult(
            method='Gaussian Curvature',
            optimal_params={'lambda1': optimal_lambda, 'lambda2': optimal_lambda2},
            optimal_idx=(i, j),
            metrics={
                'max_curvature': K.ravel()[max_index],
                'residual': residuals[i, j],
                'regularization': regularizations[i, j]
            },
            metadata={
                'curvature_surface': K,
                'grid_lambda1': grid_p1,
                'grid_lambda2': grid_p2,
                'grid_metric': grid_metric
            }
        )

    def _composite_merit_2d(self, lambda_vals, lambda2_vals, residuals,
                           regularizations, params) -> SelectionResult:
        """Composite merit function combining multiple objectives."""
        weights = params.get('weights', (0.45, 0.45, 0.10))
        penalize_extreme = params.get('penalize_extreme_lambda2', False)

        w_res, w_reg, w_bal = weights

        # Normalize to [0, 1]
        res_norm = (residuals - residuals.min()) / (residuals.max() - residuals.min())
        reg_norm = (regularizations - regularizations.min()) / \
                   (regularizations.max() - regularizations.min())

        # Diagnostic balance penalty (assumes lambda2 in [0, 1])
        lambda2_grid = lambda2_vals[np.newaxis, :]
        balance_penalty = np.abs(lambda2_grid - 0.5) * 2

        # Compute merit
        if penalize_extreme:
            merit = w_res * res_norm + w_reg * reg_norm + w_bal * balance_penalty
        else:
            merit = w_res * res_norm + w_reg * reg_norm

        # Find minimum
        i, j = np.unravel_index(np.argmin(merit), merit.shape)

        return SelectionResult(
            method='Composite Merit Function',
            optimal_params={'lambda1': lambda_vals[i], 'lambda2': lambda2_vals[j]},
            optimal_idx=(i, j),
            metrics={
                'merit_value': merit[i, j],
                'residual': residuals[i, j],
                'regularization': regularizations[i, j],
                'weights': weights
            },
            metadata={
                'merit_surface': merit,
                'penalize_extreme': penalize_extreme
            }
        )

    def _lcurve_2d(self, lambda_vals, lambda2_vals, residuals,
                   regularizations, params) -> SelectionResult:
        """L-curve analysis for each value of first parameter."""
        # For each lambda1 value, compute L-curve over lambda2 values
        # Then select the lambda1 with overall maximum curvature

        optimal_lambda1 = None
        optimal_lambda2 = None
        max_curvature_overall = -np.inf

        log_residuals = np.log10(np.clip(residuals, 1e-30, None))
        log_regularizations = np.log10(np.clip(regularizations, 1e-30, None))

        for i, lambda1 in enumerate(lambda_vals):
            # Extract L-curve data for this lambda1 value
            log_res_curve = log_residuals[i, :]
            log_reg_curve = log_regularizations[i, :]

            # Remove NaN values
            valid_mask = ~(np.isnan(log_res_curve) | np.isnan(log_reg_curve))
            if np.sum(valid_mask) < 3:
                continue

            lambda2_valid = lambda2_vals[valid_mask]
            log_res_valid = log_res_curve[valid_mask]
            log_reg_valid = log_reg_curve[valid_mask]

            # Compute curvature for this L-curve
            try:
                lambda2_sorted, curvature = self._compute_lcurve_curvature(
                    lambda2_valid, log_res_valid, log_reg_valid
                )

                max_idx = np.argmax(curvature)
                max_curv = curvature[max_idx]

                if max_curv > max_curvature_overall:
                    max_curvature_overall = max_curv
                    optimal_lambda1 = lambda1
                    optimal_lambda2 = lambda2_sorted[max_idx]
            except:
                continue

        if optimal_lambda1 is None:
            raise ValueError("Could not find optimal parameters with lcurve_2d method")

        # Find indices
        i = np.argmin(np.abs(lambda_vals - optimal_lambda1))
        j = np.argmin(np.abs(lambda2_vals - optimal_lambda2))

        return SelectionResult(
            method='L-curve 2D',
            optimal_params={'lambda1': optimal_lambda1, 'lambda2': optimal_lambda2},
            optimal_idx=(i, j),
            metrics={
                'max_curvature': max_curvature_overall,
                'residual': residuals[i, j],
                'regularization': regularizations[i, j]
            },
            metadata={}
        )

    def _discrepancy_2d(self, lambda_vals, lambda2_vals, residuals,
                       regularizations, params) -> SelectionResult:
        """Discrepancy principle: match expected noise level."""
        noise_level = params.get('noise_level')
        n_measurements = params.get('n_measurements')

        if noise_level is None or n_measurements is None:
            raise ValueError("discrepancy method requires 'noise_level' and 'n_measurements'")

        # Expected residual from noise
        expected_residual = n_measurements * noise_level**2

        # Find points matching expected residual
        tolerance = params.get('tolerance', 0.2)
        residual_mask = np.abs(residuals - expected_residual) / expected_residual < tolerance

        if not np.any(residual_mask):
            # Expand tolerance
            tolerance = 0.5
            residual_mask = np.abs(residuals - expected_residual) / expected_residual < tolerance

        if not np.any(residual_mask):
            # Find closest point
            i, j = np.unravel_index(
                np.argmin(np.abs(residuals - expected_residual)),
                residuals.shape
            )
        else:
            # Among matching, find minimum regularization
            masked_reg = np.where(residual_mask, regularizations, np.inf)
            i, j = np.unravel_index(np.argmin(masked_reg), regularizations.shape)

        return SelectionResult(
            method='Discrepancy Principle',
            optimal_params={'lambda1': lambda_vals[i], 'lambda2': lambda2_vals[j]},
            optimal_idx=(i, j),
            metrics={
                'expected_residual': expected_residual,
                'actual_residual': residuals[i, j],
                'regularization': regularizations[i, j],
                'tolerance': tolerance
            },
            metadata={
                'residual_mask': residual_mask
            }
        )

    def _pareto_2d(self, lambda_vals, lambda2_vals, residuals,
                  regularizations, params) -> SelectionResult:
        """Pareto front analysis for multi-objective optimization."""
        # Flatten 2D arrays to points
        n_lambda, n_lambda2 = residuals.shape
        points = []
        indices = []

        for i in range(n_lambda):
            for j in range(n_lambda2):
                if not (np.isnan(residuals[i, j]) or np.isnan(regularizations[i, j])):
                    points.append([residuals[i, j], regularizations[i, j]])
                    indices.append((i, j))

        points = np.array(points)

        # Find Pareto front
        is_pareto = self._find_pareto_front(points)
        pareto_points = points[is_pareto]
        pareto_indices = [indices[i] for i in range(len(indices)) if is_pareto[i]]

        # Use TOPSIS to rank Pareto points
        if len(pareto_points) > 1:
            # Ideal point: min of each objective
            ideal = np.min(pareto_points, axis=0)
            # Anti-ideal point: max of each objective
            anti_ideal = np.max(pareto_points, axis=0)

            # Normalize
            norm_points = (pareto_points - ideal) / (anti_ideal - ideal + 1e-30)

            # Distance to ideal and anti-ideal
            dist_ideal = np.linalg.norm(norm_points, axis=1)
            dist_anti_ideal = np.linalg.norm(norm_points - 1, axis=1)

            # TOPSIS score
            scores = dist_anti_ideal / (dist_ideal + dist_anti_ideal + 1e-30)
            best_idx = np.argmax(scores)
        else:
            best_idx = 0

        i, j = pareto_indices[best_idx]

        return SelectionResult(
            method='Pareto Front Analysis',
            optimal_params={'lambda1': lambda_vals[i], 'lambda2': lambda2_vals[j]},
            optimal_idx=(i, j),
            metrics={
                'residual': residuals[i, j],
                'regularization': regularizations[i, j],
                'n_pareto_points': len(pareto_points)
            },
            metadata={
                'pareto_front': pareto_points,
                'pareto_indices': pareto_indices,
                'is_pareto': is_pareto
            }
        )

    # ========================================================================
    # Utility Methods
    # ========================================================================

    @staticmethod
    def _compute_lcurve_curvature(lambda_values, log_residuals, log_regularizations,
                                  spline_order=4):
        """Compute curvature along L-curve."""
        sorted_indices = np.argsort(lambda_values)
        lambda_sorted = np.array(lambda_values)[sorted_indices]
        log_res = np.array(log_residuals)[sorted_indices]
        log_reg = np.array(log_regularizations)[sorted_indices]

        if len(lambda_sorted) < spline_order + 1:
            spline_order = max(1, len(lambda_sorted) - 1)

        try:
            spline_x = UnivariateSpline(lambda_sorted, log_res, k=spline_order, s=0)
            spline_y = UnivariateSpline(lambda_sorted, log_reg, k=spline_order, s=0)

            x_prime = spline_x.derivative(1)(lambda_sorted)
            y_prime = spline_y.derivative(1)(lambda_sorted)
            x_double_prime = spline_x.derivative(2)(lambda_sorted)
            y_double_prime = spline_y.derivative(2)(lambda_sorted)

            curvature = np.abs(x_prime * y_double_prime - y_prime * x_double_prime) / \
                       (x_prime**2 + y_prime**2)**(3/2)
        except:
            # Fallback to finite differences
            dx = np.gradient(log_res)
            dy = np.gradient(log_reg)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

        return lambda_sorted, curvature

    @staticmethod
    def _compute_gaussian_curvature(param1_vals, param2_vals, metric_data,
                                    grid_resolution=100):
        """Compute Gaussian curvature of 2D surface."""
        # Create meshgrid
        p1_lin = np.linspace(param1_vals.min(), param1_vals.max(), grid_resolution)
        p2_lin = np.linspace(param2_vals.min(), param2_vals.max(), grid_resolution)
        grid_p1, grid_p2 = np.meshgrid(p1_lin, p2_lin, indexing='ij')

        # Interpolate metric on grid
        P1, P2 = np.meshgrid(param1_vals, param2_vals, indexing='ij')
        points = np.column_stack([P1.ravel(), P2.ravel()])
        values = metric_data.ravel()

        # Remove NaN values
        valid_mask = ~np.isnan(values)
        points = points[valid_mask]
        values = values[valid_mask]

        grid_metric = griddata(points, values, (grid_p1, grid_p2), method='cubic')

        # Compute derivatives
        dp1 = p1_lin[1] - p1_lin[0]
        dp2 = p2_lin[1] - p2_lin[0]

        f_p1, f_p2 = np.gradient(grid_metric, dp1, dp2)
        f_p1_p1 = np.gradient(f_p1, dp1, axis=0)
        f_p2_p2 = np.gradient(f_p2, dp2, axis=1)
        f_p1_p2 = np.gradient(f_p1, dp2, axis=1)

        # Gaussian curvature
        K = (f_p1_p1 * f_p2_p2 - f_p1_p2**2) / (1 + f_p1**2 + f_p2**2)**2

        return grid_p1, grid_p2, grid_metric, K

    @staticmethod
    def _find_pareto_front(points):
        """Find Pareto-optimal points (minimization)."""
        is_pareto = np.ones(len(points), dtype=bool)
        for i, p in enumerate(points):
            if is_pareto[i]:
                # Check if any other point dominates this one
                # Point A dominates B if A <= B in all objectives and A < B in at least one
                dominated = np.all(points <= p, axis=1) & np.any(points < p, axis=1)
                is_pareto[dominated] = False
        return is_pareto

    # ========================================================================
    # Comparison and Visualization
    # ========================================================================

    @staticmethod
    def compare_all_methods(lambda_vals=None, lambda2_vals=None,
                           residuals=None, regularizations=None,
                           results_dict=None, results_list=None,
                           methods='auto', **method_params) -> Dict[str, SelectionResult]:
        """
        Compare results from all applicable methods.

        Parameters:
        -----------
        lambda_vals, lambda2_vals : ndarray, optional
            Parameter arrays for 2D methods
        residuals, regularizations : ndarray, optional
            Metric arrays for 2D methods
        results_dict : dict, optional
            Results dictionary for 1D methods
        results_list : list, optional
            Results list for 2D methods
        methods : str or list, optional
            'auto' (default), '1d', '2d', or list of specific methods
        **method_params : dict
            Method-specific parameters (e.g., noise_level=0.01)

        Returns:
        --------
        dict
            Dictionary mapping method names to SelectionResult objects

        Example:
        --------
        >>> results = HyperparameterSelector.compare_all_methods(
        ...     lambda_vals=lambdas, lambda2_vals=lambda2s,
        ...     residuals=res_2d, regularizations=reg_2d,
        ...     noise_level=0.01, n_measurements=100
        ... )
        >>> for name, result in results.items():
        ...     print(f"{name}: lambda1={result.optimal_params['lambda1']:.2e}")
        """
        comparison = {}

        # Determine which methods to run
        if methods == 'auto':
            if results_dict is not None:
                methods = HyperparameterSelector.METHODS_1D
            elif lambda_vals is not None and lambda2_vals is not None:
                methods = HyperparameterSelector.METHODS_2D
            else:
                raise ValueError("Cannot auto-detect methods - provide appropriate inputs")
        elif methods == '1d':
            methods = HyperparameterSelector.METHODS_1D
        elif methods == '2d':
            methods = HyperparameterSelector.METHODS_2D
        elif isinstance(methods, str):
            methods = [methods]

        # Run each method
        for method in methods:
            try:
                selector = HyperparameterSelector(method, **method_params)

                if method in HyperparameterSelector.METHODS_1D:
                    result = selector.select(results=results_dict)
                else:
                    if results_list is not None:
                        result = selector.select(results_list=results_list)
                    else:
                        result = selector.select(
                            lambda_vals=lambda_vals,
                            lambda2_vals=lambda2_vals,
                            residuals=residuals,
                            regularizations=regularizations
                        )

                comparison[method] = result

            except Exception as e:
                warnings.warn(f"Method '{method}' failed: {e}")
                continue

        return comparison

    def plot_result(self, result: SelectionResult, **plot_kwargs):
        """
        Plot the selection result.

        Parameters:
        -----------
        result : SelectionResult
            Result object from select() method
        **plot_kwargs : dict
            Plotting options (figsize, save_path, etc.)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        if self.method in self.METHODS_1D:
            self._plot_1d_result(result, **plot_kwargs)
        elif self.method in self.METHODS_2D:
            self._plot_2d_result(result, **plot_kwargs)

    def _plot_1d_result(self, result: SelectionResult, **kwargs):
        """Plot 1D selection result with L-curve."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=kwargs.get('figsize', (12, 5)))

        # L-curve
        log_res = result.metadata.get('log_residuals')
        log_reg = result.metadata.get('log_regularizations')

        if log_res is not None and log_reg is not None:
            ax1.plot(log_res, log_reg, 'bo-', label='L-curve')

            # Mark optimal point
            opt_idx = result.optimal_idx
            ax1.plot(log_res[opt_idx], log_reg[opt_idx], 'r*',
                    markersize=15, label='Optimal')

            ax1.set_xlabel('log10(Residual)')
            ax1.set_ylabel('log10(Regularization)')
            ax1.set_title('L-Curve')
            ax1.legend()
            ax1.grid(True)

        # Curvature or distance plot
        lambda_vals = result.metadata.get('lambda_values')
        if 'curvature' in result.metadata:
            curvature = result.metadata['curvature']
            ax2.semilogx(lambda_vals, curvature, 'bo-')
            ax2.axvline(result.optimal_params['lambda'], color='r',
                       linestyle='--', label=f"λ={result.optimal_params['lambda']:.2e}")
            ax2.set_ylabel('Curvature')
        elif 'distances' in result.metadata:
            distances = result.metadata['distances']
            ax2.semilogx(lambda_vals, distances, 'bo-')
            ax2.axvline(result.optimal_params['lambda'], color='r',
                       linestyle='--', label=f"λ={result.optimal_params['lambda']:.2e}")
            ax2.set_ylabel('Distance from Line')

        ax2.set_xlabel('Lambda')
        ax2.set_title(result.method)
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if 'save_path' in kwargs:
            plt.savefig(kwargs['save_path'], dpi=kwargs.get('dpi', 150))

        if kwargs.get('show', True):
            plt.show()

    def _plot_2d_result(self, result: SelectionResult, **kwargs):
        """Plot 2D selection result as contour map."""
        # This would show the metric surface with optimal point marked
        # Implementation depends on what's stored in metadata
        fig = plt.figure(figsize=kwargs.get('figsize', (10, 8)))

        # Extract indices
        i, j = result.optimal_idx

        # Plot depends on what's available in metadata
        if 'merit_surface' in result.metadata:
            surface = result.metadata['merit_surface']
            title = 'Merit Function Surface'
        elif 'curvature_surface' in result.metadata:
            surface = np.abs(result.metadata['curvature_surface'])
            title = 'Curvature Surface'
        else:
            print("No surface data available for plotting")
            return

        # Get lambda values from optimal_params keys
        param_names = list(result.optimal_params.keys())

        plt.contourf(surface, levels=20, cmap='viridis')
        plt.colorbar(label=title)
        plt.plot(j, i, 'r*', markersize=15, label='Optimal')
        plt.xlabel(param_names[1] if len(param_names) > 1 else 'Parameter 2')
        plt.ylabel(param_names[0] if len(param_names) > 0 else 'Parameter 1')
        plt.title(f"{result.method}\n{param_names[0]}={result.optimal_params[param_names[0]]:.2e}, "
                 f"{param_names[1]}={result.optimal_params[param_names[1]]:.2e}")
        plt.legend()

        if 'save_path' in kwargs:
            plt.savefig(kwargs['save_path'], dpi=kwargs.get('dpi', 150))

        if kwargs.get('show', True):
            plt.show()


# ============================================================================
# Convenience Functions
# ============================================================================

def select_optimal_hyperparameters(method='auto', **kwargs) -> SelectionResult:
    """
    Convenience function for quick hyperparameter selection.

    Parameters:
    -----------
    method : str
        Selection method or 'auto' to auto-detect
    **kwargs : dict
        Arguments passed to HyperparameterSelector

    Returns:
    --------
    SelectionResult
        Optimal hyperparameters

    Example:
    --------
    >>> result = select_optimal_hyperparameters(
    ...     method='lcurve',
    ...     results=results_dict
    ... )
    """
    if method == 'auto':
        # Try to auto-detect dimensionality
        if 'results' in kwargs and isinstance(kwargs['results'], dict):
            method = 'lcurve'
        elif 'lambda_vals' in kwargs and 'lambda2_vals' in kwargs:
            method = 'gaussian_curvature'
        else:
            raise ValueError("Cannot auto-detect method - specify explicitly")

    selector = HyperparameterSelector(method)
    return selector.select(**kwargs)


if __name__ == "__main__":
    print("=" * 70)
    print("Unified Hyperparameter Selection Module")
    print("=" * 70)
    print()
    print("Available Methods:")
    print()
    print("1D Parameter Selection:")
    for method in HyperparameterSelector.METHODS_1D:
        print(f"  - {method}")
    print()
    print("2D Parameter Selection:")
    for method in HyperparameterSelector.METHODS_2D:
        print(f"  - {method}")
    print()
    print("Usage:")
    print("  from unified_hyperparameter_selection import HyperparameterSelector")
    print()
    print("  # Single method")
    print("  selector = HyperparameterSelector(method='lcurve')")
    print("  result = selector.select(results=results_dict)")
    print()
    print("  # Compare all methods")
    print("  comparison = HyperparameterSelector.compare_all_methods(...)")
    print()
    print("For detailed examples, see HYPERPARAMETER_SELECTION_USAGE.md")
    print("=" * 70)
