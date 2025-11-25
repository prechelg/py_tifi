"""
Combined Basis Function Module

This module provides functions to create and combine different types of basis functions
for fast ion distribution tomography:

1. Slowing-down basis functions (from slowing_down_basis.py)
2. Loss basis functions (particles lost at specific locations)
3. Transport basis functions (particles transported from one location to another)

The module creates combined distribution matrices by horizontally stacking these basis
functions for use in tomographic inversions and distribution reconstruction.

Usage:
    from combined_basis import create_loss_basis, create_transport_basis, combine_basis

    # Create slowing down basis
    from slowing_down_basis import generate_simple_basis
    Phi = generate_simple_basis(E_values, p_values, ...)

    # Create loss basis
    loss_locations = [(96e3, -0.69), (95e3, -0.7)]
    LossPhi = create_loss_basis(E_values, p_values, loss_locations)

    # Combine them
    Phi_combined = combine_basis(Phi, LossPhi)
"""

import numpy as np
from typing import List, Tuple, Optional, Union


def _find_grid_index(E_values: np.ndarray, p_values: np.ndarray,
                     E_target: float, p_target: float) -> int:
    """
    Find the linear index in the flattened grid for a given (E, p) coordinate.

    Args:
        E_values: Energy grid values (eV)
        p_values: Pitch grid values (dimensionless, -1 to 1)
        E_target: Target energy (eV)
        p_target: Target pitch (dimensionless)

    Returns:
        Linear index in flattened grid

    Note:
        The linear index follows the convention: linear_index = j * len(E_values) + i
        where i is the energy index and j is the pitch index.
    """
    i = np.argmin(np.abs(E_values - E_target))
    j = np.argmin(np.abs(p_values - p_target))
    linear_index = j * len(E_values) + i
    return linear_index


def create_loss_basis(E_values: np.ndarray, p_values: np.ndarray,
                      loss_locations: List[Tuple[float, float]],
                      normalize: bool = True) -> np.ndarray:
    """
    Create loss basis functions for specified (E, p) locations.

    Loss basis functions represent particles that are lost from the distribution
    at specific energy and pitch values. Each loss location creates a basis function
    with a -1 value at that grid point.

    Args:
        E_values: Energy grid values (eV), shape (N_E,)
        p_values: Pitch grid values (dimensionless, -1 to 1), shape (N_p,)
        loss_locations: List of (E, p) tuples specifying loss locations in eV and pitch
        normalize: If True, normalize each column to unit norm (default: True)

    Returns:
        Loss basis matrix of shape (num_grid_points, num_loss_locations)
        where num_grid_points = len(E_values) * len(p_values)

    Example:
        >>> E_values = np.linspace(1e3, 150e3, 40)
        >>> p_values = np.linspace(-0.99, 0.99, 41)
        >>> loss_locations = [(96e3, -0.69), (95e3, -0.7)]
        >>> LossPhi = create_loss_basis(E_values, p_values, loss_locations)
        >>> print(LossPhi.shape)  # (1640, 2)
    """
    num_grid_points = len(E_values) * len(p_values)
    num_loss_bases = len(loss_locations)

    LossPhi = np.zeros((num_grid_points, num_loss_bases))

    for col_idx, (E_target, p_target) in enumerate(loss_locations):
        linear_index = _find_grid_index(E_values, p_values, E_target, p_target)
        LossPhi[linear_index, col_idx] = -1.0

        if normalize:
            LossPhi[:, col_idx] /= np.linalg.norm(LossPhi[:, col_idx])

    return LossPhi


def create_transport_basis(E_values: np.ndarray, p_values: np.ndarray,
                           transport_locations: List[Tuple[Tuple[float, float],
                                                           Tuple[float, float]]],
                           normalize: bool = True) -> np.ndarray:
    """
    Create transport basis functions for specified transport paths.

    Transport basis functions represent particles that move from one (E, p) location
    to another. Each transport path creates a basis function with -1 at the start
    location and +1 at the end location.

    Args:
        E_values: Energy grid values (eV), shape (N_E,)
        p_values: Pitch grid values (dimensionless, -1 to 1), shape (N_p,)
        transport_locations: List of ((E_start, p_start), (E_end, p_end)) tuples
                            specifying transport paths in eV and pitch
        normalize: If True, normalize each column to unit norm (default: True)

    Returns:
        Transport basis matrix of shape (num_grid_points, num_transport_paths)
        where num_grid_points = len(E_values) * len(p_values)

    Example:
        >>> E_values = np.linspace(1e3, 150e3, 40)
        >>> p_values = np.linspace(-0.99, 0.99, 41)
        >>> transport_locations = [
        ...     ((48e3, -0.7), (65e3, -0.4)),
        ...     ((47e3, -0.7), (64e3, -0.4))
        ... ]
        >>> TransportPhi = create_transport_basis(E_values, p_values, transport_locations)
        >>> print(TransportPhi.shape)  # (1640, 2)
    """
    num_grid_points = len(E_values) * len(p_values)
    num_transport_bases = len(transport_locations)

    TransportPhi = np.zeros((num_grid_points, num_transport_bases))

    for col_idx, ((E_start, p_start), (E_end, p_end)) in enumerate(transport_locations):
        start_index = _find_grid_index(E_values, p_values, E_start, p_start)
        end_index = _find_grid_index(E_values, p_values, E_end, p_end)

        TransportPhi[start_index, col_idx] = -1.0
        TransportPhi[end_index, col_idx] = 1.0

        if normalize:
            TransportPhi[:, col_idx] /= np.linalg.norm(TransportPhi[:, col_idx])

    return TransportPhi


def combine_basis(*basis_matrices: np.ndarray) -> np.ndarray:
    """
    Combine multiple basis matrices by horizontal stacking.

    Args:
        *basis_matrices: Variable number of basis matrices to combine.
                        Each should have shape (num_grid_points, num_basis_functions)

    Returns:
        Combined basis matrix with all input matrices stacked horizontally

    Raises:
        ValueError: If input matrices have different numbers of rows

    Example:
        >>> Phi = generate_simple_basis(E_values, p_values, ...)  # (1640, 1640)
        >>> LossPhi = create_loss_basis(E_values, p_values, [...])  # (1640, 2)
        >>> TransportPhi = create_transport_basis(E_values, p_values, [...])  # (1640, 3)
        >>> Phi_combined = combine_basis(Phi, LossPhi, TransportPhi)  # (1640, 1645)
    """
    if len(basis_matrices) == 0:
        raise ValueError("At least one basis matrix must be provided")

    # Check that all matrices have the same number of rows
    num_rows = basis_matrices[0].shape[0]
    for i, matrix in enumerate(basis_matrices[1:], start=1):
        if matrix.shape[0] != num_rows:
            raise ValueError(f"Matrix {i} has {matrix.shape[0]} rows, expected {num_rows}")

    return np.hstack(basis_matrices)


def create_combined_distribution(
    E_values: np.ndarray,
    p_values: np.ndarray,
    slowing_down_basis: np.ndarray,
    loss_locations: Optional[List[Tuple[float, float]]] = None,
    transport_locations: Optional[List[Tuple[Tuple[float, float],
                                            Tuple[float, float]]]] = None,
    normalize_extended: bool = True
) -> np.ndarray:
    """
    Create a fully combined distribution matrix with slowing down, loss, and/or
    transport basis functions.

    This is a convenience function that combines all steps into one call.

    Args:
        E_values: Energy grid values (eV), shape (N_E,)
        p_values: Pitch grid values (dimensionless, -1 to 1), shape (N_p,)
        slowing_down_basis: Pre-computed slowing down basis matrix Phi,
                           shape (num_grid_points, num_grid_points)
        loss_locations: Optional list of (E, p) tuples for loss basis functions
        transport_locations: Optional list of ((E_start, p_start), (E_end, p_end))
                           tuples for transport basis functions
        normalize_extended: If True, normalize loss and transport basis functions
                           (default: True)

    Returns:
        Combined basis matrix with slowing down and any specified extended basis
        functions stacked horizontally

    Example:
        >>> from slowing_down_basis import generate_simple_basis
        >>>
        >>> E_values = np.linspace(1e3, 150e3, 40)
        >>> p_values = np.linspace(-0.99, 0.99, 41)
        >>>
        >>> # Create slowing down basis
        >>> Phi = generate_simple_basis(E_values, p_values, S0=1e20, tau_s=1e-3)
        >>>
        >>> # Define extended basis locations
        >>> loss_locs = [(96e3, -0.69), (95e3, -0.7)]
        >>> transport_locs = [((48e3, -0.7), (65e3, -0.4))]
        >>>
        >>> # Create combined distribution
        >>> Phi_combined = create_combined_distribution(
        ...     E_values, p_values, Phi,
        ...     loss_locations=loss_locs,
        ...     transport_locations=transport_locs
        ... )
        >>> print(Phi_combined.shape)  # (1640, 1643)
    """
    basis_list = [slowing_down_basis]

    if loss_locations is not None and len(loss_locations) > 0:
        LossPhi = create_loss_basis(E_values, p_values, loss_locations,
                                     normalize=normalize_extended)
        basis_list.append(LossPhi)

    if transport_locations is not None and len(transport_locations) > 0:
        TransportPhi = create_transport_basis(E_values, p_values, transport_locations,
                                              normalize=normalize_extended)
        basis_list.append(TransportPhi)

    return combine_basis(*basis_list)


def extract_basis_coefficients(
    coefficients: np.ndarray,
    num_slowing_down: int,
    num_loss: int = 0,
    num_transport: int = 0,
    slowing_down_threshold: float = 0.0,
    loss_threshold: float = 0.0,
    transport_threshold: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and threshold coefficients for different basis function types from
    a combined coefficient vector.

    This is useful after solving an inverse problem to separate the contributions
    from slowing down, loss, and transport basis functions.

    Args:
        coefficients: Combined coefficient vector from optimization
        num_slowing_down: Number of slowing down basis functions
        num_loss: Number of loss basis functions (default: 0)
        num_transport: Number of transport basis functions (default: 0)
        slowing_down_threshold: Threshold for slowing down coefficients (default: 0.0)
        loss_threshold: Threshold for loss coefficients (default: 0.0)
        transport_threshold: Threshold for transport coefficients (default: 0.0)

    Returns:
        Tuple of (slowing_down_coeffs, loss_coeffs, transport_coeffs)
        Each is a numpy array with coefficients below threshold set to zero

    Example:
        >>> # After solving optimization problem
        >>> c_opt = problem.solve()  # Shape: (1643,)
        >>>
        >>> # Extract coefficients
        >>> sd_coeffs, loss_coeffs, transp_coeffs = extract_basis_coefficients(
        ...     c_opt, num_slowing_down=1640, num_loss=2, num_transport=1,
        ...     slowing_down_threshold=1e-2, transport_threshold=1e-4
        ... )
    """
    total_expected = num_slowing_down + num_loss + num_transport
    if len(coefficients) != total_expected:
        raise ValueError(f"Coefficient vector has {len(coefficients)} elements, "
                        f"expected {total_expected}")

    # Extract slowing down coefficients
    sd_coeffs = coefficients[:num_slowing_down].copy()
    sd_coeffs[np.abs(sd_coeffs) < slowing_down_threshold] = 0.0

    # Extract loss coefficients
    if num_loss > 0:
        loss_start = num_slowing_down
        loss_end = num_slowing_down + num_loss
        loss_coeffs = coefficients[loss_start:loss_end].copy()
        loss_coeffs[np.abs(loss_coeffs) < loss_threshold] = 0.0
    else:
        loss_coeffs = np.array([])

    # Extract transport coefficients
    if num_transport > 0:
        transport_start = num_slowing_down + num_loss
        transport_coeffs = coefficients[transport_start:].copy()
        transport_coeffs[np.abs(transport_coeffs) < transport_threshold] = 0.0
    else:
        transport_coeffs = np.array([])

    return sd_coeffs, loss_coeffs, transport_coeffs


def reconstruct_distributions(
    E_values: np.ndarray,
    p_values: np.ndarray,
    slowing_down_basis: np.ndarray,
    slowing_down_coeffs: np.ndarray,
    loss_basis: Optional[np.ndarray] = None,
    loss_coeffs: Optional[np.ndarray] = None,
    transport_basis: Optional[np.ndarray] = None,
    transport_coeffs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct separate distribution components from basis functions and coefficients.

    Args:
        E_values: Energy grid values (eV), shape (N_E,)
        p_values: Pitch grid values (dimensionless, -1 to 1), shape (N_p,)
        slowing_down_basis: Slowing down basis matrix
        slowing_down_coeffs: Coefficients for slowing down basis
        loss_basis: Optional loss basis matrix
        loss_coeffs: Optional coefficients for loss basis
        transport_basis: Optional transport basis matrix
        transport_coeffs: Optional coefficients for transport basis

    Returns:
        Tuple of (total_dist, sd_dist, loss_dist, transport_dist)
        Each is a 2D array shaped (len(p_values), len(E_values))

    Example:
        >>> total, sd, loss, transp = reconstruct_distributions(
        ...     E_values, p_values, Phi, sd_coeffs,
        ...     LossPhi, loss_coeffs, TransportPhi, transp_coeffs
        ... )
        >>> # Plot the total distribution
        >>> plt.pcolormesh(E_values/1e3, p_values, total)
    """
    # Reconstruct slowing down component
    sd_dist_flat = slowing_down_basis @ slowing_down_coeffs
    sd_dist = sd_dist_flat.reshape(len(p_values), len(E_values))

    # Reconstruct loss component
    if loss_basis is not None and loss_coeffs is not None:
        loss_dist_flat = loss_basis @ loss_coeffs
        loss_dist = loss_dist_flat.reshape(len(p_values), len(E_values))
    else:
        loss_dist = np.zeros((len(p_values), len(E_values)))

    # Reconstruct transport component
    if transport_basis is not None and transport_coeffs is not None:
        transport_dist_flat = transport_basis @ transport_coeffs
        transport_dist = transport_dist_flat.reshape(len(p_values), len(E_values))
    else:
        transport_dist = np.zeros((len(p_values), len(E_values)))

    # Total distribution
    total_dist = sd_dist + loss_dist + transport_dist

    return total_dist, sd_dist, loss_dist, transport_dist


if __name__ == "__main__":
    # Example usage
    print("Combined Basis Function Module")
    print("=" * 50)

    # Define grids
    E_values = np.linspace(1e3, 150e3, 40)
    p_values = np.linspace(-0.99, 0.99, 41)

    print(f"Energy grid: {len(E_values)} points from {E_values[0]/1e3:.1f} to {E_values[-1]/1e3:.1f} keV")
    print(f"Pitch grid: {len(p_values)} points from {p_values[0]:.2f} to {p_values[-1]:.2f}")
    print(f"Total grid points: {len(E_values) * len(p_values)}")
    print()

    # Create example loss basis
    loss_locations = [(96e3, -0.69), (95e3, -0.7)]
    LossPhi = create_loss_basis(E_values, p_values, loss_locations)
    print(f"Loss basis shape: {LossPhi.shape}")
    print(f"Loss locations: {loss_locations}")
    print()

    # Create example transport basis
    transport_locations = [
        ((48e3, -0.7), (65e3, -0.4)),
        ((47e3, -0.7), (64e3, -0.4))
    ]
    TransportPhi = create_transport_basis(E_values, p_values, transport_locations)
    print(f"Transport basis shape: {TransportPhi.shape}")
    print(f"Transport paths: {len(transport_locations)}")
    print()

    # Create a simple example slowing down basis (identity for demo)
    num_grid = len(E_values) * len(p_values)
    Phi = np.eye(num_grid)
    print(f"Slowing down basis shape: {Phi.shape}")
    print()

    # Combine all bases
    Phi_combined = combine_basis(Phi, LossPhi, TransportPhi)
    print(f"Combined basis shape: {Phi_combined.shape}")
    print(f"  = {Phi.shape[1]} (slowing down) + {LossPhi.shape[1]} (loss) + {TransportPhi.shape[1]} (transport)")
