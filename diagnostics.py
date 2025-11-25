"""
Fast Ion Diagnostics Module

This module provides functions to create weight functions (response matrices) for
various fast ion diagnostics and to generate synthetic signals from distribution functions.

Supported diagnostics:
1. FIDA (Fast-Ion D-Alpha): Measures Doppler-shifted D-alpha emission from neutralized fast ions
2. SSNPA (Solid-State Neutral Particle Analyzer): Measures energetic neutral particles

The weight functions (A matrices) relate the fast ion distribution to the diagnostic signal:
    signal = A @ distribution.flatten()

Usage:
    from diagnostics import create_fida_diagnostic, create_ssnpa_diagnostic, generate_signal

    # Create weight functions
    A_fida = create_fida_diagnostic(E_values, p_values, viewing_angles=[5, 25, 45])
    A_ssnpa = create_ssnpa_diagnostic(E_values, p_values, viewing_angles=[120, 115, 110])

    # Generate signals
    signal_fida = generate_signal(A_fida, distribution, noise_level=0.05)
    signal_ssnpa = generate_signal(A_ssnpa, distribution, noise_level=0.02)
"""

import numpy as np
from typing import List, Union, Optional, Tuple


def _generate_single_diagnostic_matrix(
    E_range: np.ndarray,
    p_range: np.ndarray,
    phi: float,
    u_range: np.ndarray,
    du: float,
    diagnostic_type: str = "FIDA",
    energy_threshold: float = 10e3,
    energy_max: float = 100e3,
    pitch_width: float = 0.025
) -> np.ndarray:
    """
    Generate the weight function matrix for a single viewing angle.

    This is the core function that computes the instrument response matrix (A matrix)
    for either FIDA or SSNPA diagnostics at a specific viewing angle.

    Parameters:
    -----------
    E_range : np.ndarray
        Energy grid in eV, shape (N_E,)
    p_range : np.ndarray
        Pitch grid (dimensionless, -1 to 1), shape (N_p,)
    phi : float
        Viewing angle in radians
    u_range : np.ndarray
        Velocity grid in m/s for the measurement, shape (N_u,)
    du : float
        Velocity bin width in m/s
    diagnostic_type : str
        "FIDA" or "SSNPA" (default: "FIDA")
    energy_threshold : float
        Minimum energy for SSNPA (eV), default: 10 keV
    energy_max : float
        Maximum energy for SSNPA (eV), default: 100 keV
    pitch_width : float
        Half-width of pitch acceptance for SSNPA (dimensionless), default: 0.025

    Returns:
    --------
    A : np.ndarray
        Weight function matrix of shape (len(u_range), len(E_range) * len(p_range))

    Notes:
    ------
    The weight function relates velocity components to energy-pitch space:
    - FIDA: Sensitive to all energies and pitches
    - SSNPA: Sensitive only to energies above threshold and narrow pitch range around viewing angle
    """
    # Particle mass (deuterium)
    m = 2 * 1.660539e-27  # kg

    # Create 2D grids
    Energy, Pitch = np.meshgrid(E_range, p_range)

    # Initialize weight function matrix
    A = np.zeros((len(u_range), len(E_range) * len(p_range)))

    for u_idx, u in enumerate(u_range):
        # Compute velocity-pitch transformation
        # These formulas relate the measured velocity component to (E, p) space
        arg1 = ((u - du/2) * np.sqrt(m / (2 * Energy * 1.60218e-19)) - np.cos(phi) * Pitch) / \
               (np.sin(phi) * np.sqrt(1 - Pitch ** 2))
        arg2 = ((u + du/2) * np.sqrt(m / (2 * Energy * 1.60218e-19)) - np.cos(phi) * Pitch) / \
               (np.sin(phi) * np.sqrt(1 - Pitch ** 2))

        # Compute weight from geometric acceptance
        gamma1 = np.arccos(np.clip(arg1, -1, 1))
        gamma2 = np.arccos(np.clip(arg2, -1, 1))
        w = (gamma1 - gamma2) / (np.pi * du)

        if diagnostic_type == "SSNPA":
            # SSNPA: Apply energy and pitch masks
            energy_mask = (Energy >= energy_threshold) & (Energy <= energy_max)
            pitch_center = np.cos(phi)  # Center of SSNPA pitch acceptance
            pitch_mask = np.abs(Pitch - pitch_center) < pitch_width

            # Reshape weight to match Energy/Pitch for proper broadcasting
            w_reshaped = w.reshape(Energy.shape)
            A[u_idx, :] = np.where(energy_mask & pitch_mask, w_reshaped, 0).flatten()

        else:  # FIDA default behavior
            A[u_idx, :] = w.flatten()

    return A


def create_fida_diagnostic(
    E_values: np.ndarray,
    p_values: np.ndarray,
    viewing_angles: Union[List[float], np.ndarray],
    u_range: Optional[np.ndarray] = None,
    degrees: bool = True
) -> np.ndarray:
    """
    Create FIDA (Fast-Ion D-Alpha) diagnostic weight function.

    FIDA measures Doppler-shifted D-alpha emission from charge-exchanged fast ions.
    Each viewing angle provides a spectrum across velocity space.

    Parameters:
    -----------
    E_values : np.ndarray
        Energy grid in eV, shape (N_E,)
    p_values : np.ndarray
        Pitch grid (dimensionless, -1 to 1), shape (N_p,)
    viewing_angles : list or array
        List of viewing angles. If degrees=True, in degrees; otherwise in radians
    u_range : np.ndarray, optional
        Velocity grid for measurements in m/s. If None, uses default range
    degrees : bool
        If True (default), viewing_angles are in degrees; otherwise in radians

    Returns:
    --------
    A_fida : np.ndarray
        FIDA weight function matrix of shape (N_angles * N_u, N_E * N_p)
        where N_angles is the number of viewing angles and N_u is the number of velocity bins

    Example:
    --------
    >>> E_values = np.linspace(1e3, 120e3, 40)
    >>> p_values = np.linspace(-0.99, 0.99, 41)
    >>> A_fida = create_fida_diagnostic(E_values, p_values, [5, 25, 45, 65, 85])
    >>> print(A_fida.shape)  # (505, 1640) - 5 angles × 101 velocity bins, 1640 grid points
    """
    # Convert to radians if needed
    if degrees:
        viewing_angles_rad = np.deg2rad(viewing_angles)
    else:
        viewing_angles_rad = np.array(viewing_angles)

    # Default velocity range if not provided
    if u_range is None:
        u_range = np.linspace(-3.7e6, 3.7e6, 101)

    du = np.diff(u_range)[0]

    # Generate weight function for each viewing angle
    A_total = None
    for phi in viewing_angles_rad:
        A_phi = _generate_single_diagnostic_matrix(
            E_values, p_values, phi, u_range, du, diagnostic_type="FIDA"
        )
        A_total = A_phi if A_total is None else np.vstack([A_total, A_phi])

    return A_total


def create_ssnpa_diagnostic(
    E_values: np.ndarray,
    p_values: np.ndarray,
    viewing_angles: Union[List[float], np.ndarray],
    u_range: Optional[np.ndarray] = None,
    energy_threshold: float = 10e3,
    energy_max: float = 100e3,
    pitch_width: float = 0.025,
    degrees: bool = True
) -> np.ndarray:
    """
    Create SSNPA (Solid-State Neutral Particle Analyzer) diagnostic weight function.

    SSNPA measures energetic neutral particles at specific viewing angles with
    narrow pitch acceptance and energy threshold.

    Parameters:
    -----------
    E_values : np.ndarray
        Energy grid in eV, shape (N_E,)
    p_values : np.ndarray
        Pitch grid (dimensionless, -1 to 1), shape (N_p,)
    viewing_angles : list or array
        List of viewing angles. If degrees=True, in degrees; otherwise in radians
    u_range : np.ndarray, optional
        Velocity grid for measurements in m/s. If None, uses default range
    energy_threshold : float
        Minimum energy threshold in eV (default: 10 keV)
    energy_max : float
        Maximum energy in eV (default: 100 keV)
    pitch_width : float
        Half-width of pitch acceptance (dimensionless), default: 0.025
    degrees : bool
        If True (default), viewing_angles are in degrees; otherwise in radians

    Returns:
    --------
    A_ssnpa : np.ndarray
        SSNPA weight function matrix of shape (N_angles * N_u, N_E * N_p)

    Example:
    --------
    >>> E_values = np.linspace(1e3, 120e3, 40)
    >>> p_values = np.linspace(-0.99, 0.99, 41)
    >>> A_ssnpa = create_ssnpa_diagnostic(E_values, p_values, [120, 115, 110, 105, 100])
    >>> print(A_ssnpa.shape)  # (505, 1640) - 5 angles × 101 velocity bins, 1640 grid points
    """
    # Convert to radians if needed
    if degrees:
        viewing_angles_rad = np.deg2rad(viewing_angles)
    else:
        viewing_angles_rad = np.array(viewing_angles)

    # Default velocity range if not provided
    if u_range is None:
        u_range = np.linspace(-3.7e6, 3.7e6, 101)

    du = np.diff(u_range)[0]

    # Generate weight function for each viewing angle
    A_total = None
    for phi in viewing_angles_rad:
        A_phi = _generate_single_diagnostic_matrix(
            E_values, p_values, phi, u_range, du,
            diagnostic_type="SSNPA",
            energy_threshold=energy_threshold,
            energy_max=energy_max,
            pitch_width=pitch_width
        )
        A_total = A_phi if A_total is None else np.vstack([A_total, A_phi])

    return A_total


def create_combined_diagnostic(
    E_values: np.ndarray,
    p_values: np.ndarray,
    fida_angles: Optional[Union[List[float], np.ndarray]] = None,
    ssnpa_angles: Optional[Union[List[float], np.ndarray]] = None,
    u_range: Optional[np.ndarray] = None,
    normalize: bool = False,
    degrees: bool = True,
    **ssnpa_params
) -> np.ndarray:
    """
    Create combined diagnostic weight function with both FIDA and SSNPA.

    This is a convenience function to create a single weight function matrix
    that combines multiple diagnostic types.

    Parameters:
    -----------
    E_values : np.ndarray
        Energy grid in eV, shape (N_E,)
    p_values : np.ndarray
        Pitch grid (dimensionless, -1 to 1), shape (N_p,)
    fida_angles : list or array, optional
        FIDA viewing angles (in degrees if degrees=True)
    ssnpa_angles : list or array, optional
        SSNPA viewing angles (in degrees if degrees=True)
    u_range : np.ndarray, optional
        Velocity grid for measurements in m/s
    normalize : bool
        If True, normalize the combined matrix (default: False)
    degrees : bool
        If True (default), angles are in degrees; otherwise in radians
    **ssnpa_params : dict
        Additional parameters for SSNPA (energy_threshold, energy_max, pitch_width)

    Returns:
    --------
    A_combined : np.ndarray
        Combined weight function matrix

    Example:
    --------
    >>> E_values = np.linspace(1e3, 120e3, 40)
    >>> p_values = np.linspace(-0.99, 0.99, 41)
    >>> A_combined = create_combined_diagnostic(
    ...     E_values, p_values,
    ...     fida_angles=[5, 25, 45, 65, 85],
    ...     ssnpa_angles=[120, 115, 110, 105, 100]
    ... )
    >>> print(A_combined.shape)  # (1010, 1640) - combined FIDA and SSNPA
    """
    diagnostic_matrices = []

    if fida_angles is not None:
        A_fida = create_fida_diagnostic(E_values, p_values, fida_angles, u_range, degrees)
        diagnostic_matrices.append(A_fida)

    if ssnpa_angles is not None:
        A_ssnpa = create_ssnpa_diagnostic(
            E_values, p_values, ssnpa_angles, u_range, degrees=degrees, **ssnpa_params
        )
        diagnostic_matrices.append(A_ssnpa)

    if len(diagnostic_matrices) == 0:
        raise ValueError("At least one of fida_angles or ssnpa_angles must be provided")

    # Combine matrices
    A_combined = np.vstack(diagnostic_matrices)

    # Normalize if requested
    if normalize:
        A_combined /= np.linalg.norm(A_combined)

    return A_combined


def generate_signal(
    weight_function: np.ndarray,
    distribution: np.ndarray,
    noise_level: float = 0.0,
    noise_type: str = 'relative',
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate diagnostic signal from weight function and distribution.

    The signal is computed as: y = A @ f.flatten()
    where A is the weight function and f is the distribution.

    Parameters:
    -----------
    weight_function : np.ndarray
        Diagnostic weight function (A matrix), shape (N_measurements, N_grid_points)
    distribution : np.ndarray
        Fast ion distribution function. Can be 1D (flattened) or 2D (pitch, energy)
    noise_level : float
        Standard deviation of noise to add (default: 0.0 = no noise)
    noise_type : str
        Type of noise to add:
        - 'relative': noise_level * signal * randn (default, signal-dependent)
        - 'absolute': noise_level * randn (signal-independent)
        - 'poisson': Poisson noise based on signal (photon counting)
    random_seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    signal : np.ndarray
        Diagnostic signal, shape (N_measurements,)

    Example:
    --------
    >>> # Create a simple distribution
    >>> distribution = np.random.rand(41, 40)  # pitch × energy
    >>>
    >>> # Generate clean signal
    >>> signal_clean = generate_signal(A_fida, distribution)
    >>>
    >>> # Generate noisy signal with 5% relative noise
    >>> signal_noisy = generate_signal(A_fida, distribution, noise_level=0.05)
    >>>
    >>> # Generate signal with absolute noise
    >>> signal_abs_noise = generate_signal(A_fida, distribution,
    ...                                     noise_level=0.1, noise_type='absolute')
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Flatten distribution if needed
    if distribution.ndim > 1:
        dist_flat = distribution.flatten()
    else:
        dist_flat = distribution

    # Check dimensions
    if len(dist_flat) != weight_function.shape[1]:
        raise ValueError(
            f"Distribution size {len(dist_flat)} does not match "
            f"weight function columns {weight_function.shape[1]}"
        )

    # Compute clean signal
    signal = weight_function @ dist_flat

    # Add noise if requested
    if noise_level > 0:
        if noise_type == 'relative':
            # Relative noise: proportional to signal amplitude
            noise = noise_level * np.random.randn(*signal.shape) * np.abs(signal)
            signal = signal + noise
        elif noise_type == 'absolute':
            # Absolute noise: independent of signal
            noise = noise_level * np.random.randn(*signal.shape)
            signal = signal + noise
        elif noise_type == 'poisson':
            # Poisson noise: for photon counting
            # Scale signal to counts, apply Poisson, scale back
            scale = 1.0 / noise_level if noise_level > 0 else 1.0
            signal_counts = np.maximum(signal * scale, 0)  # Ensure non-negative
            signal_counts = np.random.poisson(signal_counts)
            signal = signal_counts / scale
        else:
            raise ValueError(
                f"Unknown noise_type '{noise_type}'. "
                f"Use 'relative', 'absolute', or 'poisson'"
            )

    # Ensure non-negative signal (physical constraint)
    signal = np.abs(signal)

    return signal


def add_noise(
    signal: np.ndarray,
    noise_level: float,
    noise_type: str = 'relative',
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Add noise to an existing signal.

    This is a convenience function to add noise to pre-computed signals.

    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    noise_level : float
        Standard deviation of noise
    noise_type : str
        Type of noise: 'relative', 'absolute', or 'poisson'
    random_seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    noisy_signal : np.ndarray
        Signal with added noise

    Example:
    --------
    >>> signal_clean = generate_signal(A_fida, distribution, noise_level=0.0)
    >>> signal_noisy = add_noise(signal_clean, noise_level=0.05)
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    signal_noisy = signal.copy()

    if noise_type == 'relative':
        noise = noise_level * np.random.randn(*signal.shape) * np.abs(signal)
        signal_noisy = signal + noise
    elif noise_type == 'absolute':
        noise = noise_level * np.random.randn(*signal.shape)
        signal_noisy = signal + noise
    elif noise_type == 'poisson':
        scale = 1.0 / noise_level if noise_level > 0 else 1.0
        signal_counts = np.maximum(signal * scale, 0)
        signal_counts = np.random.poisson(signal_counts)
        signal_noisy = signal_counts / scale
    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. Use 'relative', 'absolute', or 'poisson'"
        )

    # Ensure non-negative
    signal_noisy = np.abs(signal_noisy)

    return signal_noisy


def visualize_weight_function(
    weight_function: np.ndarray,
    E_values: np.ndarray,
    p_values: np.ndarray,
    measurement_index: int = 0,
    ax=None
):
    """
    Visualize a single row of the weight function in (E, p) space.

    This helper function plots the sensitivity of a single measurement channel
    to different regions of energy-pitch space.

    Parameters:
    -----------
    weight_function : np.ndarray
        Diagnostic weight function, shape (N_measurements, N_grid_points)
    E_values : np.ndarray
        Energy grid in eV
    p_values : np.ndarray
        Pitch grid
    measurement_index : int
        Which measurement channel to plot (default: 0)
    ax : matplotlib axes, optional
        Axes to plot on. If None, creates new figure

    Returns:
    --------
    ax : matplotlib axes
        The axes object with the plot

    Example:
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> visualize_weight_function(A_fida, E_values, p_values, measurement_index=50, ax=ax)
    >>> plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization")

    # Reshape weight function for this measurement
    weight_2d = weight_function[measurement_index, :].reshape(len(p_values), len(E_values))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(
        weight_2d,
        extent=[E_values[0]/1e3, E_values[-1]/1e3, p_values[0], p_values[-1]],
        aspect='auto',
        cmap='hot_r',
        origin='lower',
        interpolation='bicubic'
    )

    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Pitch')
    ax.set_title(f'Weight Function (Measurement {measurement_index})')

    plt.colorbar(im, ax=ax, label='Sensitivity')

    return ax


if __name__ == "__main__":
    # Example usage and demonstration
    print("Fast Ion Diagnostics Module")
    print("=" * 60)

    # Define grids
    E_values = np.linspace(1e3, 120e3, 40)
    p_values = np.linspace(-0.99, 0.99, 41)

    print(f"Energy grid: {len(E_values)} points from {E_values[0]/1e3:.1f} to {E_values[-1]/1e3:.1f} keV")
    print(f"Pitch grid: {len(p_values)} points from {p_values[0]:.2f} to {p_values[-1]:.2f}")
    print(f"Total grid points: {len(E_values) * len(p_values)}")
    print()

    # Create FIDA diagnostic
    print("Creating FIDA diagnostic...")
    fida_angles = [5, 25, 45, 65, 85]
    A_fida = create_fida_diagnostic(E_values, p_values, fida_angles)
    print(f"FIDA weight function shape: {A_fida.shape}")
    print(f"  - {len(fida_angles)} viewing angles")
    print(f"  - {A_fida.shape[0] // len(fida_angles)} velocity bins per angle")
    print()

    # Create SSNPA diagnostic
    print("Creating SSNPA diagnostic...")
    ssnpa_angles = [120, 115, 110, 105, 100]
    A_ssnpa = create_ssnpa_diagnostic(E_values, p_values, ssnpa_angles)
    print(f"SSNPA weight function shape: {A_ssnpa.shape}")
    print(f"  - {len(ssnpa_angles)} viewing angles")
    print(f"  - {A_ssnpa.shape[0] // len(ssnpa_angles)} velocity bins per angle")
    print()

    # Create combined diagnostic
    print("Creating combined diagnostic...")
    A_combined = create_combined_diagnostic(
        E_values, p_values,
        fida_angles=fida_angles,
        ssnpa_angles=ssnpa_angles,
        normalize=True
    )
    print(f"Combined weight function shape: {A_combined.shape}")
    print()

    # Create example distribution
    print("Generating example distribution...")
    from slowing_down_basis import generate_simple_basis

    # Generate basis
    Phi = generate_simple_basis(E_values, p_values, S0=1e20, tau_s=1e-3, n_jobs=1)

    # Linear combination of basis functions
    initial_conditions = [(100e3, -0.7), (50e3, -0.7), (33e3, -0.7)]
    coefficients = [0.5, 1.0, 1.0]

    distribution = np.zeros_like(Phi[:, 0])
    for coeff, (E0, p0) in zip(coefficients, initial_conditions):
        i = np.argmin(np.abs(E_values - E0))
        j = np.argmin(np.abs(p_values - p0))
        column_index = i * len(p_values) + j
        distribution += coeff * Phi[:, column_index]

    distribution = distribution.reshape(len(p_values), len(E_values))
    print(f"Distribution shape: {distribution.shape}")
    print()

    # Generate signals
    print("Generating signals...")
    signal_fida_clean = generate_signal(A_fida, distribution, noise_level=0.0)
    signal_fida_noisy = generate_signal(A_fida, distribution, noise_level=0.05, random_seed=42)

    signal_ssnpa_clean = generate_signal(A_ssnpa, distribution, noise_level=0.0)
    signal_ssnpa_noisy = generate_signal(A_ssnpa, distribution, noise_level=0.02, random_seed=42)

    print(f"FIDA signal shape: {signal_fida_clean.shape}")
    print(f"  - Clean signal range: [{signal_fida_clean.min():.2e}, {signal_fida_clean.max():.2e}]")
    print(f"  - Noisy signal range: [{signal_fida_noisy.min():.2e}, {signal_fida_noisy.max():.2e}]")
    print()

    print(f"SSNPA signal shape: {signal_ssnpa_clean.shape}")
    print(f"  - Clean signal range: [{signal_ssnpa_clean.min():.2e}, {signal_ssnpa_clean.max():.2e}]")
    print(f"  - Noisy signal range: [{signal_ssnpa_noisy.min():.2e}, {signal_ssnpa_noisy.max():.2e}]")
    print()

    print("Example complete!")
    print()
    print("To visualize weight functions, use:")
    print("  from diagnostics import visualize_weight_function")
    print("  import matplotlib.pyplot as plt")
    print("  visualize_weight_function(A_fida, E_values, p_values, measurement_index=50)")
    print("  plt.show()")
