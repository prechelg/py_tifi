#!/usr/bin/env python
"""
Slowing-down basis function generator for fast ion distribution analysis.

This module provides two methods for generating slowing-down basis functions:
1. Simple method: Fast analytical approximation with Gaussian pitch scattering
2. Coulomb method: Rigorous Fokker-Planck solution with Legendre expansion

Both methods produce compatible output matrices suitable for distribution function
reconstruction and tomographic inversion.
"""

import numpy as np
from scipy.optimize import root_scalar

# Optional parallelization support
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    def delayed(func):
        return func

# ============================================================================
# Physical Constants
# ============================================================================

m_e = 9.1093837015e-31  # Electron mass (kg)
m_p = 1.67262192369e-27  # Proton mass (kg)
e0 = 1.60217663e-19     # Elementary charge (C or J/eV)
c0 = 2.99792458e8       # Speed of light (m/s)
mass_u = 1.66053904e-27 # Atomic mass unit (kg)
e_amu = 5.48579909070e-4  # Electron mass in amu


# ============================================================================
# Coulomb Logarithm Functions
# ============================================================================

def coulomb_logarithm_ei(ne, Te, ni, Ti, mu=2, Z=1):
    """Calculate the Coulomb logarithm for electron-ion collisions."""
    m_i = mu * m_p
    mr = m_e / m_i

    if Ti * mr < Te and Te < 10 * Z**2:
        lnLambda = 23.0 - np.log(np.sqrt(ne) * Z * Te**(-3/2))
    elif Ti * mr < 10 * Z**2 and 10 * Z**2 < Te:
        lnLambda = 24.0 - np.log(np.sqrt(ne) / Te)
    elif Te < Ti * mr:
        lnLambda = 16.0 - np.log(mu * np.sqrt(ni) * (Ti**(-3/2)) * Z**2)
    else:
        lnLambda = 16.0

    return lnLambda


def coulomb_logarithm_ii(ni1, Ti1, ni2, Ti2, mu1=2, mu2=2, Z1=1, Z2=1):
    """Calculate the Coulomb logarithm for ion-ion collisions."""
    lnLambda = 23 - np.log(((Z1 * Z2 * (mu1 + mu2)) / (mu1 * Ti2 + mu2 * Ti1)) *
                           np.sqrt((ni1 * Z1**2) / Ti1 + (ni2 * Z2**2) / Ti2))
    return lnLambda


def coulomb_logarithm_ii_counterstreaming(ni1, Ti1, ni2, Ti2, ne, Te, beta_D,
                                          mu1=2, mu2=2, Z1=1, Z2=1):
    """Calculate the Coulomb logarithm for counter-streaming ions."""
    m_i1 = mass_u * mu1
    m_i2 = mass_u * mu2
    m_e_local = mass_u * e_amu

    L1 = Ti1 * e0 / m_i1
    L2 = Ti2 * e0 / m_i2
    U = Te * e0 / m_e_local

    if max(L1, L2) < (beta_D * c0)**2 and (beta_D * c0)**2 < U:
        lnLambda = 43 - np.log((Z1 * Z2 * (mu1 + mu2) / (mu1 * mu2 * beta_D**2)) *
                               np.sqrt(ne / Te))
    else:
        lnLambda = coulomb_logarithm_ii(ni1, Ti1, ni2, Ti2, mu1, mu2, Z1, Z2)

    return lnLambda


# ============================================================================
# Critical Energy and Slowing Time Functions
# ============================================================================

def electron_ion_drag_difference(Eb, ne, Te, Ti, Zeff, Ab=2, Ai=2, Aimp=12,
                                 Zb=1, Zi=1, Zimp=6):
    """Calculate the difference between electron drag and ion drag."""
    m_b = Ab * m_p
    m_imp = Aimp * m_p
    m_i = Ai * m_p

    v_b = np.sqrt(2 * Eb * e0 / m_b)
    v_e = np.sqrt(2 * Te * e0 / m_e)

    if Zeff > 1:
        nimp = ne * (Zeff - 1) / (Zimp * (Zimp - 1))
    else:
        nimp = 0
    ni = max(ne - Zimp * nimp, 0)

    # Electron drag
    lnLambda_be = coulomb_logarithm_ei(ne, Te, ni, Ti, Ab, Zb)
    Gamma_be = (2 * np.pi * ne * e0**4 * Zb**2 * lnLambda_be) / (m_b**2)
    electron_drag = ((8 * Gamma_be * m_b) / (3 * np.sqrt(np.pi) * m_e * v_e**3)) * v_b**3

    # Ion drag
    lnLambda_bi = coulomb_logarithm_ii_counterstreaming(ni, Ti, ni, Ti, ne, Te,
                                                        v_b/c0, Ab, Ai, Zb, Zi)
    Gamma_bi = (2 * np.pi * ni * e0**4 * Zi**2 * Zb**2 * lnLambda_bi) / (m_b**2)

    lnLambda_bimp = coulomb_logarithm_ii_counterstreaming(ni, Ti, nimp, Ti, ne, Te,
                                                          v_b/c0, Ab, Aimp, Zb, Zimp)
    Gamma_bimp = (2 * np.pi * nimp * e0**4 * Zimp**2 * Zb**2 * lnLambda_bimp) / (m_b**2)

    ion_drag = 2 * m_b * (Gamma_bi / m_i + Gamma_bimp / m_imp)

    return electron_drag - ion_drag


def critical_energy(ne, Te, Ti, Zeff, Ai=2, Ab=2, Zb=1, Zi=1, Zimp=6, Emax=300e3):
    """Calculate the critical energy where ion and electron drag balance."""
    drag_diff_fun = lambda Eb: electron_ion_drag_difference(
        Eb, ne, Te, Ti, Zeff, Ab, Ai, 12, Zb, Zi, Zimp)

    try:
        sol = root_scalar(drag_diff_fun, bracket=[0, Emax], method='bisect', xtol=1e-6)
        if not sol.converged:
            raise ValueError("Failed to find critical energy")
        Ec = sol.root
    except Exception:
        raise ValueError("Failed to find critical energy")
    return Ec


def slowing_down_time(ne, Te, Ti, Zeff, Ai=2, Ab=2, Zb=1, Zimp=6):
    """Calculate the slowing-down time on electrons."""
    if Zeff > 1:
        nimp = ne * (Zeff - 1) / (Zimp * (Zimp - 1))
    else:
        nimp = 0
    ni = max(ne - Zimp * nimp, 0)

    lnLambda = coulomb_logarithm_ei(ne, Te, ni, Ti, Ab, Zb)
    tau_s = (6.27e8 * Ab * (Te ** 1.5)) / ((ne * 1e-6) * lnLambda * (Zb ** 2))
    return tau_s


# ============================================================================
# Legendre Polynomial Functions
# ============================================================================

def custom_legendre(x, nterms):
    """Calculate Legendre polynomials up to nterms using recurrence relation."""
    p = np.zeros(nterms)
    p[0] = 1.0
    if nterms > 1:
        p[1] = x

    for i in range(1, nterms-1):
        p[i+1] = ((2*(i+1)-1) * x * p[i] - (i) * p[i-1]) / (i+1)
    return p


# ============================================================================
# Coulomb Method: Rigorous Fokker-Planck Solution
# ============================================================================

def thermalization_time(v_b, v_c, tau_s):
    """Calculate the thermalization time."""
    t_th = tau_s * np.log((v_b**3 + v_c**3) / v_c**3) / 3
    return t_th


def heaviside(x, threshold=1e-10):
    """MATLAB-style Heaviside function: returns 0.5 when x == 0."""
    if np.abs(x) < threshold:
        x = 0
    if x < 0:
        return 0.0
    elif x > 0:
        return 1.0
    return 0.5


def slowing_down_legendre_expansion_full(u, P, P0, Te, m_b, v_b, v_c, Zeff, Z3, tau_s, tau_cx):
    """Perform Legendre expansion for full slowing-down distribution."""
    v_e = np.sqrt(2 * Te * e0 / m_e)
    A = 0.5 * (m_e * v_e**2 / (m_b * v_b**2) + (v_c**3 * Z3) / v_b**3)
    B = (v_b**3 + v_c**3) / v_b**3

    lex = 0.0
    for l in range(len(P)):
        Cl = 3 * (l * (l + 1) * v_c**3 * Zeff / (6 * v_b**3) + tau_s / (3 * tau_cx) - 1)
        sqrt_term = np.sqrt(1 + 4 * A * Cl / B**2)
        energy_diff_factor = 1 / (1 + sqrt_term + 2 * A * Cl / B**2)
        lex += 0.5 * (2 * l + 1) * P[l] * P0[l] * (u ** (l * (l + 1))) * energy_diff_factor

    return lex


def slowing_down_velocity_full(v, v_b, v_c, tau_s, Zeff, P, P0, Te, m_b, Z3, tau_cx,
                                tau_on=0.0, tau_off=1.0, tau=1.0):
    """Calculate slowing-down distribution for a given velocity and pitch."""
    v3 = v**3
    vb3 = v_b**3
    vc3 = v_c**3

    inv_v3vc3 = 1.0 / (v3 + vc3)
    u1 = v3 / vb3
    u2 = (vb3 + vc3) * inv_v3vc3
    u = (u1 * u2)**(Zeff / 6)

    t_b = thermalization_time(v_b, v_c, tau_s)
    t_0 = tau_on * t_b
    t_th = tau_s * np.log(u2) / 3
    t_1 = tau_off * t_b - t_0
    t = tau * t_b - t_0

    S = min(t, t_b, t_1) / t_b
    U = heaviside(t - t_th) - heaviside(t - t_1 - t_th)

    if v > v_b:
        return 0.0
    if U == 0:
        return 0.0

    lex = slowing_down_legendre_expansion_full(u, P, P0, Te, m_b, v_b, v_c, Zeff, Z3, tau_s, tau_cx)
    g = max(S * tau_s * inv_v3vc3 * lex, 0.0) / t_b
    return g


def slowing_down_full(energy, pitch, E0, p0, nterms, ne, Zeff, Te, Ti, Ai, Ab, Zb, Zi, Zimp, Z3, n_n):
    """Calculate slowing-down distribution across energy-pitch grid."""
    m_b = Ab * m_p
    v_b = np.sqrt(2 * E0 * e0 / m_b)
    tau_s = slowing_down_time(ne, Te, Ti, Zeff, Ai, Ab, Zb, Zimp)
    Ec = critical_energy(ne, Te, Ti, Zeff, Ai, Ab, Zb, Zi, Zimp)
    v_c = np.sqrt(2 * Ec * e0 / m_b)

    nenergy = len(energy)
    npitch = len(pitch)
    P0 = custom_legendre(p0, nterms)
    f_slow = np.zeros((npitch, nenergy))

    for j in range(npitch):
        P = custom_legendre(pitch[j], nterms)
        for i in range(nenergy):
            v = np.sqrt(2 * energy[i] * e0 / m_b)
            energy_keV = energy[i] / 1e3
            sigma_cx = 21e-20 / ((0.2 * energy_keV) + 1)
            tau_cx = 1 / (n_n * sigma_cx * v)
            f_slow[j, i] = (v / m_b) * slowing_down_velocity_full(
                v, v_b, v_c, tau_s, Zeff, P, P0, Te, m_b, Z3, tau_cx, 0, 1, 1)
    return f_slow


def slowing_down_with_cx_full(energy, pitch, E0, p0, nterms, ne, Zeff, Te, Ti,
                               Ai, Ab, Zb, Zi, Zimp, n_n, ni, tau_s):
    """Compute slowing-down distribution with charge exchange corrections."""
    m_b = Ab * m_p
    v_b = np.sqrt(2 * E0 * e0 / m_b)
    Ec = critical_energy(ne, Te, Ti, Zeff, Ai, Ab, Zb, Zi, Zimp)
    v_c = np.sqrt(2 * Ec * e0 / m_b)

    if Zeff > 1:
        nimp = ne * (Zeff - 1) / (Zimp * (Zimp - 1))
    else:
        nimp = 0
    ni = max(ne - Zimp * nimp, 0)

    m_i = Ai * m_p
    Z1 = np.sum(ni * (Zi**2) * m_b / (ne * m_i))
    vi2 = (2 * Ti * e0) / (m_i * Zi)
    Z3 = np.sum(ni * (Zi**2) * vi2) / (ne * Z1 * (v_b**2))

    f_slow = slowing_down_full(energy, pitch, E0, p0, nterms, ne, Zeff, Te, Ti,
                                Ai, Ab, Zb, Zi, Zimp, Z3, n_n)

    npitch, nenergy = f_slow.shape
    f_slow_cx = np.zeros((npitch, nenergy))

    for i in range(nenergy):
        for j in range(npitch):
            v = np.sqrt(2 * energy[i] * e0 / m_b)
            energy_keV = energy[i] / 1e3
            sigma_cx = 21e-20 / ((0.2 * energy_keV) + 1)
            tau_cx = 1 / (n_n * sigma_cx * v)

            prefactor = (((v**3 + v_c**3) / (v_b**3 + v_c**3))**(1/2)) * \
                       (((v_b + v_c) / (v + v_c))**(3/2))
            prefactor = prefactor**(tau_s / (3 * tau_cx))

            term1 = np.sqrt(3) * np.arctan((2 * v_b - v_c) / (np.sqrt(3) * v_c))
            term2 = np.sqrt(3) * np.arctan((2 * v - v_c) / (np.sqrt(3) * v_c))
            exponential = np.exp(-tau_s / (3 * tau_cx) * (term1 - term2))

            f_slow_cx[j, i] = f_slow[j, i] * prefactor * exponential

    return f_slow_cx


def slowing_down_legendre_energy_diffusion(P, P0, Te, m_b, v_b, v_c, Zeff, Z3, tau_s, tau_cx, v):
    """Perform Legendre summation with energy diffusion factor."""
    v_e = np.sqrt(2 * Te * e0 / m_e)
    A = 0.5 * (m_e * (v_e**2) / (m_b * (v_b**2)) + (v_c**3 * Z3) / (v_b**3))
    B = (v_b**3 + v_c**3) / (v_b**3)

    lex = 0
    for l in range(len(P)):
        Cl = 3 * (l * (l + 1) * (v_c**3) * Zeff / (6 * (v_b**3)) + tau_s / (3 * tau_cx) - 1)

        sqrt_arg = 1 + 4 * A * Cl / (B**2)
        if sqrt_arg < 0:
            continue

        energy_diff_factor_1 = 1 / (1 + np.sqrt(sqrt_arg) + 2 * A * Cl / (B**2))
        exp_arg = -1 / (2 * A) * (((v - v_b) * B / v_b) + ((v - v_b) * np.sqrt(B**2 + 4 * A * Cl) / v_b))
        energy_diff_factor_2 = np.exp(np.clip(exp_arg, -100, 100))

        lex += 0.5 * (2 * l + 1) * P[l] * P0[l] * energy_diff_factor_1 * energy_diff_factor_2

    return lex


def slowing_down_velocity_energy_diffusion(v, v_b, v_c, tau_s, Zeff, P, P0, Te, m_b, Z3, tau_cx,
                                           tau_on=0.0, tau_off=1.0, tau=1.0):
    """Calculate energy diffusion component of distribution."""
    vb3 = v_b**3
    vc3 = v_c**3
    inv_v3vc3 = 1.0 / (vb3 + vc3)

    t_b = thermalization_time(v_b, v_c, tau_s)
    t_0 = tau_on * t_b
    t_1 = tau_off * t_b - t_0
    t_val = tau * t_b - t_0

    S = min(t_val, t_b, t_1) / t_b

    lex = slowing_down_legendre_energy_diffusion(P, P0, Te, m_b, v_b, v_c, Zeff, Z3, tau_s, tau_cx, v)
    g = max(S * tau_s * inv_v3vc3 * lex, 0.0) / t_b
    return g


def slowing_down_energy_diffusion(energy, pitch, E0, p0, nterms, ne, Zeff, Te, Ti,
                                   Ai, Ab, Zb, Zi, Zimp, ni, Z1, n_n):
    """Calculate energy diffusion distribution across energy-pitch grid."""
    m_b = Ab * m_p
    v_b = np.sqrt(2 * E0 * e0 / m_b)
    tau_s = slowing_down_time(ne, Te, Ti, Zeff, Ai, Ab, Zb, Zimp)
    Ec = critical_energy(ne, Te, Ti, Zeff, Ai, Ab, Zb, Zi, Zimp)
    v_c = np.sqrt(2 * Ec * e0 / m_b)

    m_i = Ai * m_p
    vi2 = 2 * Ti * e0 / (m_i * Zi)
    Z3 = np.sum(ni * (Zi ** 2) * vi2) / (ne * Z1 * (v_b ** 2))

    nenergy = len(energy)
    npitch = len(pitch)
    P0 = custom_legendre(p0, nterms)

    idx_E0 = int(np.argmin(np.abs(np.array(energy) - E0)))
    f_slow = np.zeros((npitch, nenergy))

    for j in range(npitch):
        P = custom_legendre(pitch[j], nterms)
        for i in range(idx_E0, nenergy):
            v = np.sqrt(2 * energy[i] * e0 / m_b)
            energy_keV = energy[i] / 1e3
            sigma_cx = 21e-20 / ((0.2 * energy_keV) + 1)
            tau_cx = 1 / (n_n * sigma_cx * v)

            f_slow[j, i] = (v / m_b) * slowing_down_velocity_energy_diffusion(
                v, v_b, v_c, tau_s, Zeff, P, P0, Te, m_b, Z3, tau_cx, 0, 1, 1)

    return f_slow


# ============================================================================
# Simple Method: Analytical Approximation
# ============================================================================

def calculate_alpha(v, v0, p, p0, beta, vc):
    """Calculate pitch diffusion parameter for simple method."""
    return np.maximum((beta * (1 - p**2) / 3) *
                     np.log((1 + (vc/v)**3) / (1 + (vc/v0)**3)), 1e-30)


def slowing_down_distribution_simple(v, v0, p, p0, S0, tau_s, beta, vc):
    """Calculate simple slowing-down distribution using Gaussian approximation."""
    alpha = calculate_alpha(v, v0, p, p0, beta, vc)
    term1 = S0 * tau_s / (2 * np.sqrt(np.pi * alpha * (v**3 + vc**3)))
    term2 = np.exp(np.clip(-((p + 1e-4) - p0)**2 / (4 * alpha), -100, 100))
    return term1 * term2


# ============================================================================
# Basis Matrix Generation Functions
# ============================================================================

def compute_coulomb_basis_column(i, j, energy, pitch, nterms, ne, Zeff, Te, Ti,
                                 Ai, Ab, Zb, Zi, Zimp, n_n, ni, tau_s, method='combined'):
    """
    Compute a single column of the Coulomb basis matrix.

    Parameters:
    -----------
    method : str
        'slowing_down': Use only slowing-down component
        'energy_diffusion': Use only energy diffusion component
        'combined': Use slowing-down for low E, energy diffusion for high E (default)
    """
    E0 = energy[i]
    p0 = pitch[j]

    # Calculate derived parameters
    m_b = Ab * m_p
    m_i = Ai * m_p
    Z1 = ni * (Zi**2) * m_b / (ne * m_i) if ne > 0 and m_i > 0 else 0

    # Compute slowing-down component
    f_slow = slowing_down_with_cx_full(energy, pitch, E0, p0, nterms, ne, Zeff, Te, Ti,
                                       Ai, Ab, Zb, Zi, Zimp, n_n, ni, tau_s)

    if method == 'slowing_down':
        f_full = f_slow
    elif method == 'energy_diffusion':
        f_full = slowing_down_energy_diffusion(energy, pitch, E0, p0, nterms, ne, Zeff,
                                               Te, Ti, Ai, Ab, Zb, Zi, Zimp, ni, Z1, n_n)
    else:  # combined
        # Normalize slowing-down
        max_f_slow = np.max(f_slow)
        f_slow_norm = f_slow / np.clip(max_f_slow, 1e-30, None)

        # Compute energy diffusion
        f_diffusion = slowing_down_energy_diffusion(energy, pitch, E0, p0, nterms, ne, Zeff,
                                                     Te, Ti, Ai, Ab, Zb, Zi, Zimp, ni, Z1, n_n)
        max_f_diffusion = np.max(f_diffusion)
        f_diffusion_norm = f_diffusion / np.clip(max_f_diffusion, 1e-30, None)

        # Combine: use diffusion where slowing-down is weak and E >= E0
        J = np.arange(0, len(energy))[None, :]
        condition = (f_slow_norm < 0.2) & (J >= i)
        f_full = np.where(condition, f_diffusion_norm, f_slow_norm)

    column_index = i * len(pitch) + j
    return column_index, f_full.flatten()


def compute_simple_basis_column(i, j, E_values, p_values, v_values, m_f, S0, tau_s, beta, vc):
    """Compute a single column of the simple basis matrix.

    EXACTLY replicating old notebook code from ExtendedBasisTransport.ipynb
    """
    E0 = E_values[i]
    p0 = p_values[j]
    v0 = np.sqrt(2 * E0 * 1.60218e-19 / m_f)

    # Calculate the distribution for this (E0, p0) starting point
    # OLD NOTEBOOK APPROACH: iterate in velocity space
    f_values = np.zeros((len(p_values), len(v_values)))
    for k, p in enumerate(p_values):
        for l, v in enumerate(v_values):
            f_values[k, l] = slowing_down_distribution_simple(v, v0, p, p0, S0, tau_s, beta, vc)

    # Transform to (E, p) coordinates
    f_values_Ep = f_values * (m_f * v_values)

    # Flatten the column
    column_index = i * len(p_values) + j
    return column_index, f_values_Ep.flatten()


def generate_basis_matrix(E_values, p_values, method='simple', n_jobs=1, **params):
    """
    Generate slowing-down basis matrix.

    Parameters:
    -----------
    E_values : array
        Energy grid in eV
    p_values : array
        Pitch grid (dimensionless, -1 to 1)
    method : str
        'simple': Fast analytical method (default)
        'coulomb': Rigorous Coulomb scattering method
        'coulomb_slowing': Coulomb method with only slowing-down
        'coulomb_diffusion': Coulomb method with only energy diffusion
        'coulomb_combined': Coulomb method with combined approach
    n_jobs : int
        Number of parallel jobs (default: 1, use -1 for all cores)
    **params : dict
        Parameters for the chosen method

    Simple method params:
        S0 : float (default: 1e20) - Source strength
        tau_s : float (default: 5e-2) - Slowing-down time (s)
        beta : float - Pitch diffusion coefficient
        vc : float - Critical velocity (m/s)
        m_f : float - Fast ion mass (kg)

    Coulomb method params:
        nterms : int (default: 100) - Number of Legendre terms
        ne : float - Electron density (m^-3)
        Zeff : float - Effective charge
        Te : float - Electron temperature (eV)
        Ti : float - Ion temperature (eV)
        n_n : float - Neutral density (m^-3)
        Ai, Ab, Zb, Zi, Zimp : int - Mass/charge numbers

    Returns:
    --------
    Phi : ndarray
        Basis matrix of shape (num_grid_points, num_grid_points)
    """
    num_grid_points = len(E_values) * len(p_values)

    if method == 'simple':
        # Set default parameters for simple method
        S0 = params.get('S0', 1e20)
        tau_s = params.get('tau_s', 5e-2)
        m_f = params.get('m_f', 2 * m_p)

        # Calculate derived parameters if not provided
        if 'beta' not in params or 'vc' not in params:
            ne = params.get('ne', 1e19)
            m_e_val = params.get('m_e', 9.10938356e-31)
            v_e = params.get('v_e', 4e7)
            m_i = m_f
            Z_i = params.get('Z_i', 1)
            n_i = ne

            Z1 = np.sum((n_i * m_f * Z_i**2) / (ne * m_i))
            Z2 = np.sum((n_i * Z_i**2) / (ne * Z1))
            beta = params.get('beta', Z2 / 2)
            vc = params.get('vc', ((3 * np.sqrt(np.pi) * m_e_val / (4 * m_f) * Z1)**(1/3)) * v_e)
        else:
            beta = params['beta']
            vc = params['vc']

        v_values = np.sqrt(2 * E_values * 1.60218e-19 / m_f)

        if HAS_JOBLIB and n_jobs != 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_simple_basis_column)(i, j, E_values, p_values, v_values,
                                                     m_f, S0, tau_s, beta, vc)
                for i in range(len(E_values))
                for j in range(len(p_values))
            )
        else:
            results = [compute_simple_basis_column(i, j, E_values, p_values, v_values,
                                                   m_f, S0, tau_s, beta, vc)
                      for i in range(len(E_values))
                      for j in range(len(p_values))]

    elif method.startswith('coulomb'):
        # Set default parameters for Coulomb method
        nterms = params.get('nterms', 100)
        ne = params['ne']
        Zeff = params['Zeff']
        Te = params['Te']
        Ti = params['Ti']
        n_n = params.get('n_n', 5e13)
        Ai = params.get('Ai', 2)
        Ab = params.get('Ab', 2)
        Zb = params.get('Zb', 1)
        Zi = params.get('Zi', 1)
        Zimp = params.get('Zimp', 6)

        # Calculate derived parameters
        tau_s = slowing_down_time(ne, Te, Ti, Zeff, Ai, Ab, Zb, Zimp)

        if Zeff > 1:
            nimp = ne * (Zeff - 1) / (Zimp * (Zimp - 1))
        else:
            nimp = 0
        ni = max(ne - Zimp * nimp, 0)

        # Determine sub-method
        if method == 'coulomb_slowing':
            sub_method = 'slowing_down'
        elif method == 'coulomb_diffusion':
            sub_method = 'energy_diffusion'
        else:  # coulomb or coulomb_combined
            sub_method = 'combined'

        if HAS_JOBLIB and n_jobs != 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_coulomb_basis_column)(
                    i, j, E_values, p_values, nterms, ne, Zeff, Te, Ti,
                    Ai, Ab, Zb, Zi, Zimp, n_n, ni, tau_s, sub_method)
                for i in range(len(E_values))
                for j in range(len(p_values))
            )
        else:
            results = [compute_coulomb_basis_column(
                i, j, E_values, p_values, nterms, ne, Zeff, Te, Ti,
                Ai, Ab, Zb, Zi, Zimp, n_n, ni, tau_s, sub_method)
                for i in range(len(E_values))
                for j in range(len(p_values))]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'simple', 'coulomb', 'coulomb_slowing', 'coulomb_diffusion', or 'coulomb_combined'")

    # Assemble matrix
    Phi = np.zeros((num_grid_points, num_grid_points))
    for column_index, column_data in results:
        Phi[:, column_index] = column_data

    # Normalize
    Phi /= np.linalg.norm(Phi)

    # Replace NaNs
    np.nan_to_num(Phi, copy=False)

    return Phi


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_simple_basis(E_values, p_values, S0=1e20, tau_s=5e-2, n_jobs=1, **plasma_params):
    """
    Generate basis matrix using the simple analytical method.

    This is a convenience wrapper around generate_basis_matrix.
    """
    return generate_basis_matrix(E_values, p_values, method='simple', n_jobs=n_jobs,
                                 S0=S0, tau_s=tau_s, **plasma_params)


def generate_coulomb_basis(E_values, p_values, ne, Te, Ti, Zeff, n_jobs=1, nterms=100,
                           method='combined', **other_params):
    """
    Generate basis matrix using the Coulomb scattering method.

    This is a convenience wrapper around generate_basis_matrix.

    Parameters:
    -----------
    method : str
        'combined': Use slowing-down and energy diffusion (default)
        'slowing_down': Use only slowing-down component
        'energy_diffusion': Use only energy diffusion component
    """
    coulomb_method = f'coulomb_{method}' if method != 'combined' else 'coulomb'
    return generate_basis_matrix(E_values, p_values, method=coulomb_method, n_jobs=n_jobs,
                                 ne=ne, Te=Te, Ti=Ti, Zeff=Zeff, nterms=nterms, **other_params)
