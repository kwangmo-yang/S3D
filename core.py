from __future__ import annotations
import numpy as np


def _fresnel_up(epsilon: np.ndarray, h: np.ndarray):
    """
    Compact interface indexing:
    interface i  <-> between layer i and layer i+1
    shape: (n_layers - 1, n_u, n_w)
    """
    n_layers = epsilon.shape[0]
    shp = (n_layers - 1,) + h.shape[1:]

    r_te = np.zeros(shp, dtype=complex)
    t_te = np.zeros(shp, dtype=complex)
    r_tm = np.zeros(shp, dtype=complex)
    t_tm = np.zeros(shp, dtype=complex)

    for i in range(n_layers - 1):
        hi = np.asarray(h[i])
        hip1 = np.asarray(h[i + 1])

        ei = np.asarray(epsilon[i]).reshape(1, -1)
        eip1 = np.asarray(epsilon[i + 1]).reshape(1, -1)

        r_te[i] = (hi - hip1) / (hi + hip1)
        t_te[i] = 1 + r_te[i]

        r_tm[i] = (hi / ei - hip1 / eip1) / (hi / ei + hip1 / eip1)
        # MATLAB upward:
        # t_TM = (1 + r_TM) .* sqrt(epsilon(n-1)/epsilon(n))
        t_tm[i] = (1 + r_tm[i]) * np.sqrt(ei / eip1)

    return r_te, t_te, r_tm, t_tm


def _fresnel_down(epsilon: np.ndarray, h: np.ndarray):
    """
    Compact interface indexing:
    interface i  <-> between layer i and layer i+1
    shape: (n_layers - 1, n_u, n_w)
    """
    n_layers = epsilon.shape[0]
    shp = (n_layers - 1,) + h.shape[1:]

    r_te = np.zeros(shp, dtype=complex)
    t_te = np.zeros(shp, dtype=complex)
    r_tm = np.zeros(shp, dtype=complex)
    t_tm = np.zeros(shp, dtype=complex)

    for i in range(n_layers - 1):
        hi = np.asarray(h[i])
        hip1 = np.asarray(h[i + 1])

        ei = np.asarray(epsilon[i]).reshape(1, -1)
        eip1 = np.asarray(epsilon[i + 1]).reshape(1, -1)

        r_te[i] = (hip1 - hi) / (hip1 + hi)
        t_te[i] = 1 + r_te[i]

        r_tm[i] = (hip1 / eip1 - hi / ei) / (hip1 / eip1 + hi / ei)
        # MATLAB downward:
        # t_TM = (1 + r_TM) .* sqrt(epsilon(n+1)/epsilon(n))
        t_tm[i] = (1 + r_tm[i]) * np.sqrt(eip1 / ei)

    return r_te, t_te, r_tm, t_tm


def purcell_bottom(
    epsilon: np.ndarray,
    thickness: np.ndarray,
    dipole_layer: int,
    dipole_position_array: np.ndarray,
    u: np.ndarray,
    h: np.ndarray,
):
    """
    dipole_layer: 0-based layer index
    epsilon: (n_layers, n_w)
    h:       (n_layers, n_u, n_w)
    u:       (n_u,)
    """
    n_layers = thickness.size

    # -------------------------
    # Upward stack
    # MATLAB:
    # r_TE_up(N-2) = r_TE(N-1)
    # compact Python mapping:
    # seed layer = n_layers - 3
    # seed interface = n_layers - 3
    # -------------------------
    r_te_if, t_te_if, r_tm_if, t_tm_if = _fresnel_up(epsilon, h)

    r_te_up = np.zeros_like(h, dtype=complex)
    r_tm_up = np.zeros_like(h, dtype=complex)
    t_te_up = np.zeros_like(h, dtype=complex)
    t_tm_up = np.zeros_like(h, dtype=complex)

    seed_layer = n_layers - 3
    if seed_layer >= 0:
        r_te_up[seed_layer] = r_te_if[seed_layer]
        r_tm_up[seed_layer] = r_tm_if[seed_layer]
        t_te_up[seed_layer] = t_te_if[seed_layer]
        t_tm_up[seed_layer] = t_tm_if[seed_layer]

    for layer in range(seed_layer - 1, dipole_layer - 1, -1):
        iface = layer
        phase = np.exp(2j * h[layer + 1] * thickness[layer + 1])
        phase_half = np.exp(1j * h[layer + 1] * thickness[layer + 1])

        r_te_up[layer] = (r_te_if[iface] + r_te_up[layer + 1] * phase) / (
            1 + r_te_if[iface] * r_te_up[layer + 1] * phase
        )
        r_tm_up[layer] = (r_tm_if[iface] + r_tm_up[layer + 1] * phase) / (
            1 + r_tm_if[iface] * r_tm_up[layer + 1] * phase
        )
        t_te_up[layer] = (t_te_if[iface] * t_te_up[layer + 1] * phase_half) / (
            1 + r_te_if[iface] * r_te_up[layer + 1] * phase
        )
        t_tm_up[layer] = (t_tm_if[iface] * t_tm_up[layer + 1] * phase_half) / (
            1 + r_tm_if[iface] * r_tm_up[layer + 1] * phase
        )

    # -------------------------
    # Downward stack
    # MATLAB:
    # r_TE_down(2) = r_TE(1)
    # compact Python mapping:
    # layer 1 <= interface 0
    # -------------------------
    r_te_if, t_te_if, r_tm_if, t_tm_if = _fresnel_down(epsilon, h)

    r_te_down = np.zeros_like(h, dtype=complex)
    r_tm_down = np.zeros_like(h, dtype=complex)
    t_te_down = np.zeros_like(h, dtype=complex)
    t_tm_down = np.zeros_like(h, dtype=complex)

    if n_layers > 1:
        r_te_down[1] = r_te_if[0]
        r_tm_down[1] = r_tm_if[0]
        t_te_down[1] = t_te_if[0]
        t_tm_down[1] = t_tm_if[0]

    for layer in range(2, dipole_layer + 1):
        iface = layer - 1
        phase = np.exp(2j * h[layer - 1] * thickness[layer - 1])
        phase_half = np.exp(1j * h[layer - 1] * thickness[layer - 1])

        r_te_down[layer] = (r_te_if[iface] + r_te_down[layer - 1] * phase) / (
            1 + r_te_if[iface] * r_te_down[layer - 1] * phase
        )
        r_tm_down[layer] = (r_tm_if[iface] + r_tm_down[layer - 1] * phase) / (
            1 + r_tm_if[iface] * r_tm_down[layer - 1] * phase
        )
        t_te_down[layer] = (t_te_if[iface] * t_te_down[layer - 1] * phase_half) / (
            1 + r_te_if[iface] * r_te_down[layer - 1] * phase
        )
        t_tm_down[layer] = (t_tm_if[iface] * t_tm_down[layer - 1] * phase_half) / (
            1 + r_tm_if[iface] * r_tm_down[layer - 1] * phase
        )

    # -------------------------
    # Purcell factor
    # -------------------------
    hd = h[dipole_layer][None, :, :]
    pos = dipole_position_array[:, None, None]

    a_up_tm = r_tm_up[dipole_layer][None, :, :] * np.exp(2j * hd * pos)
    a_up_te = r_te_up[dipole_layer][None, :, :] * np.exp(2j * hd * pos)

    a_down_tm = r_tm_down[dipole_layer][None, :, :] * np.exp(
        2j * hd * (thickness[dipole_layer] - pos)
    )
    a_down_te = r_te_down[dipole_layer][None, :, :] * np.exp(
        2j * hd * (thickness[dipole_layer] - pos)
    )

    a_tm = a_up_tm * a_down_tm
    a_te = a_up_te * a_down_te

    u2 = np.asarray(u)[:, None]
    sqrt_term = np.sqrt(1 - u2**2)

    k_tm_v = 3 / 4 * np.real(
        u2**2 / sqrt_term * (1 + a_up_tm) * (1 + a_down_tm) / (1 - a_tm)
    )
    k_tm_h = 3 / 8 * np.real(
        sqrt_term * (1 - a_up_tm) * (1 - a_down_tm) / (1 - a_tm)
    )
    k_te_h = 3 / 8 * np.real(
        1 / sqrt_term * (1 + a_up_te) * (1 + a_down_te) / (1 - a_te)
    )

    f_tm_v = np.trapezoid(2 * u2[None, :, :] * k_tm_v, x=u, axis=1)
    f_tm_h = np.trapezoid(2 * u2[None, :, :] * k_tm_h, x=u, axis=1)
    f_te_h = np.trapezoid(2 * u2[None, :, :] * k_te_h, x=u, axis=1)

    return k_tm_v, k_tm_h, k_te_h, f_tm_v, f_tm_h, f_te_h


def spectrum_bottom(
    f_tm_v: np.ndarray,
    f_tm_h: np.ndarray,
    f_te_h: np.ndarray,
    pl_spectrum: np.ndarray,
    ved: float,
    plqy: float,
    wavelength: np.ndarray,
    epsilon: np.ndarray,
    thickness: np.ndarray,
    dipole_layer: int,
    dipole_position_array: np.ndarray,
    u: np.ndarray,
    h: np.ndarray,
    theta_air: np.ndarray,
):
    """
    dipole_layer: 0-based layer index
    """
    n_layers = thickness.size
    hed = 1 - ved

    # -------------------------
    # Upward stack
    # -------------------------
    r_te_if, t_te_if, r_tm_if, t_tm_if = _fresnel_up(epsilon, h)

    r_te_up = np.zeros_like(h, dtype=complex)
    r_tm_up = np.zeros_like(h, dtype=complex)
    t_te_up = np.zeros_like(h, dtype=complex)
    t_tm_up = np.zeros_like(h, dtype=complex)

    seed_layer = n_layers - 3
    if seed_layer >= 0:
        r_te_up[seed_layer] = r_te_if[seed_layer]
        r_tm_up[seed_layer] = r_tm_if[seed_layer]
        t_te_up[seed_layer] = t_te_if[seed_layer]
        t_tm_up[seed_layer] = t_tm_if[seed_layer]

    for layer in range(seed_layer - 1, dipole_layer - 1, -1):
        iface = layer
        phase = np.exp(2j * h[layer + 1] * thickness[layer + 1])
        phase_half = np.exp(1j * h[layer + 1] * thickness[layer + 1])

        r_te_up[layer] = (r_te_if[iface] + r_te_up[layer + 1] * phase) / (
            1 + r_te_if[iface] * r_te_up[layer + 1] * phase
        )
        r_tm_up[layer] = (r_tm_if[iface] + r_tm_up[layer + 1] * phase) / (
            1 + r_tm_if[iface] * r_tm_up[layer + 1] * phase
        )
        t_te_up[layer] = (t_te_if[iface] * t_te_up[layer + 1] * phase_half) / (
            1 + r_te_if[iface] * r_te_up[layer + 1] * phase
        )
        t_tm_up[layer] = (t_tm_if[iface] * t_tm_up[layer + 1] * phase_half) / (
            1 + r_tm_if[iface] * r_tm_up[layer + 1] * phase
        )

    # glass-air interface = compact interface n_layers - 2
    idx_glass_air = n_layers - 2

    t_te_glass_air = h[n_layers - 1] / h[n_layers - 2] * t_te_if[idx_glass_air] ** 2
    t_tm_glass_air = h[n_layers - 1] / h[n_layers - 2] * t_tm_if[idx_glass_air] ** 2
    r_te_glass_air = r_te_if[idx_glass_air] ** 2
    r_tm_glass_air = r_tm_if[idx_glass_air] ** 2

    t_te_eml_glass = h[n_layers - 2] / h[dipole_layer] * np.abs(
        t_te_up[dipole_layer]
    ) ** 2
    t_tm_eml_glass = h[n_layers - 2] / h[dipole_layer] * np.abs(
        t_tm_up[dipole_layer]
    ) ** 2

    # -------------------------
    # Downward stack
    # MATLAB Spectrum_bottom uses n = 3:N here
    # so we recurse all the way up to the coherent/incoherent interface
    # -------------------------
    r_te_if, t_te_if, r_tm_if, t_tm_if = _fresnel_down(epsilon, h)

    r_te_down = np.zeros_like(h, dtype=complex)
    r_tm_down = np.zeros_like(h, dtype=complex)
    t_te_down = np.zeros_like(h, dtype=complex)
    t_tm_down = np.zeros_like(h, dtype=complex)

    if n_layers > 1:
        r_te_down[1] = r_te_if[0]
        r_tm_down[1] = r_tm_if[0]
        t_te_down[1] = t_te_if[0]
        t_tm_down[1] = t_tm_if[0]

    for layer in range(2, n_layers):
        iface = layer - 1
        phase = np.exp(2j * h[layer - 1] * thickness[layer - 1])
        phase_half = np.exp(1j * h[layer - 1] * thickness[layer - 1])

        r_te_down[layer] = (r_te_if[iface] + r_te_down[layer - 1] * phase) / (
            1 + r_te_if[iface] * r_te_down[layer - 1] * phase
        )
        r_tm_down[layer] = (r_tm_if[iface] + r_tm_down[layer - 1] * phase) / (
            1 + r_tm_if[iface] * r_tm_down[layer - 1] * phase
        )
        t_te_down[layer] = (t_te_if[iface] * t_te_down[layer - 1] * phase_half) / (
            1 + r_te_if[iface] * r_te_down[layer - 1] * phase
        )
        t_tm_down[layer] = (t_tm_if[iface] * t_tm_down[layer - 1] * phase_half) / (
            1 + r_tm_if[iface] * r_tm_down[layer - 1] * phase
        )

    r_c_te = np.abs(r_te_down[n_layers - 2]) ** 2
    r_c_tm = np.abs(r_tm_down[n_layers - 2]) ** 2

    # -------------------------
    # Dipole position
    # -------------------------
    hd = h[dipole_layer][None, :, :]
    pos = dipole_position_array[:, None, None]

    a_up_tm = r_tm_up[dipole_layer][None, :, :] * np.exp(2j * hd * pos)
    a_up_te = r_te_up[dipole_layer][None, :, :] * np.exp(2j * hd * pos)

    a_down_tm = r_tm_down[dipole_layer][None, :, :] * np.exp(
        2j * hd * (thickness[dipole_layer] - pos)
    )
    a_down_te = r_te_down[dipole_layer][None, :, :] * np.exp(
        2j * hd * (thickness[dipole_layer] - pos)
    )

    a_tm = a_up_tm * a_down_tm
    a_te = a_up_te * a_down_te

    # u may be (n_u,) or (n_u, n_w)
    u_arr = np.asarray(u)
    if u_arr.ndim == 1:
        u_fac = u_arr[None, :, None]
    elif u_arr.ndim == 2:
        u_fac = u_arr[None, :, :]
    else:
        raise ValueError(f"u must be 1D or 2D, got shape {u_arr.shape}")

    sqrt_term = np.sqrt(1 - u_fac**2)

    k_tm_v = (
        3 / 8
        * (u_fac**2 / sqrt_term)
        * np.abs(1 + a_down_tm) ** 2
        / np.abs(1 - a_tm) ** 2
        * t_tm_eml_glass[None, :, :]
    )
    k_tm_h = (
        3 / 16
        * sqrt_term
        * np.abs(1 - a_down_tm) ** 2
        / np.abs(1 - a_tm) ** 2
        * t_tm_eml_glass[None, :, :]
    )
    k_te_h = (
        3 / 16
        * (1 / sqrt_term)
        * np.abs(1 + a_down_te) ** 2
        / np.abs(1 - a_te) ** 2
        * t_te_eml_glass[None, :, :]
    )

    k_out_tm_v = k_tm_v * t_tm_glass_air[None, :, :] / (
        1 - r_tm_glass_air[None, :, :] * r_c_tm[None, :, :]
    )
    k_out_tm_h = k_tm_h * t_tm_glass_air[None, :, :] / (
        1 - r_tm_glass_air[None, :, :] * r_c_tm[None, :, :]
    )
    k_out_te_h = k_te_h * t_te_glass_air[None, :, :] / (
        1 - r_te_glass_air[None, :, :] * r_c_te[None, :, :]
    )

    theta_arr = np.asarray(theta_air)
    if theta_arr.ndim == 1:
        cos_theta = np.cos(theta_arr)[None, :, None]
        sin_theta = np.sin(theta_arr)[None, :, None]
        theta_deg = np.rad2deg(theta_arr)
    elif theta_arr.ndim == 2:
        cos_theta = np.cos(theta_arr)[None, :, :]
        sin_theta = np.sin(theta_arr)[None, :, :]
        theta_deg = np.rad2deg(theta_arr)
    else:
        raise ValueError(f"theta_air must be 1D or 2D, got shape {theta_arr.shape}")

    e_ratio = (epsilon[n_layers - 1] / epsilon[dipole_layer])[None, None, :]

    p_out_tm_v = e_ratio * cos_theta / np.pi * k_out_tm_v
    p_out_tm_h = e_ratio * cos_theta / np.pi * k_out_tm_h
    p_out_te_h = e_ratio * cos_theta / np.pi * k_out_te_h
    p_out = p_out_tm_v * ved + (p_out_te_h + p_out_tm_h) * hed

    f = f_tm_v * ved + (f_tm_h + f_te_h) * hed

    wavelength_1d = np.asarray(wavelength).reshape(-1)
    pl_1d = np.asarray(pl_spectrum).reshape(-1)

    quantum_factor = pl_1d[None, :] * plqy / (f * plqy + (1 - plqy))
    prefactor = (1240.0 / wavelength_1d)[None, :] * quantum_factor
    prefactor_3d = prefactor[:, None, :]

    i_tm_v = prefactor_3d * p_out_tm_v
    i_tm_h = prefactor_3d * p_out_tm_h
    i_te_h = prefactor_3d * p_out_te_h

    i_tm = i_tm_v * ved + i_tm_h * hed
    i_te = i_te_h * hed

    u_energy = (np.pi**2 / 90.0) * np.trapezoid(
        sin_theta * p_out,
        x=theta_deg,
        axis=1,
    )
    out_effi = np.trapezoid(quantum_factor * u_energy, x=wavelength_1d, axis=1)

    return i_tm, i_te, out_effi, u_energy