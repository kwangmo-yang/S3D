from __future__ import annotations
from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd().resolve()

os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import S3DConfig
from core import purcell_bottom, spectrum_bottom
from io_utils import nk2epsilon, read_matrix, trapz_norm

def main():
    current_folder = BASE_DIR

    # %% Initialization
    cfg = S3DConfig(
        wavelength=np.arange(400, 701, 1, dtype=float),
        pl_filename='5CzBN_PL_sampleA.txt',
        el_filename='5CzBN_EL_ETL_109_sampleA.txt',
        ved=0.25,
        plqy=1.0,
        thickness_nm=np.array([np.inf, 100, 1.5, 109, 5, 30, 5, 60, 10, 150, 700000, np.inf], dtype=float),
        eml_position_1based=6,
        delta_x_nm=5.0,
    )
    
    # %% PL spectrum
    wavelength = cfg.wavelength.astype(float)
    pl_spectrum_raw = read_matrix(current_folder / 'PL' / cfg.pl_filename)
    el_spectrum_raw = read_matrix(current_folder / 'EL' / cfg.el_filename)

    # %% Variable reshaping
    i_el = np.interp(wavelength, el_spectrum_raw[:, 0], el_spectrum_raw[:, 1])
    i_el = i_el / np.max(i_el)

    pl_spectrum = np.interp(wavelength, pl_spectrum_raw[:, 0], pl_spectrum_raw[:, 1])
    pl_spectrum = trapz_norm(pl_spectrum, wavelength)

    air = np.interp(wavelength, np.array([200.0, 1920.0]), np.array([1.003, 1.003])).astype(complex)

    # %% Refractive index
    nk_dir = current_folder / 'nk'
    glass = nk2epsilon(nk_dir / 'Glass.txt', wavelength, kind='linear', extrapolate=False)
    ito = nk2epsilon(nk_dir / 'ITO.txt', wavelength, kind='linear', extrapolate=False)
    hatcn = nk2epsilon(nk_dir / 'HAT-CN.xlsx', wavelength)
    bcfn = nk2epsilon(nk_dir / 'BCFN.txt', wavelength)
    siczcz = nk2epsilon(nk_dir / 'SiCzCz.csv', wavelength)
    msitrz = nk2epsilon(nk_dir / 'mSiTrz.csv', wavelength)
    msitrz_liq = nk2epsilon(nk_dir / 'mSiTrz_Liq_1050_1061.csv', wavelength)
    liq = nk2epsilon(nk_dir / 'Liq.csv', wavelength)
    al = nk2epsilon(nk_dir / 'Al.txt', wavelength, kind='linear', extrapolate=False)
    eml = nk2epsilon(nk_dir / 'EML_50_50_281_Blue.csv', wavelength)

    # %% Structure
    thickness_nm = cfg.thickness_nm.copy()
    epsilon = np.vstack([air, al, liq, msitrz_liq, msitrz, eml, siczcz, bcfn, hatcn, ito, glass, air])
    eml_position = cfg.eml_position_1based
    delta_x_nm = cfg.delta_x_nm

    # %% Variable initialization
    thickness = thickness_nm * 1e-9
    eml_idx = eml_position - 1
    dipole_position_array = np.arange(0.0, thickness_nm[eml_idx] + delta_x_nm, delta_x_nm) * 1e-9

    x_res = 1.25e-3
    x_real = np.arange(-np.pi / 2, -x_res + 1e-15, x_res)
    x_imag = 1j * np.arange(x_res, np.pi / 2 + 1e-15, 10 * x_res)
    x = np.concatenate([x_real, x_imag])
    u_purcell = np.cos(x)

    k = 2 * np.pi * np.sqrt(epsilon) / (wavelength[None, :] * 1e-9)
    h_purcell = k[eml_idx][None, :] * np.sqrt(
        epsilon[:, None, :] / epsilon[eml_idx][None, None, :] - u_purcell[:, None] ** 2
    )

    theta_air = np.deg2rad(np.arange(0, 90, 1, dtype=float))
    u_air = np.sin(theta_air)[:, None] * k[-1][None, :] / k[eml_idx][None, :]
    h_air = k[eml_idx][None, :] * np.sqrt(
        epsilon[:, None, :] / epsilon[eml_idx][None, None, :] - u_air[None, :, :] ** 2
    )

    # %% A_ex calculation
    _, _, _, f_tm_v, f_tm_h, f_te_h = purcell_bottom(
        epsilon=epsilon,
        thickness=thickness,
        dipole_layer=eml_idx,
        dipole_position_array=dipole_position_array,
        u=u_purcell,
        h=h_purcell,
    )

    i_tm, i_te, _, _ = spectrum_bottom(
        f_tm_v=f_tm_v,
        f_tm_h=f_tm_h,
        f_te_h=f_te_h,
        pl_spectrum=pl_spectrum,
        ved=cfg.ved,
        plqy=cfg.plqy,
        wavelength=wavelength,
        epsilon=epsilon,
        thickness=thickness,
        dipole_layer=eml_idx,
        dipole_position_array=dipole_position_array,
        u=u_air,
        h=h_air,
        theta_air=theta_air,
    )

    i_total = i_tm + i_te
    a_ex = np.abs(i_total[:, 0, :]).T
    a_ex = a_ex / np.max(a_ex)

    # %% Fitting
    alphalist = np.arange(0, 2.0001, 0.05)
    n_pos = dipole_position_array.size
    x0 = np.ones(n_pos)
    bounds = (np.zeros(n_pos), 2 * np.ones(n_pos))

    exc_all = np.zeros((alphalist.size, n_pos))
    resnorm = np.zeros(alphalist.size)

    eye = np.eye(n_pos)
    zeros = np.zeros(n_pos)

    for i, alpha in enumerate(alphalist):
        a_aug = np.vstack([a_ex, alpha ** 2 * eye])
        b_aug = np.concatenate([i_el, zeros])

        def residual(r):
            return a_aug @ r - b_aug

        res = least_squares(
            residual,
            x0,
            bounds=bounds,
            method='trf',
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        exc = res.x
        exc_all[i, :] = exc / np.trapezoid(exc)
        resnorm[i] = np.sum(res.fun ** 2)

    solnorm = np.linalg.norm(exc_all, axis=1)
    idx = int(np.argmin(resnorm ** 2 + solnorm ** 2))

    # %% Plot
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.axvspan(-5, 0, ymin=0, ymax=1, color=np.array([125, 158, 134]) / 255.0)
    plt.axvspan(
        dipole_position_array[-1] * 1e9,
        dipole_position_array[-1] * 1e9 + 5,
        ymin=0,
        ymax=1,
        color=np.array([166, 146, 131]) / 255.0,
    )
    plt.plot(
        dipole_position_array * 1e9,
        exc_all[idx, :],
        color=np.array([0, 0, 180]) / 255.0,
        linewidth=2.5,
        marker='s',
    )
    plt.xlim([-2, dipole_position_array[-1] * 1e9 + 2])
    plt.ylim([0, np.max(exc_all[idx, :]) * 1.1])
    plt.xlabel('$x$ (nm)')
    plt.ylabel('$N(x)$ (a.u.)')

    plt.subplot(1, 2, 2)
    plt.plot(
        wavelength,
        i_el,
        color=np.array([0, 0, 180]) / 255.0,
        linewidth=2.5,
        marker='s',
        markevery=20,
        label='$I_{EL}$',
    )
    i_fit = a_ex @ exc_all[idx, :]
    plt.plot(
        wavelength,
        i_fit / np.max(i_fit),
        color=(0 / 255, 0 / 255, 180 / 255, 0.5),
        linewidth=5,
        label='Fit',
    )
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized intensity (a.u.)')
    plt.ylim([0, 1.1])
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
