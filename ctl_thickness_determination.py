from __future__ import annotations
from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

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
        el_filename=None,
        ved=0.25,
        plqy=1.0,
        thickness_nm=np.array([np.inf, 100, 1.5, 40, 5, 30, 5, 60, 10, 150, 700000, np.inf], dtype=float),
        eml_position_1based=6,
        ctl_position_1based=4,
        ctl_scan_nm=np.arange(30, 131, 1, dtype=float),
        delta_x_nm=5.0,
    )

    # %% PL spectrum
    wavelength = cfg.wavelength.astype(float)
    pl_spectrum_raw = read_matrix(current_folder / 'PL' / cfg.pl_filename)

    # %% Variable reshaping
    pl_spectrum = np.interp(wavelength, pl_spectrum_raw[:, 0], pl_spectrum_raw[:, 1])
    pl_spectrum = trapz_norm(pl_spectrum, wavelength)

    air = np.interp(
        wavelength,
        np.array([200.0, 1920.0]),
        np.array([1.003, 1.003])
    ).astype(complex)

    # %% Refractive index
    nk_dir = current_folder / 'nk'
    glass = nk2epsilon(nk_dir / 'Glass.txt', wavelength, kind='spline', extrapolate=True)
    ito = nk2epsilon(nk_dir / 'ITO.txt', wavelength, kind='spline', extrapolate=True)
    hatcn = nk2epsilon(nk_dir / 'HAT-CN.xlsx', wavelength, kind='spline', extrapolate=True)
    bcfn = nk2epsilon(nk_dir / 'BCFN.txt', wavelength, kind='spline', extrapolate=True)
    siczcz = nk2epsilon(nk_dir / 'SiCzCz.csv', wavelength, kind='spline', extrapolate=True)
    msitrz = nk2epsilon(nk_dir / 'mSiTrz.csv', wavelength, kind='spline', extrapolate=True)
    msitrz_liq = nk2epsilon(nk_dir / 'mSiTrz_Liq_1050_1061.csv', wavelength, kind='spline', extrapolate=True)
    liq = nk2epsilon(nk_dir / 'Liq.csv', wavelength, kind='spline', extrapolate=True)
    al = nk2epsilon(nk_dir / 'Al.txt', wavelength, kind='spline', extrapolate=True)
    eml = nk2epsilon(nk_dir / 'EML_50_50_281_Blue.csv', wavelength, kind='spline', extrapolate=True)

    # %% Structure
    thickness_nm = cfg.thickness_nm.copy()
    epsilon = np.vstack([air, al, liq, msitrz_liq, msitrz, eml, siczcz, bcfn, hatcn, ito, glass, air])
    eml_position = cfg.eml_position_1based
    ctl_position = cfg.ctl_position_1based
    ctl_scan_nm = cfg.ctl_scan_nm
    delta_x_nm = cfg.delta_x_nm

    # %% Variable initialization
    thickness = thickness_nm * 1e-9
    eml_idx = eml_position - 1
    ctl_idx = ctl_position - 1
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

    # %% CTL thickness determination
    min_res = np.zeros_like(ctl_scan_nm)
    n_pos = dipole_position_array.size

    for i, ctl_nm in enumerate(ctl_scan_nm):
        thickness_i = thickness.copy()
        thickness_i[ctl_idx] = ctl_nm * 1e-9

        _, _, _, f_tm_v, f_tm_h, f_te_h = purcell_bottom(
            epsilon=epsilon,
            thickness=thickness_i,
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
            thickness=thickness_i,
            dipole_layer=eml_idx,
            dipole_position_array=dipole_position_array,
            u=u_air,
            h=h_air,
            theta_air=theta_air,
        )

        i_total = i_tm + i_te
        a_ex = np.abs(i_total[:, 0, :]).T

        res_test = np.zeros(n_pos)
        for j in range(n_pos):
            test_i = a_ex[:, j]
            v_minus_i = np.delete(a_ex, j, axis=1)
            ans = lsq_linear(
                v_minus_i,
                test_i,
                bounds=(0, np.inf),
                method='trf',
                lsq_solver='exact',
                tol=1e-12,
            )
            res_test[j] = np.linalg.norm(test_i - v_minus_i @ ans.x) / np.linalg.norm(test_i)

        min_res[i] = np.min(res_test)
        print(f'CTL thickness: {ctl_nm:.0f} nm')

    # %% Plot
    plt.figure(figsize=(6, 4))
    plt.plot(ctl_scan_nm, min_res * 100, 'k', linewidth=2)
    plt.axhline(3, linestyle=':', linewidth=1.5)
    plt.xlabel('CTL thickness (nm)')
    plt.ylabel('min($r_i$) (%)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()