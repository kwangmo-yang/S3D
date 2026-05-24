from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd().resolve()

os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from io_utils import read_matrix, nk_to_epsilon, interp_complex

def phys_sqrt(z):
    s = np.sqrt(z + 0j)
    mask = np.imag(s) < 0
    s[mask] = -s[mask]
    mask = (np.abs(np.imag(s)) < 1e-14) & (np.real(s) < 0)
    s[mask] = -s[mask]
    return s


def eps_to_nk(eps):
    root = phys_sqrt(eps)
    n = np.real(root)
    k = np.imag(root)
    return n, k


def plot_one(material_file, kind="cubic", extrapolate=True, wl_min=400, wl_max=700):
    base_dir = BASE_DIR
    nk_dir = base_dir / "nk"

    wavelength = np.arange(wl_min, wl_max + 1, 1, dtype=float)

    data = read_matrix(nk_dir / material_file)
    wl_raw, eps_raw = nk_to_epsilon(data)
    n_raw, k_raw = eps_to_nk(eps_raw)

    eps_interp = interp_complex(
        wl_raw, eps_raw, wavelength, kind=kind, extrapolate=extrapolate
    )
    n_interp, k_interp = eps_to_nk(eps_interp)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    axes[0, 0].plot(wl_raw, n_raw, "o", ms=3, label="raw n")
    axes[0, 0].plot(wavelength, n_interp, "-", lw=2, label=f"{kind} interp n")
    axes[0, 0].set_title(f"{material_file} : n")
    axes[0, 0].set_xlabel("Wavelength (nm)")
    axes[0, 0].set_ylabel("n")
    axes[0, 0].legend()

    axes[0, 1].plot(wl_raw, k_raw, "o", ms=3, label="raw k")
    axes[0, 1].plot(wavelength, k_interp, "-", lw=2, label=f"{kind} interp k")
    axes[0, 1].set_title(f"{material_file} : k")
    axes[0, 1].set_xlabel("Wavelength (nm)")
    axes[0, 1].set_ylabel("k")
    axes[0, 1].legend()

    axes[1, 0].plot(wavelength, np.real(eps_interp), "-", lw=2)
    axes[1, 0].set_title(f"{material_file} : Re(epsilon)")
    axes[1, 0].set_xlabel("Wavelength (nm)")
    axes[1, 0].set_ylabel("Re(eps)")

    axes[1, 1].plot(wavelength, np.imag(eps_interp), "-", lw=2)
    axes[1, 1].set_title(f"{material_file} : Im(epsilon)")
    axes[1, 1].set_xlabel("Wavelength (nm)")
    axes[1, 1].set_ylabel("Im(eps)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_one("SiCzCz.csv", kind="cubic", extrapolate=True)
    plot_one("mSiTrz.csv", kind="cubic", extrapolate=True)
    plot_one("mSiTrz_Liq_1050_1061.csv", kind="cubic", extrapolate=True)
    plot_one("Liq.csv", kind="cubic", extrapolate=True)
    plot_one("EML_50_50_281_Blue.csv", kind="cubic", extrapolate=True)