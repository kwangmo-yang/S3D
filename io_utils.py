from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline


def _read_numeric_lines(path):
    rows = []

    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line = line.replace(",", " ").replace(";", " ").replace("\t", " ")

            parts = line.split()
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass

            if len(nums) >= 2:
                rows.append(nums)

    if not rows:
        raise ValueError(f"No numeric rows found in {path}")

    min_cols = min(len(r) for r in rows)
    rows = [r[:min_cols] for r in rows]

    return np.array(rows, dtype=float)


def read_matrix(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, header=None)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(how="all").dropna(how="all", axis=1)
        return df.to_numpy(dtype=float)

    elif suffix in [".csv", ".txt", ".dat"]:
        return _read_numeric_lines(path)

    else:
        return _read_numeric_lines(path)


def trapz_norm(y: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
    area = np.trapezoid(y, x=x)
    if area == 0:
        return y
    return y / area


def interp_complex(
    x_src,
    y_src,
    x_new,
    kind="linear",
    extrapolate=False,
):
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=complex)
    x_new = np.asarray(x_new, dtype=float)

    if kind == "spline":
        f_real = CubicSpline(x_src, np.real(y_src), extrapolate=extrapolate)
        f_imag = CubicSpline(x_src, np.imag(y_src), extrapolate=extrapolate)
        return f_real(x_new) + 1j * f_imag(x_new)

    fill_value = "extrapolate" if extrapolate else np.nan
    f_real = interp1d(
        x_src,
        np.real(y_src),
        kind=kind,
        bounds_error=False,
        fill_value=fill_value,
    )
    f_imag = interp1d(
        x_src,
        np.imag(y_src),
        kind=kind,
        bounds_error=False,
        fill_value=fill_value,
    )
    return f_real(x_new) + 1j * f_imag(x_new)


def nk_to_epsilon(data):
    data = np.asarray(data)

    if data.ndim == 1:
        raise ValueError(
            f"nk data is 1D, not 2D. Check delimiter/header. shape={data.shape}"
        )

    if data.shape[1] < 3:
        raise ValueError(
            f"nk data must have at least 3 columns: wavelength, n, k. shape={data.shape}"
        )

    wavelength = data[:, 0]
    n = data[:, 1]
    k = data[:, 2]

    eps = (n + 1j * k) ** 2
    return wavelength, eps


def nk2epsilon(path, wavelength, kind: str = "cubic", extrapolate: bool = True):
    """
    Load an n-k file and directly convert/interpolate it to epsilon on the
    target wavelength grid, mirroring the simplified MATLAB workflow.
    """
    data = read_matrix(path)
    wl_src, eps_src = nk_to_epsilon(data)
    return interp_complex(wl_src, eps_src, wavelength, kind=kind, extrapolate=extrapolate)