from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class S3DConfig:
    wavelength: np.ndarray
    pl_filename: str
    el_filename: str | None
    ved: float
    plqy: float
    thickness_nm: np.ndarray
    eml_position_1based: int
    ctl_position_1based: int | None = None
    ctl_scan_nm: np.ndarray | None = None
    delta_x_nm: float = 5.0
