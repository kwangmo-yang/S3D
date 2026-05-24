# S3D Python

Python tools for optical simulation and spectral analysis of bottom-emitting OLED structures.

## Overview

This repository supports three main workflows:

1. plotting and checking refractive-index / extinction-coefficient (`n`, `k`) data,
2. determining charge-transport-layer (CTL) thickness from spectral matching, and
3. extracting the recombination zone (RZ) from electroluminescence (EL) spectra.

## Repository structure

```text
S3D_Python/
├── config.py                         # Shared simulation configuration dataclass
├── core.py                           # Core optical simulation functions
├── io_utils.py                       # Data loading, interpolation, and normalization helpers
├── plot_nk_current.py                # Plot and inspect optical-constant data
├── ctl_thickness_determination.py    # CTL-thickness determination workflow
├── rz_extraction.py                  # RZ extraction workflow
├── requirements.txt                  # Python dependencies
├── nk/                               # Optical constants
├── PL/                               # Photoluminescence spectra
└── EL/                               # Electroluminescence spectra
```

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/YOUR_USERNAME/S3D_Python.git
cd S3D_Python
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

For Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Input data

Place input files in the following folders:

- `nk/`: optical constants (`n`, `k`) as `.txt`, `.csv`, or `.xlsx`
- `PL/`: photoluminescence spectra
- `EL/`: electroluminescence spectra

Each spectrum or optical-constant file should contain numeric columns readable by `io_utils.read_matrix()`.

## How to run

Run scripts from the repository root.

### 1. Check optical constants

```bash
python plot_nk_current.py
```

### 2. Determine CTL thickness

```bash
python ctl_thickness_determination.py
```

### 3. Extract recombination zone

```bash
python rz_extraction.py
```

## Recommended workflow

1. Run `plot_nk_current.py` to verify the optical constants.
2. Run `ctl_thickness_determination.py` to identify the appropriate CTL thickness.
3. Run `rz_extraction.py` to extract the RZ from the measured EL spectrum.

## Notes

- Scripts assume relative paths from the repository root.
- `pandas` and `openpyxl` are needed to read spreadsheet-based optical-constant files such as `.xlsx`.
- Sample `PL`, `EL`, and `nk` files are included for demonstration and reproducibility.
