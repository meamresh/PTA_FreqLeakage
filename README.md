# PTA Frequency Leakage Analysis

A Python package for studying frequency leakage effects in Pulsar Timing Array (PTA) anisotropy analysis. This codebase compares two frequency windowing approaches for gravitational wave background (GWB) analysis: baseline sinc windows and window-corrected implementations.

## Overview

This project investigates whether anisotropy injected in one frequency range leaks into other frequency bins during PTA data analysis. It performs Monte Carlo simulations to:

- Test frequency-dependent anisotropy injection and reconstruction
- Compare baseline vs. window-corrected frequency integration methods
- Assess leakage between non-overlapping frequency bins
- Validate recovery of dipole anisotropy from gravitational wave background


## Main Scripts

### 1. `run_baseline.py`

**What it does:** Runs Monte Carlo simulations using the baseline sinc window implementation for frequency integration.

**When to use:** Use this to establish baseline performance and compare against window-corrected results.

**Usage:**
```bash
python3 run_baseline.py
```

**Key features:**
- Uses standard `sinc((fi - ff) * Tspan)` window functions
- Performs 500 Monte Carlo runs per test case
- Tests multiple frequency bin configurations
- Saves results as `.npz` files in `output_dir/baseline/`

**Configuration:**
- Edit `current_test` (line 90) to select test case (0-3)
- Modify `N_runs`, `NN_pulsars`, `Tspan_yrs` for different simulation parameters
- Test cases include null tests and recovery tests

### 2. `run_windowed.py`

**What it does:** Runs identical simulations but uses window-corrected frequency integration kernels.

**When to use:** Use this to assess whether window corrections reduce frequency leakage compared to baseline.

**Usage:**
```bash
python3 run_windowed.py
```

**Key features:**
- Uses modified window functions with correction terms
- Same test cases as baseline for direct comparison
- Saves results in `output_dir/windowed/`
- Can be compared directly with baseline results

**Differences from baseline:**
- Adds correction terms: `cos(π * fi * Tspan)` and `sinc(ff * Tspan)`
- Modified window functions: `window_minus = sinc_minus + correction`
- Should reduce frequency leakage artifacts

### 3. `visualize_results.py`

**What it does:** Creates comprehensive visualizations of simulation results.

**When to use:** After running simulations, use this to analyze and compare results.

**Usage:**
```bash
# Visualize a single results file (all plots)
python3 visualize_results.py output_dir/baseline/Inject_Low4_Recon_High_With_Dipole_500_200.npz

# Generate only specific plots
python3 visualize_results.py output_dir/baseline/file.npz --plots distributions summary

# Process all NPZ files automatically
python3 visualize_results.py all

# Compare baseline vs windowed results
python3 visualize_results.py file.npz --baseline baseline/file.npz --windowed windowed/file.npz

# Save plots without displaying
python3 visualize_results.py file.npz --no-show --output-dir plots/
```

**Plot types:**
- `distributions`: Histograms of parameter estimates
- `comparison`: True vs estimated parameters with error bars
- `frequency`: Frequency bin configuration visualization
- `summary`: Statistics table with bias and pull calculations
- `baseline-windowed`: Side-by-side comparison of methods

## Output Format

Results are saved as `.npz` files containing:

- `means`: (N_runs, n_params) array of estimated parameters
- `stds`: (N_runs, n_params) array of uncertainties
- `clms_real_peak`: True injected coefficients
- `injection_bins`, `reconstruction_bins`: Frequency bin indices
- `test_name`, `test_description`: Metadata
- `fi_full`, `fi_recon`: Frequency arrays
- `p_vec`, `cos_IJ`: Pulsar geometry
- Simulation parameters (N_runs, npulsars, Tspan_yrs, etc.)

Human-readable `.txt` summaries are also generated alongside `.npz` files.

## Typical Workflow

1. **Run baseline simulations:**
   ```bash
   python3 run_baseline.py
   ```

2. **Run windowed simulations:**
   ```bash
   python3 run_windowed.py
   ```

3. **Visualize results:**
   ```bash
   # Individual analysis
   python3 visualize_results.py output_dir/baseline/Inject_Low4_Recon_High_With_Dipole_500_200.npz
   
   # Compare methods
   python3 visualize_results.py file.npz \
       --baseline output_dir/baseline/file.npz \
       --windowed output_dir/windowed/file.npz
   ```

4. **Analyze leakage:**
   - Check null tests (should show no significant detection)
   - Compare baseline vs windowed pull statistics
   - Examine frequency bin visualizations for overlap
s

## Project Structure

```
PTA_FreqLeakage/
├── pta_anisotropy/          # Core package modules
│   ├── constants.py        # Physical constants & JAX configuration
│   ├── freq_config.py     # Frequency bin configuration
│   ├── data_model.py      # Data model (baseline & windowed)
│   ├── simulation.py       # Monte Carlo simulation utilities
│   ├── estimation.py      # Parameter estimation algorithms
│   ├── spherical.py        # Spherical harmonics utilities
│   ├── gamma_tensors.py    # Overlap reduction tensor calculations
│   └── geometry.py         # Geometric helpers
├── run_baseline.py         # Baseline sinc window experiment
├── run_windowed.py         # Window-corrected experiment
├── visualize_results.py    # Visualization script
└── output_dir/            # Results storage
    ├── baseline/
    └── windowed/
```

## Dependencies

Required Python packages:
- `numpy`
- `jax` / `jax.numpy`
- `healpy`
- `scipy`
- `matplotlib` (for visualization)
- `tqdm`

## Test Cases

Both scripts support four test cases (select via `current_test` index):

1. **`Inject_Low4_Recon_High`** (index 0)
   - **Purpose:** Null test - anisotropy in low frequencies, fit high frequencies
   - **Injection bins:** 0-2 (low frequencies)
   - **Reconstruction bins:** 3-14 (high frequencies)
   - **Expected:** Should find no significant dipole (tests for leakage)

2. **`Inject_Low4_Recon_Mid`** (index 1)
   - **Purpose:** Null test - anisotropy in low frequencies, fit middle frequencies
   - **Injection bins:** 0-3
   - **Reconstruction bins:** 5-9
   - **Expected:** Should find no significant dipole

3. **`Inject_Low4_Recon_All`** (index 2)
   - **Purpose:** Recovery test - should recover dipole from low frequencies
   - **Injection bins:** 0-3
   - **Reconstruction bins:** 0-14 (all bins)
   - **Expected:** Should recover injected dipole signal

4. **`Inject_Mid_Recon_Edges`** (index 3)
   - **Purpose:** Null test - anisotropy in middle, fit edges
   - **Injection bins:** 5-9
   - **Reconstruction bins:** 0-4, 10-14 (edges)
   - **Expected:** Should find no significant dipole

## Core Package Modules (`pta_anisotropy/`)

### `constants.py`
- Physical constants (Hubble, parsec, light speed, year)
- JAX configuration (CPU device, x64 precision)
- Output directory management

### `freq_config.py`
- `FrequencyConfig`: Manages injection and reconstruction frequency bins
- `create_H_p_ff_binned()`: Creates frequency-dependent anisotropy maps
- Handles mapping between analysis frequencies (`fi`) and fine grid (`ff`)

### `data_model.py`
- **Baseline functions:**
  - `get_D_IJ_fifj_baseline()`: Covariance matrix with sinc windows
  - `get_D_IJ_fifj_normalization_baseline()`: Normalization factors
- **Windowed functions:**
  - `get_D_IJ_fifj_windowed()`: Covariance with corrected windows
  - `get_D_IJ_fifj_normalization_windowed()`: Windowed normalization
- Projects GWB onto PTA response functions

### `simulation.py`
- `generate_pulsar_sky_and_kpixels()`: Creates pulsar sky positions
- `generate_hpc_polarization_pixel_frequency()`: Generates GW signal realizations
- `distances_to_meters()`: Distance conversions
- Gaussian random field generation

### `estimation.py`
- `iterative_estimation()`: Maximum likelihood parameter estimation
- `get_update_estimate_diagonal()`: Fisher matrix-based updates
- `get_covariance_diagonal()`: Diagonal covariance computation
- Convergence checking

### `spherical.py`
- Real/complex spherical harmonics conversions
- HEALPix map generation from coefficients
- Handles scipy version compatibility

### `gamma_tensors.py`
- `get_correlations_lm_IJ()`: Computes overlap reduction function (ORF) tensor
- Spherical harmonic decomposition of pulsar correlations
- Handles pulsar pair geometry

### `geometry.py`
- Unit vector computations
- GW polarization tensors (plus/cross)
- PTA response functions `get_R_pc()`
- Frequency-dependent phase factors

## Key Parameters

- **N_runs**: Number of Monte Carlo realizations (default: 500)
- **NN_pulsars**: Number of pulsars in array (baseline: 200, windowed: 300)
- **Tspan_yrs**: Observation time span in years (default: 16.0)
- **nfreqs**: Number of frequency bins (default: 15)
- **l_max**: Maximum spherical harmonic degree (default: 1, monopole+dipole)
- **Nside**: HEALPix resolution (default: 12)

## Notes

- The code uses JAX for efficient computation with JIT compilation
- Failed runs are tracked and reported (NaN values in results)
- Frequency bins are indexed starting from 0
- Spherical harmonics use real-valued basis functions
- All frequencies are in Hz (converted from 1/Tspan units)

## Citation

If you use this code in your research, please cite appropriately and acknowledge the original work on PTA anisotropy analysis and frequency leakage studies.
