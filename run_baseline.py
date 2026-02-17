"""
Baseline PTA anisotropy experiment using the modular pta_anisotropy package.

This reproduces the structure of oo.py but delegates all core functionality
to the shared modules. It uses the standard sinc window implementation.
"""

import os
import numpy as np
import jax.numpy as jnp
import tqdm

from pta_anisotropy import (
    constants,
    spherical,
    data_model,
    simulation,
    gamma_tensors,
    estimation,
    freq_config,
)


def main():
    if "__file__" in globals():
        base_path = os.path.dirname(os.path.abspath(__file__))
    else:
        base_path = os.getcwd()

    # Outputs now live under ./output_dir/baseline
    output_root = constants.get_output_directory(base_path, "output_dir")
    output_dir = constants.get_output_directory(output_root, "baseline")

    print("=" * 70)
    print("FLEXIBLE FREQUENCY-DEPENDENT ANISOTROPY ANALYSIS (BASELINE WINDOW)")
    print("=" * 70)

    # Simulation parameters (mirroring oo.py)
    N_runs = 500
    NN_pulsars = [200]
    Tspan_yrs = 16.0
    Tspan = Tspan_yrs * constants.yr
    nfreqs = 15
    df_step = 0.1
    f_in = 0.5

    # Full frequency arrays
    fi_full = jnp.arange(1, nfreqs + 1) / Tspan
    ff = jnp.arange(f_in, nfreqs + 1, step=df_step) / Tspan

    S_fi_full = (fi_full / constants.f_yr) ** (-7.0 / 3.0)
    S_ff = (ff / constants.f_yr) ** (-7.0 / 3.0)

    # Spherical harmonics parameters
    Nside = 12
    l_max = 1
    n_params = spherical.get_n_coefficients_real(l_max)

    # Test cases (copied from oo.py)
    test_cases = [
        {
            "name": "Inject_Low4_Recon_High",
            "injection_bins": np.arange(0, 3),
            "reconstruction_bins": np.arange(3, 15),
            "description": "Null test: anisotropy in low freq, fit high freq",
        },
        {
            "name": "Inject_Low4_Recon_Mid",
            "injection_bins": np.arange(0, 4),
            "reconstruction_bins": np.arange(5, 10),
            "description": "Null test: anisotropy in low freq, fit middle freq",
        },
        {
            "name": "Inject_Low4_Recon_All",
            "injection_bins": np.arange(0, 4),
            "reconstruction_bins": np.arange(0, 15),
            "description": "Recovery test: should recover dipole from low freq",
        },
        {
            "name": "Inject_Mid_Recon_Edges",
            "injection_bins": np.arange(5, 10),
            "reconstruction_bins": np.concatenate(
                [np.arange(0, 5), np.arange(10, 15)]
            ),
            "description": "Null test: anisotropy in middle, fit edges",
        },
    ]

    # Select which test to run
    current_test = test_cases[0]  # ← change index 0, 1, 2, or 3

    fcfg = freq_config.FrequencyConfig(
        fi_full=np.array(fi_full),
        injection_bins=current_test["injection_bins"],
        reconstruction_bins=current_test["reconstruction_bins"],
    )

    print(f"\nRUNNING TEST: {current_test['name']}")
    print(f"Description: {current_test['description']}")
    fcfg.print_summary(constants.yr)
    print()

    # Derived arrays for reconstruction
    fi_recon = fcfg.fi_recon
    S_fi_recon = S_fi_full[fcfg.reconstruction_bins]
    f0_recon = fi_recon ** 0
    S_f0_recon = 3 / 4 * (2 * jnp.pi) ** 2 * S_fi_recon ** 0

    scenarios = [
        {"add_dipole": True, "add_quadropole": False, "name": "With_Dipole"},
    ]

    for scenario in scenarios:
        add_dipole = scenario["add_dipole"]
        add_quadropole = scenario["add_quadropole"]

        # True sky coefficients
        clms_real_peak = np.zeros(n_params)
        clms_real_peak[0] = 1 / np.sqrt(4 * np.pi)
        if add_dipole:
            clms_real_peak[2] = 1 / np.sqrt(4 * np.pi) / np.sqrt(3)
        if add_quadropole:
            clms_real_peak[6] = 1 / np.sqrt(5 * np.pi)

        print(f"\nSCENARIO: {scenario['name']}")
        print("True coefficients (in injection region):")
        print(f"  Monopole: {clms_real_peak[0]:.6f}")
        if add_dipole:
            print(f"  Dipole:   {clms_real_peak[2]:.6f}")
        if add_quadropole:
            print(f"  Quad:     {clms_real_peak[6]:.6f}")
        print()

        # Anisotropy injection
        H_p_ff = freq_config.create_H_p_ff_binned(
            np.array(fi_full), ff, fcfg, clms_real_peak, Nside, l_max, np.array(S_ff)
        )

        # Normalization (baseline window)
        C_ff = data_model.get_D_IJ_fifj_normalization_baseline(
            Tspan, fi_full, ff, H_p_ff
        )
        inv_ff = estimation.compute_inverse(C_ff)

        for npulsars in NN_pulsars:
            outname = os.path.join(
                output_dir,
                f"{current_test['name']}_{scenario['name']}_{N_runs}_{npulsars}.npz",
            )

            # Per-run log lines that will also be written to a txt file
            log_lines = []
            log_lines.append(f"TEST: {current_test['name']}")
            log_lines.append(f"Description: {current_test['description']}")
            log_lines.append(f"Scenario: {scenario['name']}")
            log_lines.append(f"N_pulsars: {npulsars}")
            log_lines.append(f"NPZ output: {outname}")

            print(f"  Running N={npulsars} pulsars...")
            print(f"  Output: {outname}")

            means = np.zeros((N_runs, n_params))
            stds = np.zeros((N_runs, n_params))

            # PTA geometry
            (
                p_vec,
                cos_IJ,
                distances_pc,
                theta_k,
                phi_k,
            ) = simulation.generate_pulsar_sky_and_kpixels(npulsars, Nside)
            distances = simulation.distances_to_meters(distances_pc)
            gamma_IJ_lm = gamma_tensors.get_correlations_lm_IJ(
                p_vec, l_max, Nside
            )

            # Initial guess
            clm_guess = np.zeros(n_params)
            clm_guess[0] = clms_real_peak[0] * 1.5
            if add_dipole:
                clm_guess[2] = 0.05
            if add_quadropole:
                clm_guess[6] = 0.05

            failed_count = 0

            for nn in tqdm.tqdm(range(N_runs), desc="    Processing"):
                h_tilde = simulation.generate_hpc_polarization_pixel_frequency(
                    H_p_ff
                )

                D_IJ_full = data_model.get_D_IJ_fifj_baseline(
                    Tspan,
                    fi_full,
                    ff,
                    jnp.array(h_tilde),
                    jnp.array(distances),
                    p_vec,
                    theta_k,
                    phi_k,
                )

                rescale_D_IJ_full = jnp.diagonal(
                    jnp.einsum("fg,glab->flab", inv_ff, D_IJ_full),
                    axis1=0,
                    axis2=1,
                ).T

                # Extract reconstruction bins
                rescale_D_IJ_recon = rescale_D_IJ_full[fcfg.reconstruction_bins]

                try:
                    theta, uncertainties, _ = estimation.iterative_estimation(
                        estimation.get_update_estimate_diagonal,
                        clm_guess,
                        rescale_D_IJ_recon,
                        gamma_IJ_lm,
                        f0_recon,
                        S_f0_recon,
                        i_max=100,
                    )
                    means[nn] = theta
                    stds[nn] = uncertainties
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 3:
                        print(f"\n      ERROR in run {nn}: {e}")
                    means[nn] = np.nan
                    stds[nn] = np.nan

            # Save results
            np.savez(
                outname,
                means=means,
                stds=stds,
                clms_real_peak=clms_real_peak,
                injection_bins=fcfg.injection_bins,
                reconstruction_bins=fcfg.reconstruction_bins,
                test_name=current_test["name"],
                test_description=current_test["description"],
                fi_full=np.array(fi_full),
                fi_recon=np.array(fi_recon),
                p_vec=np.array(p_vec),
                cos_IJ=np.array(cos_IJ),
                Nside=Nside,
                l_max=l_max,
                nfreqs=nfreqs,
                Tspan_yrs=Tspan_yrs,
                npulsars=npulsars,
                N_runs=N_runs,
            )

            # Summary
            print(
                f"\n  Results from reconstruction bins {fcfg.reconstruction_bins}:"
            )
            log_lines.append("")
            log_lines.append(
                f"Results from reconstruction bins {fcfg.reconstruction_bins}:"
            )

            print(
                f"    Monopole: {np.nanmean(means[:, 0]):.6f} ± "
                f"{np.nanstd(means[:, 0]):.6f}"
            )
            print(f"              (true: {clms_real_peak[0]:.6f})")
            log_lines.append(
                f"    Monopole: {np.nanmean(means[:, 0]):.6f} ± "
                f"{np.nanstd(means[:, 0]):.6f}"
            )
            log_lines.append(f"              (true: {clms_real_peak[0]:.6f})")

            if add_dipole:
                dipole_mean = np.nanmean(means[:, 2])
                dipole_std = np.nanstd(means[:, 2])
                dipole_sigma = dipole_mean / dipole_std if dipole_std > 0 else 0
                has_overlap = (
                    len(
                        np.intersect1d(
                            fcfg.injection_bins, fcfg.reconstruction_bins
                        )
                    )
                    > 0
                )
                expected = clms_real_peak[2] if has_overlap else 0.0

                print(
                    f"    Dipole:   {dipole_mean:.6f} ± {dipole_std:.6f} "
                    f"({dipole_sigma:.2f}σ)"
                )
                print(
                    f"              (expected: {expected:.6f}, "
                    f"true in inj region: {clms_real_peak[2]:.6f})"
                )
                log_lines.append(
                    f"    Dipole:   {dipole_mean:.6f} ± {dipole_std:.6f} "
                    f"({dipole_sigma:.2f}σ)"
                )
                log_lines.append(
                    f"              (expected: {expected:.6f}, "
                    f"true in inj region: {clms_real_peak[2]:.6f})"
                )

            if add_quadropole:
                quad_mean = np.nanmean(means[:, 6])
                quad_std = np.nanstd(means[:, 6])
                quad_sigma = quad_mean / quad_std if quad_std > 0 else 0
                has_overlap = (
                    len(
                        np.intersect1d(
                            fcfg.injection_bins, fcfg.reconstruction_bins
                        )
                    )
                    > 0
                )
                expected = clms_real_peak[6] if has_overlap else 0.0

                print(
                    f"    Quad:     {quad_mean:.6f} ± {quad_std:.6f} "
                    f"({quad_sigma:.2f}σ)"
                )
                print(
                    f"              (expected: {expected:.6f}, "
                    f"true in inj region: {clms_real_peak[6]:.6f})"
                )
                log_lines.append(
                    f"    Quad:     {quad_mean:.6f} ± {quad_std:.6f} "
                    f"({quad_sigma:.2f}σ)"
                )
                log_lines.append(
                    f"              (expected: {expected:.6f}, "
                    f"true in inj region: {clms_real_peak[6]:.6f})"
                )

            print(f"\n    Saved: {outname}")
            if failed_count > 0:
                print(f"    WARNING: {failed_count}/{N_runs} runs failed")
                log_lines.append(
                    f"WARNING: {failed_count}/{N_runs} runs failed"
                )

            # Also save human-readable summary alongside the NPZ
            txt_path = os.path.splitext(outname)[0] + ".txt"
            with open(txt_path, "w") as f:
                f.write("\n".join(log_lines) + "\n")

    print("\n" + "=" * 70)
    print("BASELINE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

