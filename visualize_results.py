"""
Visualization script for PTA frequency leakage analysis results.

This script provides comprehensive visualization of the Monte Carlo simulation
results, including parameter distributions, comparisons with true values, and
frequency bin configurations.

Usage Examples:
    # Visualize a single results file (all plots):
    python visualize_results.py output_dir/baseline/Inject_Low4_Recon_High_With_Dipole_500_200.npz

    # Generate only specific plots:
    python visualize_results.py output_dir/baseline/Inject_Low4_Recon_High_With_Dipole_500_200.npz --plots distributions summary

    # Process all NPZ files in output_dir:
    python visualize_results.py all

    # Compare baseline vs windowed results:
    python visualize_results.py output_dir/baseline/file.npz --baseline output_dir/baseline/file.npz --windowed output_dir/windowed/file.npz

    # Save plots without displaying:
    python visualize_results.py output_dir/baseline/file.npz --no-show --output-dir plots/
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from pta_anisotropy import constants


def load_results(npz_path):
    """Load results from an NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "means": data["means"],
        "stds": data["stds"],
        "clms_real_peak": data["clms_real_peak"],
        "injection_bins": data["injection_bins"],
        "reconstruction_bins": data["reconstruction_bins"],
        "test_name": str(data["test_name"]),
        "test_description": str(data["test_description"]),
        "fi_full": data["fi_full"],
        "fi_recon": data["fi_recon"],
        "p_vec": data["p_vec"],
        "cos_IJ": data["cos_IJ"],
        "Nside": int(data["Nside"]),
        "l_max": int(data["l_max"]),
        "nfreqs": int(data["nfreqs"]),
        "Tspan_yrs": float(data["Tspan_yrs"]),
        "npulsars": int(data["npulsars"]),
        "N_runs": int(data["N_runs"]),
    }


def get_parameter_labels(l_max):
    """Get labels for spherical harmonic parameters."""
    labels = []
    idx = 0
    for ell in range(l_max + 1):
        for m in range(-ell, ell + 1):
            if ell == 0:
                labels.append("Monopole (l=0, m=0)")
            elif ell == 1:
                if m == -1:
                    labels.append("Dipole Y (l=1, m=-1)")
                elif m == 0:
                    labels.append("Dipole Z (l=1, m=0)")
                elif m == 1:
                    labels.append("Dipole X (l=1, m=1)")
            elif ell == 2:
                labels.append(f"Quad (l=2, m={m})")
            else:
                labels.append(f"l={ell}, m={m}")
            idx += 1
    return labels


def plot_parameter_distributions(results, output_path=None, show=True):
    """Plot histograms of parameter estimates."""
    means = results["means"]
    clms_real_peak = results["clms_real_peak"]
    l_max = results["l_max"]
    n_params = means.shape[1]
    
    # Filter out NaN values
    valid_mask = ~np.isnan(means).any(axis=1)
    means_valid = means[valid_mask]
    
    labels = get_parameter_labels(l_max)
    
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(n_params):
        ax = axes[i]
        param_means = means_valid[:, i]
        true_val = clms_real_peak[i]
        
        ax.hist(param_means, bins=50, alpha=0.7, edgecolor="black", density=True)
        ax.axvline(true_val, color="red", linestyle="--", linewidth=2, label="True value")
        ax.axvline(
            np.mean(param_means),
            color="blue",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {np.mean(param_means):.6f}",
        )
        ax.axvline(
            np.mean(param_means) - np.std(param_means),
            color="blue",
            linestyle=":",
            linewidth=1,
            alpha=0.7,
        )
        ax.axvline(
            np.mean(param_means) + np.std(param_means),
            color="blue",
            linestyle=":",
            linewidth=1,
            alpha=0.7,
        )
        
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Density")
        ax.set_title(labels[i] if i < len(labels) else f"Parameter {i}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(
        f"{results['test_name']}\n{results['test_description']}\n"
        f"N={results['npulsars']} pulsars, {results['N_runs']} runs",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_parameter_comparison(results, output_path=None, show=True):
    """Plot estimated vs true parameters with error bars."""
    means = results["means"]
    stds = results["stds"]
    clms_real_peak = results["clms_real_peak"]
    l_max = results["l_max"]
    n_params = means.shape[1]
    
    # Filter out NaN values
    valid_mask = ~np.isnan(means).any(axis=1)
    means_valid = means[valid_mask]
    stds_valid = stds[valid_mask]
    
    labels = get_parameter_labels(l_max)
    
    # Compute mean and std across runs
    mean_estimates = np.nanmean(means_valid, axis=0)
    std_estimates = np.nanstd(means_valid, axis=0)
    mean_uncertainties = np.nanmean(stds_valid, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(n_params)
    width = 0.35
    
    # Plot true values
    bars1 = ax.bar(
        x_pos - width / 2,
        clms_real_peak,
        width,
        label="True Value",
        color="red",
        alpha=0.7,
    )
    
    # Plot estimated values with error bars
    bars2 = ax.bar(
        x_pos + width / 2,
        mean_estimates,
        width,
        yerr=std_estimates,
        label="Estimated (mean ± std)",
        color="blue",
        alpha=0.7,
        capsize=5,
    )
    
    # Add mean uncertainty as error bars
    ax.errorbar(
        x_pos + width / 2,
        mean_estimates,
        yerr=mean_uncertainties,
        fmt="none",
        color="green",
        capsize=3,
        label="Mean Uncertainty",
        alpha=0.5,
    )
    
    ax.set_xlabel("Parameter Index")
    ax.set_ylabel("Parameter Value")
    ax.set_title(
        f"{results['test_name']}: True vs Estimated Parameters\n"
        f"{results['test_description']}"
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[i] if i < len(labels) else f"P{i}" for i in range(n_params)], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_frequency_configuration(results, output_path=None, show=True):
    """Visualize frequency bin configuration."""
    fi_full = results["fi_full"]
    fi_recon = results["fi_recon"]
    injection_bins = results["injection_bins"]
    reconstruction_bins = results["reconstruction_bins"]
    Tspan_yrs = results["Tspan_yrs"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to Hz for display
    fi_full_hz = fi_full * constants.yr
    fi_recon_hz = fi_recon * constants.yr
    
    # Plot all frequency bins
    y_all = np.ones(len(fi_full_hz)) * 0.5
    ax.scatter(fi_full_hz, y_all, s=100, c="gray", alpha=0.3, label="All bins", zorder=1)
    
    # Highlight injection bins
    if len(injection_bins) > 0:
        fi_inj_hz = fi_full_hz[injection_bins]
        y_inj = np.ones(len(fi_inj_hz)) * 0.3
        ax.scatter(
            fi_inj_hz,
            y_inj,
            s=150,
            c="red",
            marker="s",
            label="Injection bins",
            zorder=3,
            edgecolors="black",
            linewidths=1,
        )
    
    # Highlight reconstruction bins
    if len(reconstruction_bins) > 0:
        fi_rec_hz = fi_full_hz[reconstruction_bins]
        y_rec = np.ones(len(fi_rec_hz)) * 0.7
        ax.scatter(
            fi_rec_hz,
            y_rec,
            s=150,
            c="blue",
            marker="^",
            label="Reconstruction bins",
            zorder=3,
            edgecolors="black",
            linewidths=1,
        )
    
    # Add frequency labels
    for i, f in enumerate(fi_full_hz):
        ax.text(f, 0.5, f"  {i}", fontsize=8, va="center", rotation=90)
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("")
    ax.set_title(
        f"{results['test_name']}: Frequency Bin Configuration\n"
        f"Injection: bins {injection_bins}, Reconstruction: bins {reconstruction_bins}"
    )
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add overlap warning if applicable
    overlap = np.intersect1d(injection_bins, reconstruction_bins)
    if len(overlap) > 0:
        ax.text(
            0.02,
            0.98,
            f"WARNING: {len(overlap)} overlapping bins: {overlap}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
        )
    else:
        ax.text(
            0.02,
            0.98,
            "✓ No overlap (clean separation)",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_baseline_windowed(baseline_path, windowed_path, output_path=None, show=True):
    """Compare baseline vs windowed results."""
    baseline = load_results(baseline_path)
    windowed = load_results(windowed_path)
    
    # Ensure same test
    if baseline["test_name"] != windowed["test_name"]:
        print("WARNING: Comparing different tests!")
    
    means_baseline = baseline["means"]
    means_windowed = windowed["means"]
    clms_real_peak = baseline["clms_real_peak"]
    l_max = baseline["l_max"]
    n_params = means_baseline.shape[1]
    
    # Filter NaN
    valid_b = ~np.isnan(means_baseline).any(axis=1)
    valid_w = ~np.isnan(means_windowed).any(axis=1)
    means_b = means_baseline[valid_b]
    means_w = means_windowed[valid_w]
    
    labels = get_parameter_labels(l_max)
    
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))
    if n_params == 1:
        axes = [axes]
    
    for i in range(n_params):
        ax = axes[i]
        
        # Histograms
        ax.hist(
            means_b[:, i],
            bins=30,
            alpha=0.5,
            label="Baseline",
            color="blue",
            density=True,
        )
        ax.hist(
            means_w[:, i],
            bins=30,
            alpha=0.5,
            label="Windowed",
            color="orange",
            density=True,
        )
        
        # True value
        ax.axvline(clms_real_peak[i], color="red", linestyle="--", linewidth=2, label="True")
        
        # Means
        mean_b = np.mean(means_b[:, i])
        mean_w = np.mean(means_w[:, i])
        ax.axvline(mean_b, color="blue", linestyle="-", linewidth=1.5, alpha=0.7)
        ax.axvline(mean_w, color="orange", linestyle="-", linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Density")
        ax.set_title(labels[i] if i < len(labels) else f"Parameter {i}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(
        f"Baseline vs Windowed Comparison: {baseline['test_name']}\n"
        f"Baseline: N={baseline['npulsars']}, Windowed: N={windowed['npulsars']}",
        fontsize=12,
    )
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_summary_statistics(results, output_path=None, show=True):
    """Create a summary statistics table plot."""
    means = results["means"]
    stds = results["stds"]
    clms_real_peak = results["clms_real_peak"]
    l_max = results["l_max"]
    n_params = means.shape[1]
    
    valid_mask = ~np.isnan(means).any(axis=1)
    means_valid = means[valid_mask]
    stds_valid = stds[valid_mask]
    
    labels = get_parameter_labels(l_max)
    
    # Compute statistics
    mean_est = np.nanmean(means_valid, axis=0)
    std_est = np.nanstd(means_valid, axis=0)
    mean_unc = np.nanmean(stds_valid, axis=0)
    
    # Compute bias and pull
    bias = mean_est - clms_real_peak
    pull = bias / std_est
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")
    
    # Create table
    table_data = []
    headers = ["Parameter", "True", "Est. Mean", "Est. Std", "Mean Unc.", "Bias", "Pull (σ)"]
    
    for i in range(n_params):
        table_data.append([
            labels[i] if i < len(labels) else f"P{i}",
            f"{clms_real_peak[i]:.6f}",
            f"{mean_est[i]:.6f}",
            f"{std_est[i]:.6f}",
            f"{mean_unc[i]:.6f}",
            f"{bias[i]:.6f}",
            f"{pull[i]:.2f}",
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code pulls
    for i in range(n_params):
        pull_val = pull[i]
        if abs(pull_val) < 1:
            color = "lightgreen"
        elif abs(pull_val) < 2:
            color = "yellow"
        else:
            color = "lightcoral"
        table[(i + 1, 6)].set_facecolor(color)
    
    plt.title(
        f"{results['test_name']}: Summary Statistics\n"
        f"{results['test_description']}\n"
        f"N={results['npulsars']} pulsars, {results['N_runs']} runs, "
        f"{np.sum(valid_mask)} valid runs",
        fontsize=12,
        pad=20,
    )
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PTA frequency leakage analysis results"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to NPZ results file (or 'all' to process all files in output_dir)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline NPZ file for comparison",
    )
    parser.add_argument(
        "--windowed",
        type=str,
        default=None,
        help="Path to windowed NPZ file for comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as input file)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots interactively",
    )
    parser.add_argument(
        "--plots",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "distributions", "comparison", "frequency", "summary", "baseline-windowed"],
        help="Which plots to generate",
    )
    
    args = parser.parse_args()
    
    show = not args.no_show
    
    # Handle "all" input
    if args.input_file == "all":
        base_path = Path(".")
        baseline_dir = base_path / "output_dir" / "baseline"
        windowed_dir = base_path / "output_dir" / "windowed"
        
        npz_files = []
        if baseline_dir.exists():
            npz_files.extend(list(baseline_dir.glob("*.npz")))
        if windowed_dir.exists():
            npz_files.extend(list(windowed_dir.glob("*.npz")))
        
        if len(npz_files) == 0:
            print("No NPZ files found in output_dir/")
            return
        
        print(f"Found {len(npz_files)} NPZ files. Processing...")
        for npz_file in npz_files:
            print(f"\nProcessing: {npz_file}")
            process_file(npz_file, args.output_dir, show, args.plots)
        
        return
    
    # Process single file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return
    
    process_file(input_path, args.output_dir, show, args.plots)
    
    # Handle comparison
    if args.baseline and args.windowed:
        baseline_path = Path(args.baseline)
        windowed_path = Path(args.windowed)
        
        if not baseline_path.exists() or not windowed_path.exists():
            print("Error: Baseline or windowed file not found")
            return
        
        output_path = None
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "baseline_vs_windowed.png"
        
        plot_comparison_baseline_windowed(
            str(baseline_path), str(windowed_path), str(output_path) if output_path else None, show
        )


def process_file(input_path, output_dir, show, plots):
    """Process a single NPZ file and generate requested plots."""
    results = load_results(str(input_path))
    
    # Determine output directory
    if output_dir:
        plot_dir = Path(output_dir)
    else:
        plot_dir = input_path.parent / "plots"
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = input_path.stem
    
    if "all" in plots or "distributions" in plots:
        output_path = plot_dir / f"{base_name}_distributions.png"
        plot_parameter_distributions(results, str(output_path), show)
    
    if "all" in plots or "comparison" in plots:
        output_path = plot_dir / f"{base_name}_comparison.png"
        plot_parameter_comparison(results, str(output_path), show)
    
    if "all" in plots or "frequency" in plots:
        output_path = plot_dir / f"{base_name}_frequency.png"
        plot_frequency_configuration(results, str(output_path), show)
    
    if "all" in plots or "summary" in plots:
        output_path = plot_dir / f"{base_name}_summary.png"
        plot_summary_statistics(results, str(output_path), show)


if __name__ == "__main__":
    main()
