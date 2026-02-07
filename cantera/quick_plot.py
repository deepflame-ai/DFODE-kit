#!/usr/bin/env python3
"""Quick plot script for a single NPZ flame file."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_flame_file(npz_file):
    """Plot flame results from a single NPZ file."""

    # Load data
    data = np.load(npz_file)
    grid = data["grid"]
    temperature = data["temperature"]
    heat_release_rate = data["heat_release_rate"]

    # Get species data
    species_keys = [k for k in data.keys() if k.startswith("Y_")]
    species_names = [k[2:] for k in species_keys]

    # Major species for H2/air
    major_species = ["H2", "O2", "H2O", "H", "O", "OH"]
    major_species = [s for s in major_species if s in species_names]

    # Get metadata
    if "iteration" in data:
        title_suffix = f" - Iteration {data['iteration']}"
    else:
        title_suffix = " - Final Solution"
        if "equivalence_ratio" in data:
            title_suffix += f" (phi={data['equivalence_ratio']:.1f})"

    flame_speed = data["flame_speed"]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"H2/Air Flame{title_suffix}\nFlame Speed: {flame_speed:.3f} m/s", fontsize=14
    )

    # Plot 1: Temperature profile
    ax = axes[0, 0]
    ax.plot(grid * 100, temperature, "r-", linewidth=2.5, label="Temperature")
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Temperature Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Heat release rate
    ax = axes[0, 1]
    ax.plot(
        grid * 100, heat_release_rate / 1e6, "b-", linewidth=2, label="Heat Release"
    )
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Heat Release Rate [MW/m³]")
    ax.set_title("Heat Release Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Major species in physical space
    ax = axes[1, 0]
    for species in major_species:
        y_data = data[f"Y_{species}"]
        if np.max(y_data) > 1e-4:
            ax.plot(grid * 100, y_data, linewidth=1.5, label=species)
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Mass Fraction")
    ax.set_title("Major Species (Physical Space)")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")

    # Plot 4: Major species vs temperature
    ax = axes[1, 1]
    for species in major_species:
        y_data = data[f"Y_{species}"]
        if np.max(y_data) > 1e-4:
            ax.plot(temperature, y_data, linewidth=1.5, label=species)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Mass Fraction")
    ax.set_title("Major Species vs Temperature")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")

    plt.tight_layout()

    # Save figure
    output_file = Path(npz_file).stem + ".png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_file}")

    plt.show()

    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_plot.py <npz_file>")
        print("\nExample:")
        print("  python quick_plot.py output_data/final_flame.npz")
        print("  python quick_plot.py output_data/iteration_00006.npz")
        sys.exit(1)

    npz_file = Path(sys.argv[1])
    if not npz_file.exists():
        print(f"Error: File '{npz_file}' not found.")
        sys.exit(1)

    plot_flame_file(npz_file)
