"""
Visualize perturbed vs unperturbed datasets and final flame profile.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_species(data, species_list):
    """Load requested species arrays from NPZ data."""
    species = []
    for name in species_list:
        key = f"Y_{name}"
        if key in data.files:
            species.append((name, data[key]))
    return species


def plot_scatter_species_vs_temperature(
    perturbed_npz, final_flame_npz, output_dir="plots"
):
    """Scatter plot of major species vs temperature with perturbed data and final flame profile line."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    pert = np.load(perturbed_npz)
    final = np.load(final_flame_npz)

    T_pe = pert["T"]
    T_final = final["temperature"]

    major_species = ["H2", "O2", "H2O", "H", "O", "OH"]

    pe_species = load_species(pert, major_species)
    final_species = load_species(final, major_species)

    # Create plot grid
    n_plots = len(pe_species)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, ((name, y_pe), (_, y_final)) in enumerate(zip(pe_species, final_species)):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.scatter(T_pe, y_pe, s=6, alpha=0.35, label="Perturbed", color="tab:orange")
        ax.plot(
            T_final, y_final, color="tab:blue", linewidth=2, label="Final Flame Profile"
        )
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel(f"Y_{name}")
        ax.set_title(f"Species {name}")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right")

    # Hide unused axes
    for idx in range(n_plots, n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.axis("off")

    plt.tight_layout()
    output_file = output_dir / "perturbation_species_vs_temperature.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved scatter plot to: {output_file}")
    plt.close()


def plot_final_flame_profile(final_flame_npz, output_dir="plots"):
    """Plot final flame profile using the same layout as plot_flame.py."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    data = np.load(final_flame_npz)
    grid = data["grid"]
    temperature = data["temperature"]
    heat_release_rate = data["heat_release_rate"]
    velocity = data["velocity"]
    flame_speed = data["flame_speed"]

    major_species = ["H2", "O2", "H2O", "H", "O", "OH"]
    species = load_species(data, major_species)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        f"H2/Air Freely Propagating Flame - Final Solution\n"
        f"Flame Speed: {flame_speed:.3f} m/s",
        fontsize=14,
    )

    ax = axes[0, 0]
    ax.plot(grid * 100, temperature, "r-", linewidth=2.5, label="Temperature")
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Temperature Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(grid * 100, velocity, "g-", linewidth=2, label="Velocity")
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(
        grid * 100, heat_release_rate / 1e6, "b-", linewidth=2, label="Heat Release"
    )
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Heat Release Rate [MW/m³]")
    ax.set_title("Heat Release Rate Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    for name, y in species:
        if np.max(y) > 1e-6:
            ax.plot(grid * 100, y, label=name, linewidth=1.5)
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Mass Fraction")
    ax.set_title("Major Species Mass Fractions")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")

    ax = axes[2, 0]
    for name, y in species:
        if np.max(y) > 1e-6:
            ax.plot(temperature, y, label=name, linewidth=1.5)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Mass Fraction")
    ax.set_title("Major Species vs Temperature")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")

    ax = axes[2, 1]
    dt_dx = np.gradient(temperature, grid)
    peak_idx = np.argmax(np.abs(dt_dx))
    peak_pos = grid[peak_idx] * 100
    ax.plot(grid * 100, temperature, "r-", linewidth=2, label="Temperature")
    ax.axvline(
        peak_pos,
        color="k",
        linestyle="--",
        alpha=0.5,
        label=f"Peak T gradient @ {peak_pos:.2f} cm",
    )
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Reaction Zone Detail")
    ax.set_xlim(peak_pos - 0.5, peak_pos + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_file = output_dir / "flame_final_solution.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved flame profile plot to: {output_file}")
    plt.close()


def main():
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python visualize_perturbation.py <perturbed_npz> <final_flame_npz>"
        )
        print("\nExample:")
        print(
            "  python visualize_perturbation.py collected_data.npz output_data/final_flame.npz"
        )
        return

    perturbed_npz = Path(sys.argv[1])
    final_flame_npz = Path(sys.argv[2])

    plot_scatter_species_vs_temperature(perturbed_npz, final_flame_npz)
    plot_final_flame_profile(final_flame_npz)


if __name__ == "__main__":
    main()
