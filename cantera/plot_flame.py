from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_flame_results(npz_file, output_dir="plots"):
    """Plot flame results from NPZ file.

    Args:
        npz_file: Path to NPZ file containing flame data
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    data = np.load(npz_file)
    grid = data["grid"]
    temperature = data["temperature"]
    heat_release_rate = data["heat_release_rate"]
    flame_speed = data["flame_speed"]
    iteration = data["iteration"]

    # Get species data (mass fractions Y and mole fractions X)
    species_keys = [k for k in data.keys() if k.startswith("Y_")]
    species_names = [k[2:] for k in species_keys]

    # Major species for H2/air
    major_species = ["H2", "O2", "H2O", "H", "O", "OH"]
    major_species = [s for s in major_species if s in species_names]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"H2/Air Freely Propagating Flame - Iteration {iteration}\nFlame Speed: {flame_speed:.3f} m/s",
        fontsize=14,
    )

    # Plot 1: Temperature profile in physical space
    ax = axes[0, 0]
    ax.plot(grid * 100, temperature, "r-", linewidth=2, label="Temperature")
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Temperature Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Heat release rate in physical space
    ax = axes[0, 1]
    ax.plot(
        grid * 100, heat_release_rate / 1e6, "b-", linewidth=2, label="Heat Release"
    )
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Heat Release Rate [MW/m³]")
    ax.set_title("Heat Release Rate Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Major species mass fractions in physical space
    ax = axes[1, 0]
    for species in major_species:
        y_data = data[f"Y_{species}"]
        if np.max(y_data) > 1e-4:  # Only plot species with significant concentration
            ax.plot(grid * 100, y_data, linewidth=1.5, label=species)
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Mass Fraction")
    ax.set_title("Major Species Mass Fractions")
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
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f"flame_iteration_{iteration:05d}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_file}")

    plt.close()

    return output_file


def plot_final_solution(npz_file, output_dir="plots"):
    """Plot final flame solution with additional details.

    Args:
        npz_file: Path to final NPZ file
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    data = np.load(npz_file)
    grid = data["grid"]
    temperature = data["temperature"]
    velocity = data["velocity"]
    heat_release_rate = data["heat_release_rate"]
    flame_speed = data["flame_speed"]
    equivalence_ratio = data["equivalence_ratio"]

    # Get species data
    species_keys = [k for k in data.keys() if k.startswith("Y_")]
    species_names = [k[2:] for k in species_keys]

    major_species = ["H2", "O2", "H2O", "H", "O", "OH"]
    major_species = [s for s in major_species if s in species_names]

    # Create comprehensive figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        f"H2/Air Freely Propagating Flame - Final Solution\n"
        f"Equivalence Ratio: {equivalence_ratio:.1f} | Flame Speed: {flame_speed:.3f} m/s",
        fontsize=14,
    )

    # Plot 1: Temperature profile
    ax = axes[0, 0]
    ax.plot(grid * 100, temperature, "r-", linewidth=2.5, label="Temperature")
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Temperature Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Velocity profile
    ax = axes[0, 1]
    ax.plot(grid * 100, velocity, "g-", linewidth=2, label="Velocity")
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Heat release rate
    ax = axes[1, 0]
    ax.plot(
        grid * 100, heat_release_rate / 1e6, "b-", linewidth=2, label="Heat Release"
    )
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Heat Release Rate [MW/m³]")
    ax.set_title("Heat Release Rate Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Major species in physical space
    ax = axes[1, 1]
    for species in major_species:
        y_data = data[f"Y_{species}"]
        if np.max(y_data) > 1e-4:
            ax.plot(grid * 100, y_data, linewidth=1.5, label=species)
    ax.set_xlabel("Position [cm]")
    ax.set_ylabel("Mass Fraction")
    ax.set_title("Major Species Mass Fractions")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")

    # Plot 5: Species vs temperature
    ax = axes[2, 0]
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

    # Plot 6: Zoom on reaction zone
    ax = axes[2, 1]
    # Find reaction zone (where dT/dx is maximum)
    dt_dx = np.gradient(temperature, grid)
    peak_idx = np.argmax(np.abs(dt_dx))
    peak_T = temperature[peak_idx]
    peak_pos = grid[peak_idx] * 100

    # Plot zoomed temperature and species
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

    # Save figure
    output_file = output_dir / "flame_final_solution.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved final solution plot to: {output_file}")

    plt.close()

    return output_file


def main():
    output_dir = Path("output_data")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Check if output data exists
    if not output_dir.exists():
        print(
            f"Error: Output directory '{output_dir}' not found. Run flame simulation first."
        )
        return

    # Plot all iterations
    iteration_files = sorted(output_dir.glob("iteration_*.npz"))
    if not iteration_files:
        print("No iteration files found.")
        return

    print(f"Found {len(iteration_files)} iteration files.")

    # Plot final solution
    final_file = output_dir / "final_flame.npz"
    if final_file.exists():
        print(f"Plotting final solution...")
        plot_final_solution(final_file, plots_dir)

    # Plot each iteration
    for npz_file in iteration_files:
        print(f"Plotting {npz_file.name}...")
        plot_flame_results(npz_file, plots_dir)

    print(f"\nAll plots saved to: {plots_dir.absolute()}")


if __name__ == "__main__":
    main()
