"""
Collect and Perturb Augmented Flame Data

Collects all augmented data into a single large numpy array with random perturbations.

Perturbation equations:
    T' = T + 100 * X * pert_t
    p' = p + (max(p) - min(p)) * 0.15 * X * pert_p
    Y_α' = Y_α^(1 + 0.15 * X) * pert_y

where X ~ Uniform(-1, 1)
"""

from pathlib import Path
import numpy as np
import cantera as ct


def collect_augmented_data(input_files, mech_file):
    """
    Collect all augmented data into single large arrays.

    Args:
        input_files: List of NPZ files to collect
        mech_file: Path to mechanism file for species order

    Returns:
        Dictionary with combined T, p, Y arrays
    """
    # Load mechanism to get species order
    gas = ct.Solution(str(mech_file))
    species_order = gas.species_names
    n_species = len(species_order)

    print(f"Species order: {species_order}")

    # Collect all data
    all_T = []
    all_p = []
    all_Y = []
    all_orig_T = []

    for npz_file in input_files:
        print(f"Loading: {npz_file.name}")
        data = np.load(npz_file)

        T = data["temperature"]
        p = data["pressure"]

        # Get Y in correct species order
        Y = np.stack([data[f"Y_{species}"] for species in species_order], axis=1)

        all_T.append(T)
        all_p.append(p)
        all_Y.append(Y)
        all_orig_T.append(T)

    # Concatenate
    all_T = np.concatenate(all_T)
    all_p = np.concatenate(all_p)
    all_orig_T = np.concatenate(all_orig_T)
    all_Y = np.concatenate(all_Y, axis=0)

    n_points = len(all_T)
    print(f"\nCollected data:")
    print(f"  Total points: {n_points}")
    print(f"  T shape: {all_T.shape}")
    print(f"  p shape: {all_p.shape}")
    print(f"  Y shape: {all_Y.shape}")

    return {
        "T": all_T,
        "p": all_p,
        "Y": all_Y,
        "species_order": species_order,
        "n_species": n_species,
        "all_orig_T": all_orig_T,
    }


def apply_random_perturbation(data_dict, pert_fraction=0.15):
    """
    Apply random perturbation to T, p, Y based on original T.

    Perturbation equations:
        T' = T + 100 * X * pert_t
        p' = p + (max(p) - min(p)) * 0.15 * X * pert_p
        Y_α' = Y_α^(1 + 0.15 * X) * pert_y

    where X ~ Uniform(-1, 1)

    Args:
        data_dict: Dictionary with T, p, Y, all_orig_T
        pert_fraction: Base perturbation fraction (0.15)

    Returns:
        Dictionary with perturbed T', p', Y'
    """
    T = data_dict["T"]
    p = data_dict["p"]
    Y = data_dict["Y"]
    all_orig_T = data_dict["all_orig_T"]
    n_points = len(T)

    print(f"\nApplying perturbation:")
    print(f"  Perturbation fraction: {pert_fraction}")

    # Calculate perturbation scaling based on original T
    # Higher T gets larger perturbation (more unstable near equilibrium)
    T_min = np.min(all_orig_T)
    T_max = np.max(all_orig_T)
    T_normalized = (all_orig_T - T_min) / (T_max - T_min + 1e-10)

    # Scale perturbation: base * (0.5 + 1.5 * T_normalized)
    pert_scale_T = pert_fraction * (0.5 + 1.5 * T_normalized)
    pert_scale_p = pert_fraction
    pert_scale_Y = pert_fraction * (1 + T_normalized)

    print(
        f"  T perturbation scale: {np.mean(pert_scale_T):.4f} (range: {pert_scale_T.min():.4f} - {pert_scale_T.max():.4f})"
    )
    print(f"  p perturbation scale: {pert_scale_p:.4f}")
    print(
        f"  Y perturbation scale: {np.mean(pert_scale_Y):.4f} (range: {pert_scale_Y.min():.4f} - {pert_scale_Y.max():.4f})"
    )

    # Generate random perturbations
    X = np.random.uniform(-1, 1, n_points)
    pert_t = 100 * X
    pert_p = (np.max(p) - np.min(p)) * X
    pert_Y = np.random.uniform(-1, 1, Y.shape)

    # Apply perturbations
    T_pert = T + pert_scale_T * pert_t
    p_pert = p + pert_scale_p * pert_p

    # Apply Y perturbation with power scaling
    scale_factors = 1 + pert_scale_Y[:, np.newaxis] * X[:, np.newaxis]
    Y_pert_power = Y**scale_factors

    # Clip to physical bounds
    Y_pert_power = np.clip(Y_pert_power, 0, 1)
    T_pert = np.clip(T_pert, T_min - 50, T_max + 100)

    # Normalize Y to sum to 1
    Y_sum = np.sum(Y_pert_power, axis=1)
    mask = Y_sum > 1e-10
    Y_pert_power[mask] = Y_pert_power[mask] / Y_sum[mask, np.newaxis]

    print(f"\nPerturbed data stats:")
    print(f"  T range: {np.min(T_pert):.1f} - {np.max(T_pert):.1f} K")
    print(f"  p range: {np.min(p_pert):.1f} - {np.max(p_pert):.1f} Pa")
    print(
        f"  Y sum: min={np.sum(Y_pert_power, axis=1).min():.6f}, max={np.sum(Y_pert_power, axis=1).max():.6f}"
    )

    return {
        "T": T_pert,
        "p": p_pert,
        "Y": Y_pert_power,
    }


def collect_and_perturb(
    input_files,
    mech_file,
    n_points=10000,
    pert_fraction=0.15,
    rounds=1,
    output_file="collected_data.npz",
):
    """
    Collect augmented data, apply perturbation, and save.

    Args:
        input_files: List of NPZ files to collect
        mech_file: Path to mechanism file for species order
        n_points: Total number of points to randomly sample
        pert_fraction: Base perturbation fraction
        rounds: Number of perturbation rounds
        output_file: Output NPZ filename
    """
    input_files = list(input_files)
    if not input_files:
        print(f"Error: No input files found")
        return None

    input_files.sort()
    print(f"Found {len(input_files)} augmented files")

    # Collect all augmented data
    collected = collect_augmented_data(input_files, mech_file)

    # Apply multiple rounds of perturbation
    perturbed_T = []
    perturbed_p = []
    perturbed_Y = []

    for round_idx in range(rounds):
        print(f"\n{'=' * 60}")
        print(f"Perturbation round {round_idx + 1}/{rounds}")
        print("=" * 60)

        perturbed = apply_random_perturbation(collected, pert_fraction=pert_fraction)

        perturbed_T.append(perturbed["T"])
        perturbed_p.append(perturbed["p"])
        perturbed_Y.append(perturbed["Y"])

    # Concatenate all rounds
    final_T = np.concatenate(perturbed_T)
    final_p = np.concatenate(perturbed_p)
    final_Y = np.concatenate(perturbed_Y, axis=0)

    total_points = len(final_T)
    print(f"\nTotal points after {rounds} rounds: {total_points}")

    # Randomly sample n_points
    if total_points > n_points:
        indices = np.random.choice(total_points, n_points, replace=False)
        final_T = final_T[indices]
        final_p = final_p[indices]
        final_Y = final_Y[indices]
        print(f"Randomly sampled {n_points} points from {total_points}")
    else:
        print(f"Using all {total_points} points (requested {n_points})")

    # Save to NPZ
    species_order = collected["species_order"]
    output_path = Path(output_file)

    # Create dictionary with proper dimension names
    save_dict = {
        "T": final_T,
        "p": final_p,
    }

    # Add each species as separate array
    for i, species in enumerate(species_order):
        save_dict[f"Y_{species}"] = final_Y[:, i]

    # Add metadata
    save_dict["n_points"] = len(final_T)
    save_dict["n_species"] = len(species_order)
    save_dict["species_order"] = np.array(species_order, dtype="S20")
    save_dict["perturbation_rounds"] = rounds
    save_dict["perturbation_fraction"] = pert_fraction

    np.savez(output_path, **save_dict)
    print(f"\nSaved collected data to: {output_path.absolute()}")
    print(f"  Shape: T={final_T.shape}, p={final_p.shape}, Y={final_Y.shape}")

    # Save unperturbed collected data for comparison
    unperturbed_path = output_path.with_name("collected_data_unperturbed.npz")
    unperturbed_dict = {
        "T": collected["T"],
        "p": collected["p"],
        "n_points": len(collected["T"]),
        "n_species": len(species_order),
        "species_order": np.array(species_order, dtype="S20"),
    }
    for i, species in enumerate(species_order):
        unperturbed_dict[f"Y_{species}"] = collected["Y"][:, i]
    np.savez(unperturbed_path, **unperturbed_dict)
    print(f"Saved unperturbed data to: {unperturbed_path.absolute()}")

    return output_path


def main():
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python collect_augmented.py <augmented_dir> <mech_file> [n_points] [pert_fraction] [rounds]"
        )
        print("\nArguments:")
        print("  augmented_dir    - Directory containing augmented NPZ files")
        print("  mech_file         - Path to mechanism YAML file")
        print("  n_points          - Total number of points to sample (default: 10000)")
        print("  pert_fraction    - Perturbation fraction (default: 0.15)")
        print("  rounds            - Number of perturbation rounds (default: 1)")
        print("\nExamples:")
        print(
            "  python collect_augmented.py augmented_data ../mechanisms/Burke2012_s9r23.yaml 50000"
        )
        print(
            "  python collect_augmented.py augmented_data ../mechanisms/Burke2012_s9r23.yaml 10000 0.2 3"
        )
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)

    # Find all augmented NPZ files in the directory
    input_files = list(input_dir.glob("*_augmented.npz"))

    if not input_files:
        print(f"Error: No augmented files found in {input_dir}")
        print(f"  Looking for files matching: *_augmented.npz")
        print(f"  Found files in directory: {list(input_dir.glob('*.npz'))}")
        sys.exit(1)

    input_files.sort()
    mech_file = Path(sys.argv[2])
    if not mech_file.exists():
        print(f"Error: Mechanism file not found: {mech_file}")
        sys.exit(1)

    # Parse optional arguments
    n_points = 10000
    pert_fraction = 0.15
    rounds = 1

    if len(sys.argv) >= 4:
        try:
            n_points = int(sys.argv[3])
        except ValueError:
            print(f"Error: n_points must be an integer, got '{sys.argv[3]}'")
            sys.exit(1)
    if len(sys.argv) >= 5:
        try:
            pert_fraction = float(sys.argv[4])
        except ValueError:
            print(f"Error: pert_fraction must be a number, got '{sys.argv[4]}'")
            sys.exit(1)
    if len(sys.argv) >= 6:
        try:
            rounds = int(sys.argv[5])
        except ValueError:
            print(f"Error: rounds must be an integer, got '{sys.argv[5]}'")
            sys.exit(1)

    print(f"Configuration:")
    print(f"  Input directory: {input_dir}")
    print(f"  Mechanism: {mech_file}")
    print(f"  Target points: {n_points}")
    print(f"  Perturbation: {pert_fraction}")
    print(f"  Rounds: {rounds}")

    # Collect and perturb
    collect_and_perturb(
        input_files=input_files,
        mech_file=mech_file,
        n_points=n_points,
        pert_fraction=pert_fraction,
        rounds=rounds,
        output_file="collected_data.npz",
    )


if __name__ == "__main__":
    main()
