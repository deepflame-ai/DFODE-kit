from pathlib import Path
import numpy as np
import cantera as ct


def save_iteration_data(f, iteration, output_dir):
    """Save grid data for current iteration to NPZ file."""
    output_file = output_dir / f"iteration_{iteration:05d}.npz"

    data_dict = {
        "grid": f.grid,
        "velocity": f.velocity,
        "temperature": f.T,
        "density": f.density,
        "pressure": f.P,
    }

    for i, species in enumerate(f.gas.species_names):
        data_dict[f"Y_{species}"] = f.Y[i, :]
        data_dict[f"X_{species}"] = f.X[i, :]

    data_dict["heat_release_rate"] = f.heat_release_rate
    data_dict["iteration"] = iteration
    data_dict["flame_speed"] = f.velocity[0]
    data_dict["transport_model"] = f.transport_model

    np.savez(output_file, **data_dict)


def set_h2_air_mixture(gas, equivalence_ratio=1.0):
    """Set H2/air mixture using Cantera's equivalence_ratio method.

    Args:
        gas: Cantera Solution object
        equivalence_ratio: Equivalence ratio (phi). phi=1 is stoichiometric.
    """
    air = "O2:0.21,N2:0.79"
    fuel = "H2:1"
    gas.set_equivalence_ratio(phi=equivalence_ratio, fuel=fuel, oxidizer=air)


def solve_with_manual_refinements(
    f, loglevel=0, max_refinements=50, output_dir=Path("output")
):
    """Solve flame and save data each time grid is refined manually."""
    output_dir.mkdir(exist_ok=True)

    iteration = 0
    previous_grid_size = len(f.grid)

    # Initial state
    save_iteration_data(f, iteration, output_dir)
    print(
        f"Iteration {iteration}: grid points = {previous_grid_size}, flame speed = {f.velocity[0]:.6f} m/s"
    )
    iteration += 1

    # Initial solve without refinement
    f.solve(loglevel=loglevel, auto=False, refine_grid=False)
    current_grid_size = len(f.grid)
    save_iteration_data(f, iteration, output_dir)
    print(
        f"Iteration {iteration}: grid points = {current_grid_size}, flame speed = {f.velocity[0]:.6f} m/s"
    )
    previous_grid_size = current_grid_size
    iteration += 1

    # Manually refine and solve
    no_refinement_count = 0

    while iteration <= max_refinements + 2:
        try:
            # Manual refinement
            f.refine()
            current_grid_size = len(f.grid)

            if current_grid_size == previous_grid_size:
                no_refinement_count += 1
                if no_refinement_count >= 3:
                    print(f"No more grid refinements needed")
                    break
            else:
                no_refinement_count = 0
                save_iteration_data(f, iteration, output_dir)
                print(
                    f"Iteration {iteration}: grid points = {current_grid_size}, flame speed = {f.velocity[0]:.6f} m/s"
                )
                previous_grid_size = current_grid_size
                iteration += 1

            # Solve on refined grid
            f.solve(loglevel=loglevel, auto=False, refine_grid=False)

        except Exception as e:
            print(f"Error at iteration {iteration}: {e}")
            break

    # Final solve with auto=True to ensure convergence
    f.solve(loglevel=loglevel, auto=True)
    current_grid_size = len(f.grid)
    if current_grid_size != previous_grid_size:
        save_iteration_data(f, iteration, output_dir)
        print(
            f"Iteration {iteration}: grid points = {current_grid_size}, flame speed = {f.velocity[0]:.6f} m/s"
        )
    else:
        print(
            f"Final state: grid points = {current_grid_size}, flame speed = {f.velocity[0]:.6f} m/s"
        )

    return f, iteration


def main():
    p = ct.one_atm
    Tin = 300.0
    equivalence_ratio = 1.0
    width = 0.03
    loglevel = 0

    output_dir = Path("output_data")
    output_dir.mkdir(exist_ok=True)

    mech_file = Path(__file__).parent.parent / "mechanisms" / "Burke2012_s9r23.yaml"
    print(f"Loading mechanism from: {mech_file}")
    print(f"Equivalence ratio: {equivalence_ratio}")

    gas = ct.Solution(str(mech_file))
    gas.TP = Tin, p
    set_h2_air_mixture(gas, equivalence_ratio)

    f = ct.FreeFlame(gas, width=width)
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)

    f.transport_model = "mixture-averaged"
    f.flux_gradient_basis = "mass"
    f.soret_enabled = False

    f, final_iteration = solve_with_manual_refinements(
        f, loglevel=loglevel, max_refinements=50, output_dir=output_dir
    )

    print(f"\nFinal flame speed = {f.velocity[0]:.7f} m/s")
    print(f"Final grid points = {len(f.grid)}")
    print(f"Temperature range: {f.T.min():.1f} - {f.T.max():.1f} K")

    final_output = output_dir / "final_flame.npz"
    data_dict = {
        "grid": f.grid,
        "velocity": f.velocity,
        "temperature": f.T,
        "density": f.density,
        "pressure": f.P,
    }

    for i, species in enumerate(f.gas.species_names):
        data_dict[f"Y_{species}"] = f.Y[i, :]
        data_dict[f"X_{species}"] = f.X[i, :]

    data_dict["heat_release_rate"] = f.heat_release_rate
    data_dict["flame_speed"] = f.velocity[0]
    data_dict["equivalence_ratio"] = equivalence_ratio

    np.savez(final_output, **data_dict)
    print(f"\nFinal solution saved to: {final_output}")

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
