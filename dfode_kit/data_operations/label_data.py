import time
import numpy as np
import cantera as ct
import os

def advance_reactor(gas, state, reactor, reactor_net, min_time_step, max_time_step):
    """Advance the reactor simulation for a given state."""
    gas.TPY = state[0], state[1], state[2:2 + gas.n_species]
    
    time_step = np.random.uniform(min_time_step, max_time_step)
    reactor.syncState()
    reactor_net.reinitialize()
    reactor_net.advance(time_step)
    
    return time_step, gas

def main(original_data_path, mech_path, min_time_step, max_time_step, save_file_path):
    # Check if the original data file exists
    if not os.path.isfile(original_data_path):
        raise FileNotFoundError(f"Original data file not found: {original_data_path}")
    
    # Check if the mechanism file exists
    if not os.path.isfile(mech_path):
        raise FileNotFoundError(f"Chemical mechanism file not found: {mech_path}")

    # Validate time step values
    if min_time_step <= 0 or max_time_step <= 0:
        raise ValueError("Time steps must be positive values.")
    if min_time_step >= max_time_step:
        raise ValueError("Minimum time step must be less than the maximum time step.")

    # Load the chemical mechanism
    gas = ct.Solution(mech_path)
    n_species = gas.n_species

    # Load the dataset containing initial states for the reactor
    test_data = np.load(original_data_path)
    print(f"Loaded dataset from: {original_data_path}")
    print(f"{test_data.shape=}")

    # Prepare an array to store labeled data
    labeled_data = np.empty((test_data.shape[0], 2 * n_species + 5))

    # Initialize Cantera reactor
    reactor = ct.Reactor(gas, name='Reactor1', energy='off')
    reactor_net = ct.ReactorNet([reactor])
    reactor_net.rtol, reactor_net.atol = 1e-6, 1e-10

    # Start timing the simulation
    start_time = time.time()

    # Process each state in the dataset
    for i, state in enumerate(test_data):
        time_step, gas = advance_reactor(gas, state, reactor, reactor_net, min_time_step, max_time_step)
        labeled_data[i, :2 + n_species] = state[:2 + n_species]
        labeled_data[i, 2 + n_species] = time_step
        labeled_data[i, 2 + n_species + 1:] = np.array([gas.T, gas.P] + list(gas.Y))

    # End timing of the simulation
    end_time = time.time()
    total_time = end_time - start_time

    # Save the labeled dataset to a file
    np.save(save_file_path, labeled_data)

    # Print the total time used and the path of the saved data
    print(f"Total time used: {total_time:.2f} seconds")
    print(f"Saved dataset to: {save_file_path}")