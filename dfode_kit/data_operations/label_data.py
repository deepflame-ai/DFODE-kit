import time
import numpy as np
import cantera as ct

from .h5_kit import advance_reactor

def label_npy(
    mech_path, 
    time_step,
    source_path,
):
    # Load the chemical mechanism
    gas = ct.Solution(mech_path)
    n_species = gas.n_species

    # Load the dataset containing initial states for the reactor
    test_data = np.load(source_path)
    print(f"Loaded dataset from: {source_path}")
    print(f"{test_data.shape=}")

    # Prepare an array to store labeled data
    labeled_data = np.empty((test_data.shape[0], 2 * n_species + 4))

    # Initialize Cantera reactor
    reactor = ct.Reactor(gas, name='Reactor1', energy='off')
    reactor_net = ct.ReactorNet([reactor])
    reactor_net.rtol, reactor_net.atol = 1e-6, 1e-10

    # Start timing the simulation
    start_time = time.time()

    # Process each state in the dataset
    for i, state in enumerate(test_data):
        gas = advance_reactor(gas, state, reactor, reactor_net, time_step)
        labeled_data[i, :2 + n_species] = state[:2 + n_species]
        labeled_data[i, 2 + n_species:] = np.array([gas.T, gas.P] + list(gas.Y))

    # End timing of the simulation
    end_time = time.time()
    total_time = end_time - start_time

    # Print the total time used and the path of the saved data
    print(f"Total time used: {total_time:.2f} seconds")
    
    return labeled_data