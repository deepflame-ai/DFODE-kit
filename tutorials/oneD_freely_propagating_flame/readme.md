## DEMO:

# All commands should be used in the `tutorials/oneD_freely_propagating_flame` bash.

## 1.prepare
#### Low-dimensional manifold sampling
# A key challenge in preparing training data is achieving sufficient coverage of the relevant thermochemical composition space, 
# which is often prohibitively high-dimensional when detailed chemistry involves tens to hundreds of species. 
# To address this, DFODE-kit adopts a low-dimensional manifold sampling strategy, where thermochemical states are extracted from 
# canonical flame configurations that retain the essential topology of high-dimensional turbulent flames. This approach ensures 
# both computational efficiency and physical representativeness of the training datasets.
# In this tutorial, we will demonstrate how to use DFODE-kit to sample a low-dimensional manifold of thermochemical states from 
# a one-dimensional laminar freely propagating flame simulated with DeepFlame. Users can use `dfode_kit_init.ipynb` to in the 
# `tutorials/oneD_freely_propagating_flame` directory initialize the simulation and update the dictionary files for the simulation.

dfode_kit_init.ipynb

## 2.Allrun

# Note that at the point, the simulation is not yet started. The user would need to ensure a working version of DeepFlame is 
# available and run the `Allrun` script from command line to start the simulation.

./Allrun

# After the simulation is completed, we proceed to use DFODE-kit to gather and manage the thermochemical data.
# Note: Cases in DFODE-kit is designed for DeepFlame-v1.6 (or beyond)
# Note: If the version of DeepFlame is below v1.5, please check the system/fvSolution of cases belonging to dfLowMachFoam, 
# the h should be changed by ha.
    # "(U|h|k|epsilon)"   //"(U|ha|k|epsilon)"
    # {
    #     solver          PBiCGStab;
    #     preconditioner  DILU;
    #     tolerance       1e-6;
    #     relTol          0.1;
    # }

    # "(U|h|k|epsilon)Final"   //"(U|ha|k|epsilon)"
    # {
    #     $U;
    #     relTol          0;
    # }

## 3. sample and augment

# In this step, we use DFODE-kit to sample thermochemical states from the one-dimensional freely propagating 
# flame simulation. The --mech option specifies the mechanism file, while the --case option points to the 
# simulation case directory. The sampled data will be saved in the specified HDF5 file.

dfode-kit sample --mech ../../mechanisms/Burke2012_s9r23.yaml \
    --case . \
    --save ./tutorial_data.h5 --include_mesh

# This command augments the sampled data and filter them by element ratio.
# The --h5_file option specifies the input HDF5 file containing the original sampled data,
# while the --output_file option determines where the augmented dataset will be saved.
# The --dataset_num option specifies the number of synthetic data points to generate.

dfode-kit augment --mech ../../mechanisms/Burke2012_s9r23.yaml \
    --h5_file ./tutorial_data.h5 \
    --output_file ./data \
    --dataset_num 20000


## 4.train

# In this step, we use DFODE-kit to label the synthetic thermochemical data.
# The --mech option specifies the mechanism file that defines the chemical kinetics
# relevant to the labeling process. The --time option sets the time point at which
# the labeling occurs, in this case, 1 microsecond (1e-06). 
# The --source option points to the input NumPy file containing the synthetic data
# generated in the previous step. Finally, the --save option specifies the path
# where the labeled dataset will be saved as a NumPy file.

dfode-kit label --mech ../../mechanisms/Burke2012_s9r23.yaml \
    --time 1e-06 \
    --source ./data.npy \
    --save ./dataset.npy

## 5.train

# Finally, we train the model using the augmented dataset.
# The --source_file option specifies the NumPy file containing the training data,
# and the --output_path option determines where the trained model will be saved.
# The mechanism file is again specified to ensure the model is aware of the chemical kinetics involved.

dfode-kit train --mech ../../mechanisms/Burke2012_s9r23.yaml     \
    --source_file ./dataset.npy     \
    --output_path ./demo_model.pt