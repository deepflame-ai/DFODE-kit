import os
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

DFODE_ROOT = os.environ['DFODE_ROOT']
from dfode_kit.data_operations import integrate_h5, touch_h5
# from dfode_kit.dfode_core.test.test import test_npy
from dfode_kit.dfode_core.model.mlp import MLP

mech_path = f'{DFODE_ROOT}/mechanisms/Burke2012_s9r23.yaml'
gas = ct.Solution(mech_path)
n_species = gas.n_species

model_settings = {
    'model_path': "model.pt",
    'device': 'cpu',
    'model_class': MLP,
    'model_layers': [n_species+2, 400, 400, 400, 400, n_species-1],
    'time_step': 1e-6,
    'mech': f"{DFODE_ROOT}/mechanisms/Burke2012_s9r23.yaml"
}

integrate_h5("tutorial_data.h5", 1e-6, nn_integration=False, model_settings=model_settings)
touch_h5("tutorial_data.h5")

# test_npy("test.npy", model_settings=model_settings)
