import os
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

from dfode_kit import DFODE_ROOT
from dfode_kit.data_operations import integrate_h5, touch_h5, calculate_error
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

file_path = f"{DFODE_ROOT}/model_test/priori/tutorial_data.h5"
save_path1 = f"{DFODE_ROOT}/model_test/priori/label.h5"
save_path2 = f"{DFODE_ROOT}/model_test/priori/model_output.h5"
integrate_h5(file_path, save_path1, save_path2, 1e-6, nn_integration=True, model_settings=model_settings)
touch_h5(save_path1)   ##  Time_step, T, P, Y
touch_h5(save_path2)
touch_h5(file_path)
calculate_error(mech_path, save_path1, save_path2)
