import torch
import numpy as np
import os
import cantera as ct

from dfode_kit.data_operations.h5_kit import advance_reactor

DFODE_ROOT = os.environ['DFODE_ROOT']
def test_npy(
    mech_path: str,
    source_file: str,
    output_path: str,
    time_step: float = 1e-6,
) -> np.ndarray:
    
    test_data = np.load(source_file) 
    