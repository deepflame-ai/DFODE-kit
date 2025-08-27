from .utils import read_openfoam_scalar, BCT, inverse_BCT, BCT_torch, inverse_BCT_torch

from .df_interface.sample_case import gather_species_arrays, df_to_h5

from .data_operations.h5_kit import touch_h5, get_TPY_from_h5
from .data_operations.h5_kit import advance_reactor
from .data_operations.h5_kit import load_model, predict_Y, nn_integrate, integrate_h5

import os
from pathlib import Path

DFODE_ROOT = Path(__file__).resolve().parent.parent
os.environ["DFODE_ROOT"] = str(DFODE_ROOT)