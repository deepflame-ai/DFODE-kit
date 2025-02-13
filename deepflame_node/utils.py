import re
from pathlib import Path
import numpy as np

def is_numeric_string(input_string):
    pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$'

    if re.match(pattern, input_string):
        return True
    else:
        return False
    
def load_openfoam_scalar_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    sample_bool = False
    start_bracket_idx, end_bracket_idx = 0, 0
    
    for i, line in enumerate(lines):
        if "internalField" in line:
            sample_bool = True
        if "(" in line and sample_bool:
            start_bracket_idx = i + 1
        if ")" in line and sample_bool:
            end_bracket_idx = i
            break
    
    selected_lines = lines[start_bracket_idx:end_bracket_idx]
    
    # Load the selected lines into a NumPy array
    dim_array = np.loadtxt(selected_lines).reshape(-1, 1)
    
    return dim_array