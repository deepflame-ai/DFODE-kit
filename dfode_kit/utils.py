import numpy as np

def is_number(s):
    """Check if the string can be converted to a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_openfoam_scalar(file_path):
    """Read scalar values from an OpenFOAM file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize flags and indices
    is_internal_field = False
    start_index, end_index = None, None
    
    # Find the indices for the internal field values
    for i, line in enumerate(lines):
        if "internalField" in line:
            is_internal_field = True
        if is_internal_field:
            if "(" in line:  # Start of the values
                start_index = i + 1
            if ")" in line:  # End of the values
                end_index = i
                break
    
    # Ensure that indices are found
    if start_index is None or end_index is None:
        raise ValueError("Could not find internalField values in the file.")

    # Extract the selected lines and convert them to a numpy array
    selected_lines = lines[start_index:end_index]
    dim_array = np.loadtxt(selected_lines).reshape(-1, 1)

    return dim_array