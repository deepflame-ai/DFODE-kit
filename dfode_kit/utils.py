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

    # Initialize variables
    internal_field_index = None
    uniform_value = None
    expected_count = None
    start_index = None
    end_index = None

    # Find the indices for the internal field values
    for i, line in enumerate(lines):
        if "internalField" in line:
            internal_field_index = i
            if ";" in line: # For uniform fields
                parts = line.split()
                # Assuming format: "internalField uniform <value>;"
                uniform_value = float(parts[-1][:-1])
                return uniform_value  # Return uniform as a 1x1 array
            else:
                # Read the next line for the expected count
                expected_count = int(lines[i + 1].strip())  # Convert to int
                # print(expected_count)
                # Check the next line for "("
                if "(" in lines[i + 2] and ")" in lines[i + 2 + expected_count + 1]:
                    start_index = i + 2  # Start reading values from the next line
                    end_index = i + 2 + expected_count + 1
                    
                    selected_lines = lines[start_index+1:end_index]
                    dim_array = np.loadtxt(selected_lines).reshape(-1, 1)
                    break  # Exit loop after processing

    return dim_array

def BCT(x, lam=0.1):
    """
    Box-Cox Transformation (BCT)

    Args:
        x (np.ndarray): Input array.
        lam (float): Lambda parameter for the transformation.

    Returns:
        np.ndarray: Transformed values.

    Raises:
        ValueError: If any values in x are negative.
    """
    if np.any(x < 0):
        raise ValueError("Input array contains negative values. Box-Cox transformation is not defined for negative values.")

    if lam == 0:
        return np.log(x)
    else:
        return (np.power(x, lam) - 1) / lam

def inverse_BCT(y, lam=0.1):
    """
    Inverse Box-Cox Transformation

    Args:
        y (np.ndarray): Transformed array.
        lam (float): Lambda parameter for the reverse transformation.

    Returns:
        np.ndarray: Original values.

    Raises:
        ValueError: If lam is zero or if y contains negative values.
    """
    if lam == 0:
        return np.exp(y)
    else:
        return np.power(lam * y + 1, 1 / lam)