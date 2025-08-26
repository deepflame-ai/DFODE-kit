import numpy as np

def random_perturb(
    array: np.ndarray, 
    alpha: float = 0.1,
    inert_idx: int = -1,
) -> np.ndarray:
    """
    Apply random perturbations to the input array to enhance data variability.

    This function simulates the effects of multi-dimensional transport and turbulence disturbances 
    by perturbing the original values of temperature, pressure, and inert species mass fractions. 
    The perturbation is designed to overcome limitations in laminar canonical flames, where 
    sampled states are confined to predefined flamelet manifolds, leading to over-constrained 
    representations and insufficient coverage of thermochemical variations.

    The perturbation is applied as follows:
    
    1.  For the first two columns (e.g., temperature and pressure), the perturbation is computed 
        using the formula:
        xR = x + α · β · (xmax - xmin)
        where xR is the perturbed value, x is the original value, α is the user-defined perturbation 
        amplitude, and β is a random variable sampled uniformly from (-1, 1).
        
    2.  For the remaining columns (excluding the inert index), an exponential perturbation strategy 
        is employed:
        yR = y^(1 + α · β)
        where yR is the perturbed mass fraction and y is the original mass fraction.
        
    3.  The perturbed species mass fractions are then normalized to maintain mass conservation:
        Y_iR = y_iR · (1 - xN2_R) / Σ y_iR

    Parameters:
    - array (np.ndarray): The input array to be perturbed.
    - alpha (float): The scaling factor for the perturbation. Default is 0.1.
    - inert_idx (int): The index of the column that should not be transformed. Default is -1 (last column).

    Returns:
    - np.ndarray: The perturbed array.
    """
    new_array = np.copy(array)
    beta = np.random.uniform(-1, 1, size=array.shape)
    
    new_array[:, :2] += alpha * beta[:, :2] * (np.max(array[:, :2], axis=0) - np.min(array[:, :2], axis=0))
    new_array[:, 2:] = np.power(array[:, 2:], 1 + alpha * beta[:, 2:])
    
    new_array[:, 2:] = new_array[:, 2:] * (1 - array[:, inert_idx].reshape(-1, 1)) / (np.sum(new_array[:, 2:], axis=1, keepdims=True) - new_array[:, inert_idx].reshape(-1, 1))
    new_array[:, inert_idx] = array[:, inert_idx]
    
    return new_array
    
    
    