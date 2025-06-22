import numpy as np

class DataPreprocessor:
    def __init__(self, data):
        """
        Args:
            data (np.ndarray): The dataset as a numpy array.
        """
        self.data = data
        self.final_data = data.copy()
        
        self.num_rows, self.num_cols = data.shape
        
        # assuming the dataset cols are structured as follows:
        # 0: temperature, 1: pressure, [2 - n_species+2]: species concentrations
        # n_species+2: time step
        # n_species+3: temperature after reaction, n_species+4: pressure after reaction
        # [n_species+5 - 2*n_species+5]: species concentrations after reaction
        self.n_species = (self.num_cols - 5) // 2  # Derived from the structure described

        if (self.num_cols - 5) % 2 != 0:
            raise ValueError("The number of columns does not match the expected structure.")
        
        self.TP_cols = [0, 1, self.n_species + 3, self.n_species + 4]  # Temperature and Pressure columns
        self.Y_cols = list(range(2, 2 + self.n_species)) + list(range(self.n_species + 5, 2 * self.n_species + 5))  # Species concentration columns
        self.delta_t_col = [2 + self.n_species]  # Time step column
        
        self.TP0_cols = [0, 1]  # Initial Temperature and Pressure columns
        self.TP1_cols = [self.n_species + 3, self.n_species + 4]  # Final Temperature and Pressure columns
        self.Y1_cols = list(range(2, 2 + self.n_species))  # Initial species concentration columns
        self.Y2_cols = list(range(self.n_species + 5, 2 * self.n_species + 5))
                

    def op(self, columns, operation, *args, **kwargs):
        """
        Apply an operation to specified columns.
        
        Args:
            columns (list or str): List of column indices to operate on or 'all' to operate on all columns.
            operation (callable): Function to apply to the specified columns.
            *args: Additional positional arguments to pass to the operation function.
            **kwargs: Additional keyword arguments to pass to the operation function.
        """
        if columns == "all":
            columns = range(self.num_cols)  # Operate on all columns
        
        for col in columns:
            self.final_data[:, col] = operation(self.final_data[:, col], *args, **kwargs)
        
        print(f"Applied {operation.__name__} to columns: {columns}")
        
#     def clip_values(self, columns, min_value, max_value):
#         """
#         Clip values in specified columns to a given range.
        
#         Args:
#             columns (list or str): List of column indices to clip or 'all' to clip all columns.
#             min_value (float): Minimum value for clipping.
#             max_value (float): Maximum value for clipping.
#         """
#         if columns == "all":
#             columns = range(self.data.shape[1])  # Clip all columns
        
#         for col in columns:
#             self.data[:, col] = np.clip(self.data[:, col], min_value, max_value)

#     def normalize_z_score(self, columns):
#         """
#         Normalize specified columns using Z-score normalization.
        
#         Args:
#             columns (list or str): List of column indices to normalize or 'all' to normalize all columns.
#         """
#         if columns == "all":
#             columns = range(self.data.shape[1])  # Normalize all columns
        
#         for col in columns:
#             self.data[:, col] = (self.data[:, col] - np.mean(self.data[:, col])) / np.std(self.data[:, col])

#     def normalize_min_max(self, columns):
#         """
#         Normalize specified columns using Min-Max normalization.
        
#         Args:
#             columns (list or str): List of column indices to normalize or 'all' to normalize all columns.
#         """
#         if columns == "all":
#             columns = range(self.data.shape[1])  # Normalize all columns
        
#         for col in columns:
#             min_val = np.min(self.data[:, col])
#             max_val = np.max(self.data[:, col])
#             self.data[:, col] = (self.data[:, col] - min_val) / (max_val - min_val)

#     def box_cox_transform(self, columns):
#         """
#         Apply Box-Cox transformation to specified columns.
        
#         Args:
#             columns (list or str): List of column indices to transform or 'all' to transform all columns.
#         """
#         if columns == "all":
#             columns = range(self.data.shape[1])  # Transform all columns
        
#         for col in columns:
#             transformed_data, _ = stats.boxcox(self.data[:, col] + 1e-5)  # Add small value to avoid zero issues
#             self.data[:, col] = transformed_data

# # Example usage
# data = np.random.rand(100, 10)  # Example data with 10 columns
# preprocessor = DataPreprocessor(data)

# # Apply various transformations
# preprocessor.clip_values(columns="all", min_value=0.2, max_value=0.8)  # Clip all columns
# preprocessor.normalize_z_score(columns=[4, 5])  # Z-score normalization on specific columns
# preprocessor.normalize_min_max(columns="all")  # Min-Max normalization on all columns
# preprocessor.box_cox_transform(columns=[8, 9])  # Box-Cox transformation on specific columns

# # Compute targets
# target = preprocessor.compute_target(y1_cols=[2, 3], y2_cols=[7, 8])
# print("Computed Target:", target)