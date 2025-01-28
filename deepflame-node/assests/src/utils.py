import os
import re
import numpy as np

from pathlib import Path

def read_OpenFOAM_data(file_name):
    def is_numeric_string(input_string):
        if input_string is None:
            return False
        pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$'
        return bool(re.fullmatch(pattern, input_string))
        
    def find_line_index_with_string(file_name, target_string):
        with open(file_name, 'r') as file:
            for line_index, line in enumerate(file):
                if target_string in line:
                    next_line = next(file, None)
                    return line_index, int(next_line) if next_line else None
        return -1, None
    
    try:      
        try:    
            start_index, grid_num = find_line_index_with_string(file_name, 'internalField')
            
            with open(file_name, 'r') as file:
                file_lines = file.readlines()
                lines = file_lines[start_index+3:start_index+3+grid_num]
                data = np.array([float(line.strip()) for line in lines])
                
                return data

        except:
            try:
                with open(file_name, 'r') as file:
                    lines = file.readlines()[start_index]
                    words = [substring for substring in lines.replace(' ', ';').replace('\n', ';').split(';') if substring]
                    data0 = float(words[-1]) if is_numeric_string(words[-1]) else '-'
                    data=np.full(grid_num,data0)
                    
                    return data

            except:
                return None
    
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None

    except ValueError:
        print(f"Invalid data format in '{file_name}'.")
        return None


def my_trunc(number, digits):
    """Truncates a number to a specified number of decimal digits."""
    stepper = np.power(10, digits)
    return np.trunc(stepper * number) / stepper