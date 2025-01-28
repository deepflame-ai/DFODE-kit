import os
import re
import json
import numpy as np
import cantera as ct

from pathlib import Path

def is_numeric_string(input_string):
    # Regular expression pattern for numeric string with decimal point or scientific expression
    pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$'

    # Check if the input string matches the pattern
    if re.match(pattern, input_string):
        return True
    else:
        return False
    
def find_line_index_with_string(file_path, target_string):
    with open(file_path, 'r') as tmpfile:
        for line_index, line in enumerate(tmpfile):
            if target_string in line:
                next_line = next(tmpfile, None)  # Retrieve the next line
                return line_index, int(re.findall(r'\d+', next_line)[0]) if next_line else None
    return -1, None  # Indicates that the target string was not found in any line

dfode_root = os.environ.get('DFODE_ROOT')
if not dfode_root:
    raise ValueError("DFODE_ROOT environment variable not set")

configuration_file_path = Path(dfode_root) / 'samplingStrategy.json'
with open(configuration_file_path, 'r') as file:
    parameters = json.load(file)
    
    mechanism = parameters['mechanism']
    min_temperature = parameters['inlet_temperature'] + 10

print("STARTING DATA COLLECTION\n")
gas = ct.Solution('Okafor26_noCO2.yaml')
TPspecies_list = ['T', 'p'] + gas.species_names

data_tag = '1Dflame'
main_dir = Path('.')
time_dirs = [glob_finding for glob_finding in main_dir.rglob("*") if glob_finding.is_dir() and is_numeric_string(glob_finding.name)]

selected_count, total_count = 0, 0
data_collector = []

for time_dir in time_dirs:
    data_dict = {}
    start_index, grid_num = find_line_index_with_string(time_dir / 'T', 'internalField')
    for filename in TPspecies_list:
            file_path = time_dir / filename
            
            try:    
                with open(file_path, 'r') as tmpfile:
                    file_lines = tmpfile.readlines()
                    lines = file_lines[start_index+3:start_index+3+grid_num]
                    data = np.array([float(line.strip()) for line in lines])
                    data_dict[filename] = data
            
            except:
                try:
                    with open(file_path, 'r') as tmpfile:
                        lines = tmpfile.readlines()[start_index]
                        words = [substring for substring in lines.replace(' ', ';').replace('\n', ';').split(';') if substring]
                        data0 = float(words[-1]) if is_numeric_string(words[-1]) else '-'
                        data=np.full(grid_num,data0)
                        data_dict[filename] = data

                except:
                    pass
                            
    time_total_data = np.array(list(data_dict.values())).T
    time_selected_data = time_total_data[time_total_data[:,0]>min_temperature]
    
    selected_count += time_selected_data.shape[0]
    total_count += time_total_data.shape[0]
    print(f"{time_selected_data.shape[0]} / {time_total_data.shape[0]} data points from {time_dir}")
    data_collector.extend(time_selected_data)

output_data_array = np.array(data_collector)
np.save(data_tag, output_data_array)

print(f"\n{len(time_dirs):>8} TIME DIRECTORIES FOUND")
print(f"{len(TPspecies_list):>8} DIMENSIONS IN DATA")
# print(len(data_collector))  ###检查  网格数*算例数*采集比例   对于1d算例约为0.5,对于0d算例约为1.0
print(f"Using mechanism: {mechanism}")
print(f"\nTotal number of selected data points: {selected_count} / {total_count}")
print("FINAL DATA DIMENSION: ", output_data_array.shape)
print(f"DATA SAVED TO <{data_tag}.npy>")
