import os
import numpy as np

# ================================================================================
def read_flameSpeedLog(path):
    data = {}

    for root, dirs, files in os.walk(path):
        for file in files:
            if 'log' in file:
                temp = []
                start_reading = False
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        if line.strip() == '':
                            continue
                        if 'Create time' in line:
                            start_reading = True
                            continue
                        if not start_reading:
                            continue
                        words = line.split()
                        if words[0] == 'Time' or words[0] == 'flamePropagationSpeed' or words[0] == 'flameSpeed':
                            temp.append(float(words[2]))
                            if len(temp) == 3:
                                dir_name = os.path.basename(root)
                                if 'data' in dir_name:
                                    dir_name = dir_name.split('data_', 1)[-1]
                                if 'result' in dir_name:
                                    dir_name = file.split('validation_', 1)[-1]
                                    dir_name = dir_name.split('_log', 1)[0]
                                if dir_name not in data:
                                    data[dir_name] = []
                                data[dir_name].append(temp)
                                temp = []

    for key in data:
        data[key] = np.array(data[key])
        
    return data
# ================================================================================

# ================================================================================
def check_time(arr1, arr2):
    time_arr1 = arr1[:, 0]
    time_arr2 = arr2[:, 0]
    
    if len(time_arr1) != len(time_arr2):
        print('Lengths of the time arrays are different.')
        return False
    else:
        if np.array_equal(time_arr1, time_arr2):
            print('Time arrays are equal.')
            return True
        else:
            print('Time arrays are not equal.')
            return False
# ================================================================================

# ================================================================================
def flame_speed_extraction(arr):
    for i in range(len(arr) - 1, 0, -1):
        if arr[i, 1] < 1e-6 and abs((arr[i, 2] - arr[i-1, 2]) / arr[i-1, 2]) < 0.01:
            return arr[i, 0], arr[i, 2], i
    return None, None, None
# ================================================================================


# ================================================================================
# ================================================================================


cvode_data = read_flameSpeedLog('/home/xk/Uni/4_dnn/1_NH3_CH4/1_CurrentWork/0_copy/1_sampling/sampling')
dnn_data = read_flameSpeedLog('/home/xk/Uni/4_dnn/1_NH3_CH4/1_CurrentWork/0_copy/3_validation/0_1DFlame/results')

print(cvode_data)
print(dnn_data)

for key in cvode_data.keys():
    print(key)
for key in dnn_data.keys():
    if key in cvode_data.keys():
        print()
        print(key)
        check_time(cvode_data[key], dnn_data[key])
        selected_time, flame_speed, index = flame_speed_extraction(dnn_data[key])
        time_cvode, flame_speed_cvode = cvode_data[key][index,0], cvode_data[key][index, 2]
        print('Selected time:', time_cvode, selected_time)
        print('Flame speed of cvode:', flame_speed_cvode)
        print('Flame speed:', flame_speed)
        print(cvode_data[key], dnn_data[key])