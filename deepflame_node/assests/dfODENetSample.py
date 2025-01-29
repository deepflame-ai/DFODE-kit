from src.workingcondition import WorkingCondition, OneDSettings
from src.utils import my_trunc

import os
import re
import sys
import math
import json
import shutil
import subprocess
import numpy as np
import cantera as ct

from pathlib import Path
from tabulate import tabulate, SEPARATING_LINE

# ================================================================================
# Parameter Setup for Sampling Process:
# Mechanism files should be put into src/mechanisms and $CANTERA_DATA

project_root_path = Path.cwd()
os.environ['DFODE_ROOT'] = os.getcwd()
oneD_flame_template_path = project_root_path / '0_1DFlameTemplate'
mechanism_dir_path = project_root_path / 'src' / 'mechanisms'
ct.add_directory(mechanism_dir_path)

# ================================================================================
with open('samplingStrategy.json', 'r') as file:
    parameters = json.load(file)

mechanism                   =   parameters['mechanism']
air                         =   parameters['air_composition']
T_inlet                     =   parameters['inlet_temperature']
p                           =   parameters['inlet_pressure']

fuel_compositions           =   parameters['fuel_compositions']

inert_specie                =   parameters['inert_specie']

simulation_time_step        =   float(parameters['simulation_time_step'])   # time step for CFD simulation
num_mesh_cells              =   parameters['num_mesh_cells']                # number of mesh grids in 1d-flame simulation
sampling_number             =   parameters['sampled_times_per_case']        # number of sampled times in 1d-flame simulation
sampling_temperature        =   T_inlet + 10

min_equivalence_ratio       =   parameters['min_equivalence_ratio']
max_equivalence_ratio       =   parameters['max_equivalence_ratio']
num_equivalence_ratio       =   parameters['num_equivalence_ratio']

equiv_ratio_values          =   np.linspace(min_equivalence_ratio,
                                            max_equivalence_ratio,
                                            num_equivalence_ratio)

ratio_digit_ctrl            =   parameters['ratio_digit_ctrl']
fuel_composition_digit_ctrl =   parameters['fuel_composition_digit_ctrl']

oneD_flame_mechanism_path = oneD_flame_template_path / mechanism
src_mechanism_path = mechanism_dir_path / mechanism
if not src_mechanism_path.exists():
    raise FileNotFoundError(f"Cannot find the mechanism file: {mechanism} in src/mechanisms")
if not oneD_flame_mechanism_path.exists() and src_mechanism_path.exists():
    shutil.copy(src_mechanism_path, oneD_flame_mechanism_path)
    
# ================================================================================

one_d_settings = OneDSettings(
    mechanism,
    inert_specie,
    simulation_time_step,
    num_mesh_cells
)

print("OVERVIEW OF SAMPLING SETUP:")

if len(equiv_ratio_values) >= 6:
    equiv_ratio_values_str = f"From {equiv_ratio_values[0]:.3f} to {equiv_ratio_values[-1]:.3f} with interval of {(equiv_ratio_values[1]-equiv_ratio_values[0]):.3f}"
else:
    equiv_ratio_values_str = "[ "+', '.join(f'{value:.3f}' for value in equiv_ratio_values)+" ]"

table = [['Mechanism', mechanism, ''],
         SEPARATING_LINE,
         ['Temperature', f"{T_inlet:.0f}", "K"],
         ['Pressure', f"{p:.0f}", "Pa"],
         ['Fuel Compositions', fuel_compositions[0], f"{len(fuel_compositions)} value(s) in total"],
         ['Equivalence Ratio', equiv_ratio_values_str, f"{len(equiv_ratio_values)} value(s) in total"],
         SEPARATING_LINE,
         ['Simulation Time Step', f"{simulation_time_step:.0e}", "s"],
         ['Minimum Sampling Temperature', f"{sampling_temperature:.0f}", "K"],]
print(tabulate(table, colalign=("left","right", "left")))

# ================================================================================



# ================================================================================
print("\n"+"="*50)

oneD_flames = []
case_index = 0
for fuel_composition in fuel_compositions:
    for equiv_ratio in equiv_ratio_values:
        case_index +=1
        print("For Case "+str(case_index)+":\n")
        
        case = WorkingCondition(
            one_d_settings,
            case_index,
            mechanism,
            T_inlet,
            p,
            fuel_composition,
            air,
            equiv_ratio
        )
        
        LFS, thickness = case.calculate_flame_speed()
        
        flame_dict = {'case': case}
        
        minimum_mesh_size = thickness * 1000 * num_mesh_cells / 10  # mm
        mesh_size = minimum_mesh_size//1 + 1                        # mm
        flame_dict['minimum_mesh_size'] = minimum_mesh_size
        flame_dict['mesh_size'] = mesh_size
        
        if '--debug' not in sys.argv:
            sampling_time_span = my_trunc((case.chemical_time_scale*1000*5.5), 1)
            sampling_time_step = my_trunc(((case.chemical_time_scale*5.5/sampling_number/simulation_time_step)//1)*simulation_time_step*1000, 3)
            flame_dict['sampling_time_span'] = sampling_time_span
            flame_dict['sampling_time_step'] = sampling_time_step
        else:
            flame_dict['sampling_time_span'] = 0.02
            flame_dict['sampling_time_step'] = 0.005
        
        oneD_flames.append(flame_dict)
        
        table = [
            [f"Case {case_index}", '', ''],
            ['Temperature',                  case.temperature,                  "K"],
            ['Pressure',                     case.pressure,                     "Pa"],
            ['Fuel Composition',             case.fuel_composition,             ''],
            ['Equivalence Ratio',            case.equivalence_ratio,            ''],
            SEPARATING_LINE,
            ['Laminar Flame Speed',          case.laminar_flame_speed,          "m/s"],
            ['Laminar Flame Thickness',      case.laminar_flame_thickness*1e6,  "μm"],
            ['Chemical Time Scale',          case.chemical_time_scale*1000,     "ms"],
            ['Sampling Time Span',           flame_dict['sampling_time_span'],  "ms"],
            ['Minimum Mesh Size',            minimum_mesh_size,                 "mm"],
            ['Mesh Size',                    mesh_size,                         "mm"],
            SEPARATING_LINE,
            ['Simulation Time Step',         simulation_time_step,              "s"],
            ['Sampling Time Step',           flame_dict['sampling_time_step'],  "ms"],
            ['Minimum Sampling Temperature', sampling_temperature,              "K"]
        ]
        # TODO: Save all parameters in base SI units; convert to desired units when printing
        
        
        print("\nPlease Check The Following Setup Information:")
        print(tabulate(table, colalign=("left","right", "left"), headers="firstrow"))
        print("=" * 50+'\n')
        
# ================================================================================



# ================================================================================
# Run the simulations
# ================================================================================
for index, flame in enumerate(oneD_flames):
    case = flame['case']
    case_index = case.tag
    case_name_stem = f"{case_index}_1DFlame_{case.fuel_composition}_eqr{case.equivalence_ratio:.3f}"
    case_name_stem = re.sub(r"[.:,]", "", case_name_stem)
    print("CHECK: "+case_name_stem)
    
    run_path = project_root_path / '1_dataGeneration' / case_name_stem
    
    if run_path.exists():
        shutil.rmtree(run_path)
    
    shutil.copytree(oneD_flame_template_path, run_path)
    
    os.chdir(run_path)
    print("Going Into:   "+str(Path.cwd()))
    print("=" * 50+"\n\n")
    
    settings = {
        'mechanism':                case.mechanism,
        'case_index':               case_index,
        'T_inlet':                  case.temperature,
        'p':                        case.pressure,
        'fuel_':                    case.fuel_composition,
        'air':                      case.oxidizer_composition,
        'phi_v':                    case.equivalence_ratio,
        'inert_specie':             case.inert_specie,
        'LFS':                      case.laminar_flame_speed,
        'LFT':                      case.laminar_flame_thickness*1e6,
        'CTS':                      case.chemical_time_scale*1e3,
        'total_sampling_time':      flame['sampling_time_span'],
        'num_of_cells':             num_mesh_cells,
        'MMS':                      flame['minimum_mesh_size'],
        'case_length_setup':        flame['mesh_size'],
        'cfd_time_step':            simulation_time_step,
        'sampling_time_interval':   flame['sampling_time_step'],
        'sampling_temperature':     sampling_temperature
    }

    json_file = 'dnn_scripts/settings.json'

    with open(json_file, 'w') as file:
        json.dump(settings, file, indent=4)

    subprocess.run(["./run_dnn.sh"])
    
    validation_case_path = (
        project_root_path 
        / '4_validation' 
        / '0_1DFlame' 
        / case_name_stem
    )
    
    shutil.copytree(run_path, validation_case_path)
    
    print("Running 1D-Flame Case...")
    sample_data_path = (
        project_root_path 
        / '2_sampling' 
        / 'sampling' 
        / case_name_stem
    )
    
    print(str(sample_data_path))
    #subprocess.run(["./Allrun", str(sample_data_path)])
    
    print([path.stem for path in sample_data_path.parent.iterdir() if path.is_dir()])
    print([path.stem for path in validation_case_path.parent.iterdir() if path.is_dir()])
    
    print("=" * 50+"\n\n")
        
        
        
# ================================================================================
