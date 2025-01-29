from workingcondition import WorkingCondition, OneDSettings

import cantera as ct
import numpy as np

import math
from tabulate import tabulate, SEPARATING_LINE
tabulate.PRESERVE_WHITESPACE = True

# ================================================================================
# Sampling Setup
# ================================================================================
# Mechanism files should be put into /your/path/miniconda3/envs/your_env/lib/python3.8/site-packages/cantera/data/

mechanism                   =   'Okafor2018_s59r356.yaml'
air                         =   "O2:0.21,N2:0.79"
T_inlet                     =   300
p                           =   ct.one_atm

primary_fuel                =   "NH3"
secondary_fuel              =   "CH4"          #  use "" if single fuel
diluent_name                =   ""             #  use "" if no diluent
inert_specie                =   "AR"

CFD_dT                      =   1e-6
num_mesh_cells              =   500            #  number of mesh cells in 1d-flame simulation
sampling_number             =   100            #  number of sampled times in 1d-flame simulation
sampling_temperature        =   T_inlet + 10

blending_ratio_values       =   np.linspace(0.0, 0.6, 2) if secondary_fuel != "" else [1.0]
equiv_ratio_values          =   np.linspace(0.8, 1.2, 2)
# equiv_ratio_values          =   np.arange(0.8, 1.0, 0.05)

ratio_digit_ctrl            =   3
fuel_composition_digit_ctrl =   3

dilution_mole               =   0.1

# ================================================================================

def my_trunc(number, digits):
    """Truncates a number to a specified number of decimal digits."""
    stepper = np.power(10, digits)
    return np.trunc(stepper * number) / stepper


        
        

# ================================================================================

one_d_settings = OneDSettings(
    mechanism,
    inert_specie,
    CFD_dT,
    num_mesh_cells
)

fuel = primary_fuel if secondary_fuel == "" else primary_fuel+" & "+secondary_fuel

if len(blending_ratio_values) >= 6:
    blending_ratio_values_str = f"From {blending_ratio_values[0]:.3f} to {blending_ratio_values[-1]:.3f} with interval of {(blending_ratio_values[1]-blending_ratio_values[0]):.3f}" if secondary_fuel != "" else "-"
else:
    blending_ratio_values_str = "[ "+', '.join(f'{value:.3f}' for value in blending_ratio_values)+" ]"
blending_ratio_table_val = f"{len(blending_ratio_values)} values in total" if secondary_fuel != "" else "-"

if len(equiv_ratio_values) >= 6:
    equiv_ratio_values_str = f"From {equiv_ratio_values[0]:.3f} to {equiv_ratio_values[-1]:.3f} with interval of {(equiv_ratio_values[1]-equiv_ratio_values[0]):.3f}"
else:
    equiv_ratio_values_str = "[ "+', '.join(f'{value:.3f}' for value in equiv_ratio_values)+" ]"

table = [['Mechanism', mechanism, ''],
         ['Fuel', fuel, ''],
         SEPARATING_LINE,
         ['Temperature', f"{T_inlet:.0f}", "K"],
         ['Pressure', f"{p:.0f}", "Pa"],
         ['Blending Ratio', blending_ratio_values_str, blending_ratio_table_val],
         ['Equivalence Ratio', equiv_ratio_values_str, f"{len(equiv_ratio_values)} values in total"],
         SEPARATING_LINE,
         ['Simulation Time Step', f"{CFD_dT:.0e}", "s"],
         ['Minimum Sampling Temperature', f"{sampling_temperature:.0f}", "K"],]
print(tabulate(table, tablefmt="rst", colalign=("left","right", "left")))

# ================================================================================

case_list = []

case_count = 0

for blending_ratio in blending_ratio_values:
    for equiv_ratio in equiv_ratio_values:
        case_count +=1
        
        if secondary_fuel == "" :
            fuel_composition = primary_fuel+":1"
        elif diluent_name == "" :  
            fuel_composition = primary_fuel+f":{blending_ratio:.{fuel_composition_digit_ctrl}f},"\
                                +secondary_fuel+f":{1-blending_ratio:.{fuel_composition_digit_ctrl}f}"
        else:
            fuel_composition = primary_fuel+f":{blending_ratio/(1+dilution_mole):.{fuel_composition_digit_ctrl}f},"\
                                +secondary_fuel+f":{(1-blending_ratio)/(1+dilution_mole):.{fuel_composition_digit_ctrl}f},"\
                                +diluent_name+f":{dilution_mole/(1+dilution_mole):.{fuel_composition_digit_ctrl}f}"

        case = WorkingCondition(
            one_d_settings,
            case_count,
            mechanism,
            T_inlet,
            p,
            fuel_composition,
            air,
            equiv_ratio
        )
        
        LFS, thickness = case.calculate_flame_speed()
        
        case_list.append(case)
        
laminar_flame_speeds = np.array([case.laminar_flame_speed for case in case_list])
flame_thicknesses = np.array([case.laminar_flame_thickness for case in case_list])

domain_sizes = flame_thicknesses * 511 / 10

turbulent_velocity = 8
num_vortex = 10

turbulent_length_scales = domain_sizes / num_vortex

y = turbulent_velocity / laminar_flame_speeds
x = turbulent_length_scales / flame_thicknesses

Re = x * y
Ka = (x**-0.5) * (y**1.5)

print("\n\n")

table = []
headers = [
    'Fuel Comp.', 
    'E.R. \n[-]', 
    'L.F.S. \n[mm/s]',
    'T.V. \n[m/s]', 
    'y \n[-]',
    'L.F.T. \n[mm]', 
    'D.S. \n[mm]', 
    'T.L.S. \n[mm]', 
    'x \n[-]',
    'Re \n[-]', 
    'Ka \n[-]'
]

for i, case in enumerate(case_list):
    table.append([
        case.fuel_composition,
        f"{case.equivalence_ratio:.1f}",
        f"{case.laminar_flame_speed*1000:.2f}",
        f"{turbulent_velocity:.0f}",
        f"{y[i]:.2f}",
        f"{case.laminar_flame_thickness*1000:.2f}",
        f"{domain_sizes[i]*1000:.2f}",
        f"{turbulent_length_scales[i]*1000:.2f}",
        f"{x[i]:.2f}",
        f"{Re[i]:.2f}",
        f"{Ka[i]:.2f}"
    ])

table_align_fmt = ("left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right")
print(tabulate(table, headers, tablefmt="simple", colalign=table_align_fmt, disable_numparse=True))

# print(Re)
# print(Ka) 

