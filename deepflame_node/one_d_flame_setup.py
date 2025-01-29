import cantera as ct
import numpy as np
import shutil

def calculate_laminar_flame_properties(mechanism, gas_state):
    
    # Initialize the gas object
    flame_speed_gas = ct.Solution(mechanism)
    flame_speed_gas.TP = gas_state['initial_temperature'], gas_state['initial_pressure']
    
    # Set the equivalence ratio
    flame_speed_gas.set_equivalence_ratio(
        gas_state['equivalence_ratio'],
        fuel=gas_state['fuel_composition'],
        oxidizer=gas_state['oxidizer_composition'],
    )

    # Create and solve the flame
    width = 0.1
    flame = ct.FreeFlame(flame_speed_gas, width=width)
    flame.set_refine_criteria(ratio=3, slope=0.05, curve=0.1, prune=0.0)

    print("Solving premixed flame...")
    flame.solve(loglevel=0, auto=True)

    # Access laminar flame speed
    laminar_flame_speed = flame.velocity[0]
    print(f'{"Laminar Flame Speed":<25}:{laminar_flame_speed:>15.10f} m/s')

    # Calculate laminar flame thickness
    z, T = flame.grid, flame.T
    grad = (T[1:] - T[:-1]) / (z[1:] - z[:-1])
    laminar_flame_thickness = (max(T) - min(T)) / max(grad)
    print(f'{"Laminar Flame Thickness":<25}:{laminar_flame_thickness:>15.10f} m')
    
    final_flame = flame.to_solution_array()

    return laminar_flame_speed, laminar_flame_thickness, final_flame

def update_case_parameters(
        mechanism,
        gas_state,
        flame_speed, 
        flame_thickness
    ):
    params = {
        'flame_thickness': flame_thickness,
        'flame_speed': flame_speed,
        'domain_width': flame_thickness / 10 * 50,
        'domain_length': 10 * (flame_thickness / 10 * 50),
        'target_time_step': 1e-6,
        'chemical_time_scale': flame_thickness / flame_speed,
        'sample_time_steps': 100,
    }
    params['estimated_time_step'] = params['chemical_time_scale'] * 10 / params['sample_time_steps']
    params['estimated_sim_time'] = params['estimated_time_step'] * (params['sample_time_steps'] + 1)
    params['estimated_write_time_step'] = params['estimated_time_step']

    for key, value in params.items():
        print(f"{key.replace('_', ' ').title()}: {value:.2e}")

    unburnt_gas = ct.Solution(mechanism)
    unburnt_gas.TP = gas_state['initial_temperature'], gas_state['initial_pressure']

    unburnt_gas.set_equivalence_ratio(
        gas_state['equivalence_ratio'],
        fuel=gas_state['fuel_composition'],
        oxidizer=gas_state['oxidizer_composition'],
    )

    params['unburnt_gas'] = unburnt_gas

    equilibrium_gas = ct.Solution(mechanism)
    equilibrium_gas.TP = gas_state['initial_temperature'], gas_state['initial_pressure']

    equilibrium_gas.set_equivalence_ratio(
        gas_state['equivalence_ratio'],
        fuel=gas_state['fuel_composition'],
        oxidizer=gas_state['oxidizer_composition'],
    )

    equilibrium_gas.equilibrate('HP')

    params['equilibrium_gas'] = equilibrium_gas

    return params

def update_one_d_sample_config(case_params, gas_state):
    orig_file_path = 'system/sampleConfigDict.orig'
    new_file_path = 'system/sampleConfigDict'

    shutil.copy(orig_file_path, new_file_path)

    with open(new_file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if "domainWidth" in line:
            lines[i] = line.replace("placeHolder", f'{case_params["domain_width"]}')
        if "targetNetTimeStep" in line:
            lines[i] = line.replace("placeHolder", f'{case_params["target_time_step"]}')
        if "simEndTime" in line:
            lines[i] = line.replace("placeHolder", f'{case_params["estimated_sim_time"]}')
        if "simSampleTimeStep" in line:
            lines[i] = line.replace("placeHolder", f'{case_params["estimated_time_step"]}')
        if "simWriteInterval" in line:
            lines[i] = line.replace("placeHolder", f'{case_params["estimated_write_time_step"]}')
        if "UInlet" in line:
            lines[i] = line.replace("placeHolder", f'{case_params["flame_speed"]}')
        if "pInternal" in line:
            lines[i] = line.replace("placeHolder", f'{gas_state["initial_pressure"]}')
        if "unburntStates" in line:
            state_strings = [f'{"TUnburnt":<20}{case_params["unburnt_gas"].T:>16.10f};']
            for _, species in enumerate(case_params["unburnt_gas"].species_names):
                key_string = f'{species}Unburnt'
                state_strings.append(f'{key_string:<20}{case_params["unburnt_gas"].Y[_]:>16.10f};')
            lines[i] = '\n'.join(state_strings) + '\n\n'
        if "equilibriumStates" in line:
            state_strings = [f'{"TBurnt":<20}{case_params["equilibrium_gas"].T:>16.10f};']
            for _, species in enumerate(case_params["equilibrium_gas"].species_names):
                key_string = f'{species}Burnt'
                state_strings.append(f'{key_string:<20}{case_params["equilibrium_gas"].Y[_]:>16.10f};')
            lines[i] = '\n'.join(state_strings) + '\n\n'

    with open(new_file_path, 'w') as file:
        file.writelines(lines)

def create_0_species_files(case_params):
    orig_0_file_path = '0/Ydefault.orig'
    for _, species in  enumerate(case_params["unburnt_gas"].species_names):
        new_0_file_path = f'0/{species}.orig'
        shutil.copy(orig_0_file_path, new_0_file_path)

        with open(new_0_file_path, 'r') as file:
            lines = file.readlines()
        
        for i, line in enumerate(lines):
            if "Ydefault" in line:
                lines[i] = line.replace("Ydefault", f'{species}')
            if "uniform 0" in line:
                lines[i] = line.replace("0", f'{case_params["unburnt_gas"].Y[_]}')
        
        with open(new_0_file_path, 'w') as file:
            file.writelines(lines)

def update_set_fields_dict(case_params):
    orig_setFieldsDict_path = 'system/setFieldsDict.orig'
    new_setFieldsDict_path = 'system/setFieldsDict'

    shutil.copy(orig_setFieldsDict_path, new_setFieldsDict_path)

    with open(new_setFieldsDict_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if "unburntStatesPlaceHolder" in line:
            state_strings = [f'\tvolScalarFieldValue {"T":<10} $TUnburnt']
            for _, species in enumerate(case_params["unburnt_gas"].species_names):
                state_strings.append(f'volScalarFieldValue {species:<10} ${species}Unburnt')
            lines[i] = '\n\t'.join(state_strings) + '\n'
        if "equilibriumStatesPlaceHolder" in line:
            state_strings = [f'\t\t\tvolScalarFieldValue {"T":<10} $TBurnt']
            for _, species in enumerate(case_params["equilibrium_gas"].species_names):
                state_strings.append(f'volScalarFieldValue {species:<10} ${species}Burnt')
            lines[i] = '\n\t\t\t'.join(state_strings) + '\n'
            
            
    with open(new_setFieldsDict_path, 'w') as file:
        file.writelines(lines)