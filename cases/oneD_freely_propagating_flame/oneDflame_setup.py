import cantera as ct
import numpy as np
import shutil

from pathlib import Path

from dfode_kit.df_interface.flame_configurations import OneDFreelyPropagatingFlameConfig

def update_one_d_sample_config(cfg: OneDFreelyPropagatingFlameConfig, case_path):
    case_path = Path(case_path).resolve()
    orig_file_path = case_path / 'system/sampleConfigDict.orig'
    new_file_path = case_path / 'system/sampleConfigDict'
    shutil.copy(orig_file_path, new_file_path)
    
    replacements = {
        "CanteraMechanismFile_": f'"{Path(cfg.mech_path).resolve()}"',
        "inertSpecie_": f'"{cfg.inert_specie}"',
        
        "domainWidth": cfg.domain_width,
        "domainLength": cfg.domain_length,
        "ignitionRegion": cfg.ignition_region,
        
        "simTimeStep": cfg.sim_time_step,
        "simTime": cfg.sim_time,
        "simWriteInterval": cfg.sim_write_interval,
        
        "UInlet": cfg.inlet_speed,
        "pInternal": cfg.p0,
    }
    
    with open(new_file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        for key, value in replacements.items():
            if key in line:
                lines[i] = line.replace("placeHolder", str(value))
                
        # Update unburnt states
        if "unburntStates" in line:
            state_strings = [f'{"TUnburnt":<20}{cfg.initial_gas.T:>16.10f};']
            state_strings += [
                f'{species}Unburnt'.ljust(20) + f'{cfg.initial_gas.Y[idx]:>16.10f};'
                for idx, species in enumerate(cfg.species_names)
            ]
            lines[i] = '\n'.join(state_strings) + '\n\n'

        # Update equilibrium states
        if "equilibriumStates" in line:
            state_strings = [f'{"TBurnt":<20}{cfg.burnt_gas.T:>16.10f};']
            state_strings += [
                f'{species}Burnt'.ljust(20) + f'{cfg.burnt_gas.Y[idx]:>16.10f};'
                for idx, species in enumerate(cfg.species_names)
            ]
            lines[i] = '\n'.join(state_strings) + '\n\n'

    with open(new_file_path, 'w') as file:
        file.writelines(lines)

def create_0_species_files(cfg: OneDFreelyPropagatingFlameConfig, case_path):
    case_path = Path(case_path).resolve()
    orig_0_file_path = case_path / '0/Ydefault.orig'
    
    for _, species in  enumerate(cfg.species_names):
        new_0_file_path = f'0/{species}.orig'
        shutil.copy(orig_0_file_path, new_0_file_path)

        with open(new_0_file_path, 'r') as file:
            lines = file.readlines()
        
        for i, line in enumerate(lines):
            if "Ydefault" in line:
                lines[i] = line.replace("Ydefault", f'{species}')
            if "uniform 0" in line:
                lines[i] = line.replace("0", f'{cfg.initial_gas.Y[_]}')
        
        with open(new_0_file_path, 'w') as file:
            file.writelines(lines)

def update_set_fields_dict(cfg: OneDFreelyPropagatingFlameConfig, case_path):
    case_path = Path(case_path).resolve()
    orig_setFieldsDict_path = case_path / 'system/setFieldsDict.orig'
    new_setFieldsDict_path = case_path / 'system/setFieldsDict'
    shutil.copy(orig_setFieldsDict_path, new_setFieldsDict_path)

    with open(new_setFieldsDict_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if "unburntStatesPlaceHolder" in line:
            state_strings = [f'\tvolScalarFieldValue {"T":<10} $TUnburnt']
            for _, species in enumerate(cfg.species_names):
                state_strings.append(f'volScalarFieldValue {species:<10} ${species}Unburnt')
            lines[i] = '\n\t'.join(state_strings) + '\n'
        if "equilibriumStatesPlaceHolder" in line:
            state_strings = [f'\t\t\tvolScalarFieldValue {"T":<10} $TBurnt']
            for _, species in enumerate(cfg.species_names):
                state_strings.append(f'volScalarFieldValue {species:<10} ${species}Burnt')
            lines[i] = '\n\t\t\t'.join(state_strings) + '\n'
            
            
    with open(new_setFieldsDict_path, 'w') as file:
        file.writelines(lines)

def setup_one_d_flame_case(cfg: OneDFreelyPropagatingFlameConfig, case_path):
    case_path = Path(case_path).resolve()
    
    # Update sampleConfigDict
    update_one_d_sample_config(cfg, case_path)
    
    # Create 0/ species files
    create_0_species_files(cfg, case_path)
    
    # Update setFieldsDict
    update_set_fields_dict(cfg, case_path)
    
    print(f"One-dimensional flame case setup completed at: {case_path}")