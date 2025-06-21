from pathlib import Path
from dataclasses import dataclass, field

import cantera as ct

@dataclass
class OneDFreelyPropagatingFlameConfig:
    mechanism: str
    T0: float  
    p0: float     
    fuel: str       
    oxidizer: str   
    eq_ratio: float    
    
    initial_gas: ct.Solution = field(init=False)
    burnt_gas: ct.Solution = field(init=False)
    n_species: int = field(init=False)
    species_names: list = field(init=False)
    n_dims: int = field(init=False)
    dim_names: list = field(init=False)
    mech_path: Path = field(init=False)
    
    flame_speed: float = field(init=False, default=None)
    flame_thickness: float = field(init=False, default=None)
    flame: ct.SolutionArray = field(init=False, default=None)
    
    domain_length: float = field(init=False, default=None)
    domain_width: float = field(init=False, default=None)
    ignition_region: float = field(init=False, default=None)
    sim_time_step: float = field(init=False, default=None)
    sim_time: float = field(init=False, default=None)
    sim_write_interval: float = field(init=False, default=None)
    num_output_steps: int = field(init=False, default=None)
    inlet_speed: float = field(init=False, default=None)
    inert_specie: str = field(init=False, default='N2')
    
    
    def __post_init__(self):
        """Post-initialization to set up initial and burnt gas states."""
        self.initial_gas = ct.Solution(self.mechanism)
        self.initial_gas.TP = self.T0, self.p0
        self.initial_gas.set_equivalence_ratio(self.eq_ratio, self.fuel, self.oxidizer)
        
        self.burnt_gas = ct.Solution(self.mechanism)
        self.burnt_gas.TP = self.T0, self.p0
        self.burnt_gas.set_equivalence_ratio(self.eq_ratio, self.fuel, self.oxidizer)
        self.burnt_gas.equilibrate('HP')
        
        self.n_species = self.initial_gas.n_species
        self.species_names = self.initial_gas.species_names
        self.n_dims = 2 + self.n_species
        self.dim_names = ['T', 'p'] + self.species_names
        
        self.mech_path = Path(self.mechanism).resolve()

    def calculate_laminar_flame_properties(self):
        """Calculate laminar flame speed and thickness."""
        
        # Initialize the gas object
        flame_speed_gas = ct.Solution(self.mechanism)
        flame_speed_gas.TP = self.T0, self.p0
        
        # Set the equivalence ratio
        flame_speed_gas.set_equivalence_ratio(self.eq_ratio, self.fuel, self.oxidizer)

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
        
        self.flame_speed = laminar_flame_speed
        self.flame_thickness = laminar_flame_thickness
        self.flame = final_flame

    def update_config(self, params: dict):
        """Update the configuration with new parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{key}' is not a valid attribute of OneDFreelyPropagatingFlameConfig")
        
        if self.flame_speed is None or self.flame_thickness is None:
            self.calculate_laminar_flame_properties()
        
        if self.domain_length is None:
            self.domain_length = self.flame_thickness / 10 * 500
        
        if self.domain_width is None:
            self.domain_width = self.domain_length / 10
            
        if self.ignition_region is None:
            self.ignition_region = self.domain_length / 2
        
        if self.sim_time_step is None:
            self.sim_time_step = 1e-6
        
        if self.num_output_steps is None:
            self.num_output_steps = 100
        
        if self.sim_write_interval is None:
            chem_time_scale = self.flame_thickness / self.flame_speed
            self.sim_write_interval = chem_time_scale * 10 / self.num_output_steps
        
        if self.sim_time is None:
            self.sim_time = self.sim_write_interval * (self.num_output_steps + 1)
        
        if self.inlet_speed is None:
            self.inlet_speed = self.flame_speed
        
        