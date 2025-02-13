import cantera as ct

class OneDSettings:
    def __init__(
        self,
        mechanism,
        inert_specie,
        simulation_time_step,
        num_of_cells
    ):
        self.mechanism = mechanism
        self.inert_specie = inert_specie
        self.simulation_time_step = simulation_time_step
        self.num_of_cells = num_of_cells

class WorkingCondition(OneDSettings):
    def __init__(
        self,
        one_d_settings,
        tag,
        mechanism,
        temperature,
        pressure,
        fuel_composition,
        oxidizer_composition,
        equivalence_ratio
    ):
        super().__init__(
            mechanism=one_d_settings.mechanism,
            inert_specie=one_d_settings.inert_specie,
            simulation_time_step=one_d_settings.simulation_time_step,
            num_of_cells=one_d_settings.num_of_cells
        )
        self.tag = tag
        self.mechanism = mechanism
        self.temperature = temperature
        self.pressure = pressure
        self.fuel_composition = fuel_composition
        self.oxidizer_composition = oxidizer_composition
        self.equivalence_ratio = equivalence_ratio
        
        self.gas = ct.Solution(self.mechanism)
        self.gas.TP = self.temperature, self.pressure
        self.gas.set_equivalence_ratio(
            phi=self.equivalence_ratio,
            fuel=self.fuel_composition,
            oxidizer=self.oxidizer_composition
        )
        
        self.num_species = self.gas.n_species
        self.species_names = self.gas.species_names
        
    def calculate_flame_speed(self):
        flame_speed_gas = ct.Solution(self.mechanism)
        flame_speed_gas.state = self.gas.state
        
        width = 0.1  # [m]
        flame = ct.FreeFlame(flame_speed_gas, width=width)
        flame.set_refine_criteria(ratio=3, slope=0.07, curve=0.14)
        
        print(f"Solving...")
        # loglevel: amount of diagnostic output (0 to 8)
        flame.solve(loglevel=0, auto=True)
        
        self.laminar_flame_speed = flame.velocity[0]
        
        z= flame.grid
        T = flame.T
        size = len(z)-1
        grad = [0] * size
        for i in range(size):
            grad[i] = (T[i+1]-T[i])/(z[i+1]-z[i])
        self.laminar_flame_thickness = (max(T) -min(T)) / max(grad)
        
        self.chemical_time_scale = self.laminar_flame_thickness / self.laminar_flame_speed
        
        # laminar_flame_speed       : m/s
        # laminar_flame_thickness   : m
        # chemical_time_scale       : s
        return self.laminar_flame_speed, self.laminar_flame_thickness
    
    def set_equilibrium(self):
        equilibrium_gas = ct.Solution(self.mechanism)
        equilibrium_gas.state = self.gas.state
        
        equilibrium_gas.equilibrate('HP')
        
        return equilibrium_gas