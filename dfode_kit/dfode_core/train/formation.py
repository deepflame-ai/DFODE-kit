import cantera as ct
import numpy as np

def formation_calculate(mechanism):
    gas = ct.Solution(mechanism) 
    gas.TPY = 298.15, ct.one_atm,'O2:1' 
    partial_molar_enthalpy = gas.partial_molar_enthalpies/gas.molecular_weights
    print(partial_molar_enthalpy)
    return partial_molar_enthalpy