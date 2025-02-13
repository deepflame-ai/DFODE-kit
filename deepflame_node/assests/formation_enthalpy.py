import numpy as np
import cantera as ct

def get_formation_enthalpies(mechanism):
    gas = ct.Solution(mechanism)
    gas.TPY = 298.15, ct.one_atm, "O2:1"
    formation_h = gas.partial_molar_enthalpies/gas.molecular_weights
    return formation_h

if __name__ == '__main__':
    formation_h = get_formation_enthalpies('Okafor2018_s59r356.yaml')
    np.save('formation_enthalpies', formation_h)