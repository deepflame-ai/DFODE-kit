import random
import numpy as np
import cantera as ct
from pathlib import Path
from tabulate import tabulate, SEPARATING_LINE
import multiprocessing as mp
from multiprocessing import shared_memory

# =================
# =================
chem = "./Okafor26_noCO2.yaml"
gas = ct.Solution(chem)
n_species = gas.n_species
n_process = 50
time_step = 5e-07

alpha = 0.15
beta = 0.2
gamma = 0.05

min_mix_frac = 0.05
max_mix_frac = 0.95
min_H_N_ratio = 0     ## in air
max_H_N_ratio = 1.04  ## in main jet, plus ~10%
min_O_N_ratio = 0.16  ## in main jet, plus ~10%
max_O_N_ratio = 0.28  ## in air, plus ~10%


def single_step(npstate):
    """
    Perform a single step of the simulation.

    Args:
        npstate (numpy.ndarray): The state of the system as a numpy array. The first element is the temperature (T),
            the second element is the pressure (P), and the remaining elements are the mass fractions of the species (Y).

    Returns:
        list: A list containing the state of the system before and after the step. The first two elements are the
            temperature (T) and pressure (P) before the step, followed by the mass fractions of the species (Y) and the
            partial molar enthalpies divided by the molecular weights after the step.
    """
    gas = ct.Solution(chem)
    T_old, P_old, Y_old = npstate[0], npstate[1], npstate[2:]
    gas.TPY = T_old, P_old, Y_old
    res_1st = [T_old, P_old] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    r = ct.IdealGasConstPressureReactor(gas, name='R1')
    sim = ct.ReactorNet([r])

    sim.advance(time_step)
    new_TPY = [gas.T, gas.P] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    res_1st += new_TPY

    return res_1st


print("Loading 2D Flame Sample Data...\n")
test = np.load('/data/lihan/zhz/NH3_2D/3_dataFilter-Augment/data/2Dflame_vel1.npy')
print(f"Shape of 2D Flame Sample Data:    {test.shape}")
# test = test[np.random.choice(test.shape[0], 1000)]
# print(f"Shape of 2D Flame Sample Data:    {test.shape}")
print("="*50)


## -----------------------------------------------
## Save dataset_cfd.npy
## -----------------------------------------------
test_2Ddata = test.copy()
if test.shape[0] > 100000:
    test_2Ddata = test[np.random.choice(test.shape[0], 100000)]

test_n_rows = test_2Ddata.shape[0]
shm = shared_memory.SharedMemory(create=True, size =8*test_n_rows*(4+4*n_species))
cantera_out = np.ndarray((test_2Ddata.shape[0],4+4*n_species), dtype=np.float64, buffer=shm.buf)
rows_per_process = (test_n_rows + n_process - 1)//n_process

def worker_evolution(ii):
    for i in range(ii*rows_per_process,(ii+1)*rows_per_process):
        cantera_out[i] = np.array((single_step(test_2Ddata[i, :2+n_species])))
pool = mp.Pool(n_process)
for ii in range(n_process):
    pool.apply_async(worker_evolution, (ii,))
pool.close()
pool.join()

np.save('dataset_2Dflame', cantera_out)
print(f"Shape of dataset_2Dflame:    {cantera_out.shape}")

shm.close()
shm.unlink()


## -----------------------------------------------
## calculate C/H/O/N ratio for CFD data
## -----------------------------------------------
fuel = 'NH3:0.4,H2:0.45,N2:0.15'
main_jet = 'NH3:0.2031, H2:0.2553, O2:0.096, N2:0.4456, AR:0.0046'
air = 'N2:0.78,O2:0.21,AR:0.01'

cfd_H_N_ratio = np.zeros(test.shape[0])
cfd_O_N_ratio = np.zeros(test.shape[0])
cfd_eq_ratio = np.zeros(test.shape[0])
cfd_Z = np.zeros(test.shape[0])

cfd_H_molar = np.zeros(test.shape[0])
cfd_O_molar = np.zeros(test.shape[0])
cfd_N_molar = np.zeros(test.shape[0])

filter_by_equivalence_ratio = []
filter_by_Z = []

for i in range(test.shape[0]):
    
    gas.TPY = test[i, 0], test[i, 1], test[i, 2:]
    
    cfd_H_molar[i] = gas.elemental_mole_fraction('H')
    cfd_O_molar[i] = gas.elemental_mole_fraction('O')
    cfd_N_molar[i] = gas.elemental_mole_fraction('N')
    cfd_eq_ratio[i] = gas.equivalence_ratio(fuel=fuel, oxidizer=air, basis='mole')
    cfd_Z[i] = gas.mixture_fraction(fuel=main_jet, oxidizer=air, basis='mole')
    
    if min_mix_frac <= cfd_Z[i] and cfd_Z[i] <= max_mix_frac:
        filter_by_Z.append(i)

test_filtered_by_Z = test[filter_by_Z, :]
print(test.shape)
print(test_filtered_by_Z.shape)

cfd_H_N_ratio[:] = cfd_H_molar[:] / cfd_N_molar[:]
cfd_O_N_ratio[:] = cfd_O_molar[:] / cfd_N_molar[:]

table = [
    [              '',                'CFD data'],
    ['H/N ratio: max',  np.max(cfd_H_N_ratio[:])],
    [           'min',  np.min(cfd_H_N_ratio[:])],
    [          'mean', np.mean(cfd_H_N_ratio[:])],
    [],
    ['O/N ratio: max',  np.max(cfd_O_N_ratio[:])],
    [           'min',  np.min(cfd_O_N_ratio[:])],
    [          'mean', np.mean(cfd_O_N_ratio[:])],
    [],
    [ 'EQ ratio: max',   np.max(cfd_eq_ratio[:])],
    [           'min',   np.min(cfd_eq_ratio[:])],
    [          'mean',  np.mean(cfd_eq_ratio[:])],
    [ 'Z       : max',   np.max(cfd_Z[:])],
    [           'min',   np.min(cfd_Z[:])],
    [          'mean',  np.mean(cfd_Z[:])],
]

print(tabulate(table, colalign=("right","decimal"), headers="firstrow", floatfmt=".4f"))

np.save("2Dflame_filtered_by_Z", test_filtered_by_Z)

# exit()

## ---------------------------------
## generate randomly augmented data
## ---------------------------------
test = test_filtered_by_Z
# test = test[np.random.choice(test.shape[0], 400000)]
maxT = np.max(test[:,0])
minT = np.min(test[:,0])
maxP = np.max(test[:,1])
minP = np.min(test[:,1])
maxAr = np.max(test[:,-1])
minAr = np.min(test[:,-1])
N2_index = gas.species_index('N2')+2
max_N2 = np.max(test[:,N2_index])
min_N2 = np.min(test[:,N2_index])
print(maxT,minT,maxAr,minAr)
print(max_N2,min_N2)
print(test.shape)

r_n = 6 # random number
test_tmp = np.copy(test)
# test_r = np.copy(test_tmp)
test_r = []
# test_rr = []

#H_C_ratio=0.
#O_N_ratio=0.

for count in range(r_n):
    print(count)

    for j in range(test_tmp.shape[0]):

        #print(j)
        
        H_N_ratio = 0.
        O_N_ratio = 0.
        eq_ratio = 0.
        mix_frac = 0.
        ### random for each point of the test
        k = 0
        while not ((min_H_N_ratio < H_N_ratio < max_H_N_ratio and min_O_N_ratio < O_N_ratio < max_O_N_ratio) \
            and (min_mix_frac < mix_frac) and mix_frac <= max_mix_frac and (minT * (1 - gamma)) <= test_tmp[j, 0] <= (maxT * (1 + gamma))):
            a = np.random.random()
            #b = np.random.random()
            k = k + 1
            # print(k)

            test_tmp[j, 0] = test[j,0] + (maxT - minT)*(2*np.random.random() - 1.0)*alpha
            test_tmp[j, 1] = test[j,1] + (maxP - minP)*(2*np.random.random() - 1.0)*alpha
            # test_tmp[j, -1] = test[j,-1] + (maxAr - minAr)*(2*np.random.random() - 1.0)*alpha

            # for i in range(2, test.shape[1] -1):
            #     test_tmp[j, i] = test[j, i]**(1 + (2*np.random.random() -1)*beta)

            for i in range(2, test.shape[1]):
                test_tmp[j, i] = np.abs(test[j, i])**(1+(2*np.random.random() - 1.0)*alpha)
                
            test_tmp[j, N2_index] = test[j, N2_index] + (max_N2 - min_N2)*(2*np.random.random() - 1.0)*alpha
            test_tmp[j, 2: -1] = test_tmp[j, 2:-1]/np.sum(test_tmp[j, 2:-1])*(1 - test_tmp[j, -1])
            
            gas.TPY = test_tmp[j, 0], test_tmp[j, 1], test_tmp[j, 2:] 

            H_mole_fraction = gas.elemental_mole_fraction("H")
            O_mole_fraction = gas.elemental_mole_fraction("O")
            N_mole_fraction = gas.elemental_mole_fraction("N")   
            
            eq_ratio = gas.equivalence_ratio(fuel=fuel, oxidizer=air, basis='mole')
            mix_frac = gas.mixture_fraction(fuel=main_jet, oxidizer=air, basis='mole')

            H_N_ratio = H_mole_fraction/N_mole_fraction
            O_N_ratio = O_mole_fraction/N_mole_fraction
        
            if(k > 20):break
                
        if(k <= 20):test_r.append(test_tmp[j,:])
        # print(test_r[0])
    # test_tmp[:, 2: -1] = test_tmp[:, 2:-1]/np.sum(test_tmp[:, 2:-1],axis=1)[:,np.newaxis]*(1 - test_tmp[:, -1])[:, np.newaxis]
    
    # test_r = test_tmp

    # test_r = np.concatenate((test_r, test_tmp), axis=0)

test_r = np.array(test_r) 
print(test_r.shape)

r_H_N_ratio = np.zeros(test_r.shape[0])
r_O_N_ratio = np.zeros(test_r.shape[0])
r_eq_ratio = np.zeros(test_r.shape[0])

r_H_molar = np.zeros(test_r.shape[0])
r_O_molar = np.zeros(test_r.shape[0])
r_N_molar = np.zeros(test_r.shape[0])

np.save('2Dflame_random',test_r)
# if test_r.shape[0] > 1000000:
#     test_r = test_r[np.random.choice(test_r.shape[0], 1000000)]
# np.save('2Dflame_random_100w',test_r)
print(test_r.shape[0])

# exit()
## -----------------------------------------------
## Generate dataset.npy
## -----------------------------------------------
final_test = np.load('./2Dflame_random.npy')
#final_test = final_test[np.random.choice(final_test.shape[0], 1000000)]
print(f"final_test.shape {final_test.shape}")

augmentSet = final_test.copy()

# cantera2_out = np.ndarray((augmentSet.shape[0],4+4*n_species), dtype=np.float64)

# for i in range(augmentSet.shape[0]):
#     cantera2_out[i] = np.array(single_step(augmentSet[i, :2 + n_species]))

# # 打印输出数组的形状
# print(cantera2_out.shape)

test_n_rows = augmentSet.shape[0]
shm1 = shared_memory.SharedMemory(create=True, size =8*test_n_rows*(4+4*n_species))
cantera_out2 = np.ndarray((augmentSet.shape[0],4+4*n_species), dtype=np.float64, buffer=shm1.buf)
rows_per_process = (test_n_rows + n_process - 1)//n_process

def worker_evolution(ii):
    for i in range(ii*rows_per_process,(ii+1)*rows_per_process):
        cantera_out2[i] = np.array((single_step(augmentSet[i, :2+n_species])))
pool = mp.Pool(n_process)
for ii in range(n_process):
    pool.apply_async(worker_evolution, (ii,))
pool.close()
pool.join()

np.save('dataset_vel1', cantera_out2)
print(f"Shape of dataset.npy:    {cantera_out2.shape}")

# if cantera_out2.shape[0] > 1000000:
#     cantera_out2 = cantera_out2[np.random.choice(cantera_out2.shape[0], 1000000)]

# np.save('dataset_100w', cantera_out2)
# print(f"Shape of dataset_100w.npy:    {cantera_out2.shape}")


shm1.close()
shm1.unlink()
