import random
import cantera as ct
import numpy as np

import multiprocessing as mp
from multiprocessing import shared_memory

from tabulate import tabulate, SEPARATING_LINE
# import tqdm


print("="*80)
chem = 'Okafor26_noCO2.yaml'
gas = ct.Solution(chem)
n_species = gas.n_species
n_process = 50
time_step = 5e-7

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

def double_step(npstate):
    gas = ct.Solution(chem)
    T_old, P_old, Y_old = npstate[0], npstate[1], npstate[2:]
    gas.TPY = T_old, P_old, Y_old   
    res_1st = [T_old, P_old] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    r = ct.IdealGasConstPressureReactor(gas, name='R1')
    sim = ct.ReactorNet([r])
    
    sim.advance(time_step)
    new_TPY = [gas.T, gas.P] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    res_1st += new_TPY

    T_old = gas.T
    P_old = gas.P
    res_2nd = [T_old, P_old]
    Y_old = gas.Y
    res_2nd += list(Y_old) + list(gas.partial_molar_enthalpies/gas.molecular_weights)

    sim.advance(2*time_step)
    new_TPY = [gas.T, gas.P] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    res_2nd += new_TPY
    return [res_1st, res_2nd]

print("Loading 1D Flame Sample Data...\n")
test = np.load('./1Dflame.npy')
test = test[np.random.choice(test.shape[0], 600000)]
print(f"Shape of 1D Flame Sample Data:    {test.shape}")
print("="*80)

# ================================================================================


# Single Step Cantera Simulation
# ================================================================================

randomset2 = test.copy()

test_n_rows = test.shape[0]
shm = shared_memory.SharedMemory(create=True, size =8*test_n_rows*(4+4*n_species))
cantera_out = np.ndarray((test.shape[0],4+4*n_species), dtype=np.float64, buffer=shm.buf)
rows_per_process = (test_n_rows + n_process - 1)//n_process

def worker_evolution(ii):
    for i in range(ii*rows_per_process,(ii+1)*rows_per_process):
        cantera_out[i] = np.array((single_step(randomset2[i, :2+n_species])))
        
print("Performing Single Step Cantera Simulation...\n")
pool = mp.Pool(n_process)
for ii in range(n_process):
    pool.apply_async(worker_evolution, (ii,))
pool.close()
pool.join()
print(f"Shape of Sample Data After Single Step Cantera Simulation:    {cantera_out.shape}")

np.save('dataset_cfd', cantera_out)
print("="*80)

# ================================================================================

## -------------------------------------------
## calculate Qdot for CFD data
## -------------------------------------------
#qdot_noram0 = []
#qdot_noram1 = []

formation_enthalpies = np.load('./formation_enthalpies.npy')

test_n_rows = test.shape[0]
shm1 = shared_memory.SharedMemory(create=True, size =8*test_n_rows)
shm2 = shared_memory.SharedMemory(create=True, size =8*test_n_rows)
qdot_noram0 = np.ndarray((test_n_rows,), dtype=np.float64, buffer=shm1.buf)
qdot_noram1 = np.ndarray((test_n_rows,), dtype=np.float64, buffer=shm2.buf)
rows_per_process = (test_n_rows + n_process - 1)//n_process
# def worker_calc_qdot(ii):
#     for i in range(ii*rows_per_process, (ii+1)*rows_per_process):
#         if i > test_n_rows:
#             continue
#         twores_noram = double_step(test[i])
#         twores_noram_np0 = np.array(twores_noram[0])
#         twores_noram_np1 = np.array(twores_noram[1])
#         qdot_noram0[i] = (-(formation_enthalpies*(twores_noram_np0[4+2*n_species:4+3*n_species]-twores_noram_np0[2:2+n_species])/time_step).sum())
#         qdot_noram1[i] = (-(formation_enthalpies*(twores_noram_np1[4+2*n_species:4+3*n_species]-twores_noram_np1[2:2+n_species])/time_step).sum())

def worker_calc_qdot(ii):
    for i in range(ii*rows_per_process, (ii+1)*rows_per_process):
        if i > test_n_rows:
            continue
        twores_noram = single_step(test[i])
        twores_noram_np0 = np.array(twores_noram)
        qdot_noram0[i] = (-(formation_enthalpies*(twores_noram_np0[4+2*n_species:4+3*n_species]-twores_noram_np0[2:2+n_species])/time_step).sum())

pool = mp.Pool(n_process)
for ii in range(n_process):
    pool.apply_async(worker_calc_qdot, (ii,))
pool.close()
pool.join()

# ---------------------------------
# calculate Qdot for random data and generate random data for each sample
# ---------------------------------
maxT = np.max(test[:,0])
minT = np.min(test[:,0])
maxP = np.max(test[:,1])
minP = np.min(test[:,1])
maxN2 = np.max(test[:,-1])
minN2 = np.min(test[:,-1])

table = [['Max Temperature', maxT, 'K'],
         ['Min Temperature', minT, 'K'],
         ['Max Pressure', maxP, 'Pa'],
         ['Min Pressure', minP, 'Pa'],
         ['Max Mass Fraction', maxN2, '-'],
         ['Min Mass Fraction', minN2, '-']]
print(tabulate(table, tablefmt="rst", colalign=("left","right", "left")))
print(f"Shape of Sample Data Before Data Augmentation:    {test.shape}")



print("Performing Random Data Generation...\n")
r_n         = 5 # random number
test_tmp    = np.copy(test[0])
alpha       = 0.15

cq          = 200 * alpha
    
rows_per_process = (test.shape[0] + n_process - 1)//n_process

# pbar = tqdm.tqdm(total=r_n*n_process)

def worker_evo_sel(ii, res):
    for count_r in range(r_n):
        # print(f"Random Data Generation:  Round {count_r+1}/{r_n} for Process {ii+1}...")
        for i in range(ii*rows_per_process, (ii+1)*rows_per_process):
            if(i % 10000 == 0 and ii == 0): print(i)      # print progress?
            if i > test.shape[0]:
                break

            test_r = test[i]

            qdot_tmp = 0
            count = 0
            while not (qdot_tmp > 1/cq*qdot_noram0[i] and qdot_tmp < cq*qdot_noram0[i]):

                count = count + 1

                if count > 10:
                    break

                test_tmp[0] = test_r[0] + (maxT - minT)*(2*np.random.rand() - 1.0)*alpha
                test_tmp[1] = test_r[1] + (maxP - minP)*(2*np.random.rand() - 1.0)*alpha*20
                # test_tmp[-1] = test_r[-1] + (maxN2 - minN2)*(2*np.random.rand() - 1)*alpha

                for j in range(2, test.shape[1] -1):
                    test_tmp[j] = np.abs(test_r[j])**(1 + (2*np.random.rand() -1)*alpha)

                test_tmp[2: -1] = test_tmp[2:-1]/np.sum(test_tmp[2:-1])*(1 - test_tmp[-1])
                
                

                test_tmp2 = single_step(test_tmp)
                test_tmp2_np = np.array(test_tmp2)
                qdot_tmp = (-(formation_enthalpies*(test_tmp2_np[4+2*n_species:4+3*n_species]-test_tmp2_np[2:2+n_species])/time_step).sum())

            if count < 10:
                res.append(test_tmp2)
        
        # pbar.update(1)

pool = mp.Pool(n_process)
manager = mp.Manager()
res = manager.list()
for ii in range(n_process):
    pool.apply_async(worker_evo_sel, (ii, res))
pool.close()
pool.join()

res = np.array(res)
if res.shape[0] > 400000:
    res = res[np.random.choice(res.shape[0], 400000)]
print(f"res.shape {res.shape}")
np.save('dataset_unfilter', res)

res_filter = []
for i in range(res.shape[0]):
    if (res[i, 0] > 500 and res[i, 2+2*n_species] < 2600):
        res_filter.append(res[i, :])

res_filter = np.array(res_filter)
print(f"res_filter.shape {res_filter.shape}")
res_filter.shape

np.save('dataset', res_filter)

shm.close()
shm.unlink()
shm1.close()
shm1.unlink()
shm2.close()
shm2.unlink()
