import cantera as ct
import numpy as np
import os
import multiprocessing as mp
from multiprocessing import shared_memory

def single_step(npstate):
    gas = ct.Solution("Okafor26.yaml")
    T_old, P_old, Y_old = npstate[0], npstate[1], npstate[2:]
    gas.TPY = T_old, P_old, Y_old
    res_1st = [T_old, P_old] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    r = ct.IdealGasConstPressureReactor(gas, name='R1')
    sim = ct.ReactorNet([r])


    sim.advance(2e-6)
    new_TPY = [gas.T, gas.P] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    res_1st += new_TPY

    return res_1st

def double_step(npstate):
    gas = ct.Solution("Okafor26.yaml")
    T_old, P_old, Y_old = npstate[0], npstate[1], npstate[2:]
    gas.TPY = T_old, P_old, Y_old   
    res_1st = [T_old, P_old] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    r = ct.IdealGasConstPressureReactor(gas, name='R1')
    sim = ct.ReactorNet([r])
    
    sim.advance(1e-6)
    new_TPY = [gas.T, gas.P] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    res_1st += new_TPY

    T_old = gas.T
    P_old = gas.P
    res_2nd = [T_old, P_old]
    Y_old = gas.Y
    res_2nd += list(Y_old) + list(gas.partial_molar_enthalpies/gas.molecular_weights)

    sim.advance(2e-6)
    new_TPY = [gas.T, gas.P] + list(gas.Y) + list(gas.partial_molar_enthalpies/gas.molecular_weights)
    res_2nd += new_TPY
    return [res_1st, res_2nd]


path_var = '/media/mzhang/Expansion/of7/DNNtraining/NH3_H2/1_sampling/'

test = np.load(path_var +'1Dflame.npy')


##------------------------------------------------
##interpolation data
##------------------------------------------------
gas = ct.Solution("Okafor26.yaml")
gas.TPX = 300, 5*ct.one_atm, "O2:1 N2:3.76"
standard_state_lean = [gas.T] + [gas.P] + list(gas.Y)
standard_state_lean = np.array(standard_state_lean)
print(standard_state_lean)

gas.TPX = 300, 5*ct.one_atm, "NH3:0.8038 H2:0.1962 N2:0.0661"
standard_state_rich = [gas.T] + [gas.P] + list(gas.Y)
standard_state_rich = np.array(standard_state_rich)
print(standard_state_rich)

data_PhiLean  = np.load('./phiLimits/1DphiLeanflame.npy')
print(data_PhiLean.shape[0])
addtional_set_lean = []
for i in range(data_PhiLean.shape[0]):
    for lamb in np.arange(0, 1, 0.05):
        temp_T = lamb*300 + (1-lamb)*data_PhiLean[i, 0]
        temp_P = 5*ct.one_atm
        temp_Y = lamb*standard_state_lean[2:] + (1-lamb)*data_PhiLean[i,2:]
        addtional_set_lean.append(np.array([temp_T, temp_P] + list(temp_Y)))
addtional_set_lean = np.array(addtional_set_lean)

data_PhiRich= np.load('./phiLimits/1DphiRichflame.npy')
addtional_set_rich = []
for i in range(data_PhiRich.shape[0]):
    for lamb in np.arange(0, 1, 0.05):
        temp_T = lamb*300 + (1-lamb)*data_PhiRich[i, 0]
        temp_P = 5*ct.one_atm
        temp_Y = lamb*standard_state_rich[2:] + (1-lamb)*data_PhiRich[i,2:]
        addtional_set_rich.append(np.array([temp_T, temp_P] + list(temp_Y)))
addtional_set_rich = np.array(addtional_set_rich)

test = np.concatenate([test,addtional_set_lean], axis=0)
test = np.concatenate([test,addtional_set_rich], axis=0)

np.save(path_var + "CFDdata_interpolation", test)

## -----------------------------------------------
## calculate C/H/O/N ratio for CFD data
## -----------------------------------------------

gas = ct.Solution('Okafor26.yaml')

nSpecies = gas.n_species
n_process = 10

cfd_H_N_ratio = np.zeros(test.shape[0])
cfd_O_N_ratio = np.zeros(test.shape[0])
cfd_eq_ratio = np.zeros(test.shape[0])

cfd_H_molar = np.zeros(test.shape[0])
cfd_O_molar = np.zeros(test.shape[0])
cfd_N_molar = np.zeros(test.shape[0])

for i in range(test.shape[0]):
    
    gas.TPY = test[i, 0], test[i, 1], test[i, 2:nSpecies+2]
    
    cfd_H_molar[i] = gas.elemental_mole_fraction('H')
    cfd_O_molar[i] = gas.elemental_mole_fraction('O')
    cfd_N_molar[i] = gas.elemental_mole_fraction('N')
    cfd_eq_ratio[i] = gas.equivalence_ratio(basis='mole')

cfd_H_N_ratio[:] = cfd_H_molar[:] / cfd_N_molar[:]
cfd_O_N_ratio[:] = cfd_O_molar[:] / cfd_N_molar[:]

print('H/N ratio in CFD:')
print(np.max(cfd_H_N_ratio[:]))
print(np.min(cfd_H_N_ratio[:]))
print(np.mean(cfd_H_N_ratio[:]))

print('O/N ratio in CFD:')
print(np.max(cfd_O_N_ratio[:]))
print(np.min(cfd_O_N_ratio[:]))
print(np.mean(cfd_O_N_ratio[:]))

print('EQ ratio in CFD:')
print(np.max(cfd_eq_ratio[:]))
print(np.min(cfd_eq_ratio[:]))
print(np.mean(cfd_eq_ratio[:]))

## ---------------------------------
## generate random data
## ---------------------------------
maxT = np.max(test[:,0])
minT = np.min(test[:,0])
maxP = np.max(test[:,1])
minP = np.min(test[:,1])
maxAr = np.max(test[:,-1])
minAr = np.min(test[:,-1])
print(maxT,minT,maxAr,minAr)
print(test.shape)


r_n = 1 # random number
test_tmp = np.copy(test)
# test_r = np.copy(test_tmp)
test_r = []
# test_rr = []

alpha = 0.15
beta = 0.2

#H_C_ratio=0.
#O_N_ratio=0.

for count in range(r_n):

    for j in range(test_tmp.shape[0]):

        #print(j)
        
        H_N_ratio = 0.
        O_N_ratio = 0.
        eq_ratio = 0.
        ### random for each point of the test
        k = 0
        while not ((0.035 < H_N_ratio < 2.8 and 0.018 < O_N_ratio < 0.315) and (0.05 < eq_ratio  < 25.0)):
        # while not ((2.65 < H_C_ratio < 4.67 and 0.254 < O_N_ratio < 0.32) and (0.05 < eq_ratio  < 1.75) and test_tmp[j, 0] > 500):
            a = np.random.random()
            #b = np.random.random()
            k = k +1
            # print(k)

            test_tmp[j, 0] = test[j,0] + (maxT - minT)*(2*np.random.random() - 1.0)*alpha
            test_tmp[j, 1] = test[j,1] + (maxP - minP)*(-np.random.random())*alpha*5  #default 10
            # test_tmp[j, -1] = test[j,-1] + (maxAr - minAr)*(2*np.random.random() - 1.0)*alpha

            for i in range(2, test.shape[1] -1):
                test_tmp[j, i] = test[j, i]**(1 + (2*np.random.random() -1)*beta)

            gas.TPY = test_tmp[j, 0], test_tmp[j, 1], test_tmp[j, 2:] 

            H_mole_fraction = gas.elemental_mole_fraction("H")
            O_mole_fraction = gas.elemental_mole_fraction("O")
            N_mole_fraction = gas.elemental_mole_fraction("N")   
            
            eq_ratio = gas.equivalence_ratio(basis='mole')
            # print(af)

            H_N_ratio = H_mole_fraction/N_mole_fraction
            O_N_ratio = O_mole_fraction/N_mole_fraction

            test_tmp[j, 2: -1] = test_tmp[j, 2:-1]/np.sum(test_tmp[j, 2:-1])*(1 - test_tmp[j, -1])
        
            if(k > 41):break
                
        if(k <= 41):test_r.append(test_tmp[j,:])
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

for i in range(test_r.shape[0]):
    
    gas.TPY = test_r[i, 0], test_r[i, 1], test_r[i, 2:nSpecies+2]
 
    r_H_molar[i] = gas.elemental_mole_fraction('H')
    r_O_molar[i] = gas.elemental_mole_fraction('O')
    r_N_molar[i] = gas.elemental_mole_fraction('N')
    r_eq_ratio[i] = gas.equivalence_ratio(basis='mole')

r_H_N_ratio[:] = r_H_molar[:] / r_N_molar[:]
r_O_N_ratio[:] = r_O_molar[:] / r_N_molar[:]

print(np.max(r_H_N_ratio))
print(np.min(r_H_N_ratio))
print(np.mean(r_H_N_ratio))

print(np.max(r_O_N_ratio))
print(np.min(r_O_N_ratio))
print(np.mean(r_O_N_ratio))

print(np.max(r_eq_ratio))
print(np.min(r_eq_ratio))
print(np.mean(r_eq_ratio))

## -------------------------------------------
## calculate DNN input data
## -------------------------------------------

rows_per_process = (test_r.shape[0] + n_process - 1)//n_process
def worker(ii, res):
    for i in range(ii*rows_per_process, (ii+1)*rows_per_process):
        if i >= test_r.shape[0]:
            break
        twores = single_step(test_r[i])
        res.append(twores)

pool = mp.Pool(n_process)
manager = mp.Manager()
res = manager.list()
for ii in range(n_process):
    pool.apply_async(worker, (ii, res))
pool.close()
pool.join()
res = np.array(res)
print(res.shape)

# res = res[np.all(res[:,2*nSpecies+4:3*nSpecies+4]>=0,axis=1) & np.all(res[:,2:2+nSpecies]>=0,axis=1)]
#res_filter = res[(res[:, 2:22] > 0).all(axis=1) & (res[:, 44:64] > 0).all(axis=1)]
res_filter = res[(res[:, 2:2+nSpecies] > 0).all(axis=1) & (res[:, 2*nSpecies+4:3*nSpecies+4] > 0).all(axis=1)]


# np.save("./sampling/" + path_var + "dataset_NegQdot", res_filter)
np.save(path_var + "dataset", res_filter)
print(res_filter.shape)


##-----------------------------------------------
##calculate Qdot
#-----------------------------------------------
# formation_enthalpies = np.load('./formation_enthalpies.npy')

# res_Qdot =[]
# for i in range(res_filter.shape[0]):
#     qdot0 = -(formation_enthalpies*(res_filter[i,4+2*nSpecies:4+3*nSpecies]-res_filter[i,2:2+nSpecies])/1e-6).sum()
#     if (qdot0 > 0.):res_Qdot.append(res_filter[i,:])

# res_Qdot = np.array(res_Qdot)
# np.save("./sampling/" + path_var + "dataset", res_Qdot)
# print(res_Qdot.shape)




#------------------------------------------
#calculate the mean and std of dataset
#------------------------------------------

def BCT(x, lam = 0.1):
    if lam == 0:
        return np.log(x)
    else:
        return (np.power(x, lam) - 1)/lam

res_np = res_filter #res_Qdot 
data_in = res_np[:,:nSpecies+2].copy()
print(np.min(data_in))
data_out = res_np[:,2*nSpecies+4:4+3*nSpecies-1].copy()
print(np.min(data_out))
data_in[:, 2:] = BCT(data_in[:, 2:])
data_out = BCT(data_out)
data_target = data_out - data_in[:,2:-1]

data_in_mean = data_in.mean(axis = 0)
data_in_std = data_in.std(axis = 0, ddof = 1)
data_target_mean = data_target.mean(axis = 0)
data_target_std = data_target.std(axis = 0, ddof = 1)

np.save( path_var + "data_in_mean", data_in_mean)
np.save( path_var + "data_target_mean", data_target_mean)
np.save( path_var + "data_in_std", data_in_std)
np.save( path_var + "data_target_std" , data_target_std)