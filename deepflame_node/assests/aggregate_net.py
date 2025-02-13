import torch
import numpy as np
import cantera as ct

gas = ct.Solution('./HyChem41s.yaml')
n_species = gas.n_species

def BCT(x, lam = 0.1):
    if lam == 0:
        return np.log(x)
    else:
        return (np.power(x, lam) - 1)/lam

if __name__ == '__main__':

    # calculate mean and std
    test = np.load('./dataset.npy')
    res_np = test
    data_in = np.abs(res_np[:,:2+n_species]).copy()
    data_out = np.abs(res_np[:,4+2*n_species:4+3*n_species-1]).copy()
    data_in[:, 2:] = BCT(data_in[:, 2:])
    data_out = BCT(data_out)
    data_target = data_out - data_in[:,2:-1]

    data_in_mean = data_in.mean(axis = 0)
    data_in_std = data_in.std(axis = 0, ddof = 1)
    data_target_mean = data_target.mean(axis = 0)
    data_target_std = data_target.std(axis = 0, ddof = 1)
    np.save('data_in_mean', data_in_mean)
    np.save('data_target_mean', data_target_mean)
    np.save('data_in_std', data_in_std)
    np.save('data_target_std', data_target_std)

    # aggregate DNN models
    aggregate_net_dict = {}

    data_in_mean = np.load('./data_in_mean.npy')
    data_in_std = np.load('./data_in_std.npy')
    data_target_mean = np.load('./data_target_mean.npy')
    data_target_std = np.load('./data_target_std.npy')

    aggregate_net_dict['data_in_mean'] = data_in_mean
    aggregate_net_dict['data_in_std'] = data_in_std
    aggregate_net_dict['data_target_mean'] = data_target_mean
    aggregate_net_dict['data_target_std'] = data_target_std

    for i in range(n_species-1):
        aggregate_net_dict[f'net{i}'] = (torch.load(f'Temporary_Chemical_{i}.pt',map_location='cpu'))['state_dict']

    torch.save(aggregate_net_dict, 'DNN_model.pt')


