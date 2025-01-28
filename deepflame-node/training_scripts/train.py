import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import cantera as ct

chem = './HyChem41s.yaml'
gas = ct.Solution(chem)
n_species = gas.n_species
time_step = 1e-7

def BCT(x, lam = 0.1):
    if lam == 0:
        return np.log(x)
    else:
        return (np.power(x, lam) - 1)/lam

class NN_MLP(nn.Module):
    def __init__(self, layer_info):
        super(NN_MLP, self).__init__()
        self.net = nn.Sequential()
        n = len(layer_info) - 1
        for i in range(n - 1):
            self.net.add_module('linear_layer_%d' %(i), nn.Linear(layer_info[i], layer_info[i + 1]))
            self.net.add_module('gelu_layer_%d' %(i), nn.GELU())
        self.net.add_module('linear_layer_%d' %(n - 1), nn.Linear(layer_info[n - 1], layer_info[n]))
        
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    device = "cuda:3"
    res_np = np.load('./dataset.npy')
    formation_enthalpies = np.load('./formation_enthalpies.npy')
    formation_enthalpies = torch.tensor(formation_enthalpies).float().to(device)
    print("total sample shape: {}".format(res_np.shape))

    h_i_in = res_np[:, 2+n_species:2+2*n_species]
    h_i_out = res_np[:, 4+3*n_species:4+4*n_species]

    data_in = np.abs(res_np[:,:2+n_species]).copy()
    data_out = np.abs(res_np[:,4+2*n_species:4+3*n_species-1]).copy()

    data_in[:, 2:] = BCT(data_in[:, 2:])
    data_out = BCT(data_out)
    data_target = data_out - data_in[:,2:-1]

    data_in_mean = data_in.mean(axis = 0)
    data_in_std = data_in.std(axis = 0, ddof = 1)
    data_target_mean = data_target.mean(axis = 0)
    data_target_std = data_target.std(axis = 0, ddof = 1)

    nn_in_mean = torch.tensor(data_in_mean).float().to(device)
    nn_in_std = torch.tensor(data_in_std).float().to(device)
    nn_target_mean = torch.tensor(data_target_mean).float().to(device)
    nn_target_std = torch.tensor(data_target_std).float().to(device)
    
    nn_input = (data_in - data_in_mean)/data_in_std
    nn_target = (data_target - data_target_mean)/data_target_std

    nn_target = np.concatenate((nn_target, h_i_in, h_i_out), axis=1) 
    print("nn_target shape: {}".format(nn_target.shape))

    #H_conservation = torch.tensor([1., 0., 2., 1., 0., 2.]).float().to(device)
    #O_conservation = torch.tensor([0., 1., 1., 1., 2., 0.]).float().to(device)

    nn_input_t = torch.tensor(nn_input).float()
    nn_target_t = torch.tensor(nn_target).float()
    
    layers = [2+n_species, 400, 200, 100, 1]
    modellist = []
    for i in range(n_species-1):
        modellist.append(NN_MLP(layers))
#    for i in range(6):
#        modellist[i].load_state_dict(torch.load(f'Temporary_Chemical_{i}.pt')['state_dict'])
    for i in range(n_species-1):
        modellist[i] = modellist[i].to(device)
    #model.load_state_dict(torch.load('Temporary_Chemical.pt')['state_dict'])
    #model = model.to(device)

    batch_size = 4000
    tensor_data_set = TensorDataset(nn_input_t, nn_target_t)
    data_loader = DataLoader(tensor_data_set, batch_size=batch_size, num_workers=10, shuffle=True)

    max_epoch = int(2e3)
    criterion = nn.L1Loss()
    optim_lr = 1e-3
    optim = []
    for i in range(n_species-1):
        optim.append(torch.optim.Adam(modellist[i].parameters(), lr=optim_lr))
    print_interval = 20
    save_interval = 400

    loss_total = []
    last_time = time.time()
    for epoch in range(max_epoch):
        for batch_input, batch_target in data_loader:
            batch_input_t = batch_input.to(device)
            batch_target_t = batch_target.to(device)
            result = []
            for i in range(n_species-1):
                result.append(modellist[i](batch_input_t))
            result = torch.cat(result,dim=1)
            loss1 = criterion(result, batch_target_t[:,:n_species-1])
            Y_in = ((batch_input_t[:,2:-1]*nn_in_std[2:-1] + nn_in_mean[2:-1])*0.1 + 1)**10
            Y_out = (((result*nn_target_std + nn_target_mean) + (batch_input_t[:,2:-1]*nn_in_std[2:-1] + nn_in_mean[2:-1]))*0.1 + 1)**10
            Y_target = (((batch_target_t[:,:n_species-1]*nn_target_std + nn_target_mean) + (batch_input_t[:,2:-1]*nn_in_std[2:-1] + nn_in_mean[2:-1]))*0.1 + 1)**10
            loss2 = criterion(Y_out.sum(axis=1), Y_in.sum(axis=1))
            Y_in_total = torch.cat((Y_in, (1-Y_in.sum(axis=1)).reshape(Y_in.shape[0],1)), axis = 1)
            Y_out_total = torch.cat((Y_out, (1-Y_out.sum(axis=1)).reshape(Y_out.shape[0],1)), axis = 1)
            Y_target_total = torch.cat((Y_target, (1-Y_target.sum(axis=1)).reshape(Y_target.shape[0],1)), axis = 1)
            loss3 = criterion((batch_target_t[:,-n_species:]*Y_out_total).sum(axis=1), (batch_target_t[:,-2*n_species:-n_species]*Y_in_total).sum(axis=1))/time_step
            loss6 = criterion((formation_enthalpies*Y_out_total).sum(axis=1), (formation_enthalpies*Y_target_total).sum(axis=1))/time_step
            #loss4 = (((H_conservation*(Y_out - Y_in)).sum(axis=1))**2).mean()
            #loss5 = (((O_conservation*(Y_out - Y_in)).sum(axis=1))**2).mean()
            loss = loss1 + loss2 + loss3/1e13 + loss6/1e13
            for i in range(n_species-1):
                optim[i].zero_grad()
            loss.backward()
            for i in range(n_species-1):
                optim[i].step()
    
        
        loss_total.append(loss.item())
        if (epoch + 1) % print_interval == 0:
            print('epoch:{} , loss1:{:.4e}, loss2:{:.4e}, loss3:{:.4e}, loss6:{:.4e}, time: {}m'.format(epoch + 1, 
                loss1.item(), loss2.item(), loss3.item(), loss6.item(), (time.time() - last_time)/60))
            last_time = time.time()
    
        if (epoch + 1) % save_interval == 0:
            for i in range(n_species-1):
                torch.save({'state_dict': modellist[i].state_dict()}, f'Temporary_Chemical_{i}.pt')
            optim_lr *= 0.1
            optim = []
            for i in range(n_species-1):
                optim.append(torch.optim.Adam(modellist[i].parameters(), lr=optim_lr))
    

    loss_total = np.array(loss_total)
    np.save('loss', loss_total)
