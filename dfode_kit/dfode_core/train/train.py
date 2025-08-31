import torch
import numpy as np
import os
import cantera as ct
from dfode_kit.dfode_core.model.mlp import MLP
from dfode_kit.dfode_core.train.formation import formation_calculate
from dfode_kit.utils import BCT
from dfode_kit.data_operations import label_npy
DFODE_ROOT = os.environ['DFODE_ROOT']
def train(
    mech_path: str,
    source_file: str,
    output_path: str,
    time_step: float = 1e-6,
) -> np.ndarray:
    
    labeled_data = label_npy(
    mech_path=mech_path,
    time_step= time_step,
    source_path=source_file,
    )

    gas = ct.Solution(mech_path)
    n_species = gas.n_species
    formation_enthalpies = formation_calculate(mech_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model instantiation
    demo_model = MLP([2+n_species, 400, 400, 400, 400, n_species-1]).to(device)

    # Data loading
    thermochem_states1 = labeled_data[:, 0:2+n_species]
    thermochem_states2 = labeled_data[:, 2+n_species:]

    print(thermochem_states1.shape, thermochem_states2.shape)
    thermochem_states1[:, 2:] = np.clip(thermochem_states1[:, 2:], 0, 1)
    thermochem_states2[:, 2:] = np.clip(thermochem_states2[:, 2:], 0, 1)

    features = torch.tensor(BCT(thermochem_states1), dtype=torch.float32).to(device)
    labels = torch.tensor(BCT(thermochem_states2[:, 2:-1]) - BCT(thermochem_states1[:, 2:-1]), dtype=torch.float32).to(device)

    features_mean = torch.mean(features, dim=0)
    features_std = torch.std(features, dim=0)
    features = (features - features_mean) / features_std

    labels_mean = torch.mean(labels, dim=0)
    labels_std = torch.std(labels, dim=0)
    labels = (labels - labels_mean) / labels_std

    formation_enthalpies = torch.tensor(formation_enthalpies, dtype=torch.float32).to(device)

    # Training
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(demo_model.parameters(), lr=1e-3)

    demo_model.train()  
    for epoch in range(100):
        optimizer.zero_grad()
        preds = demo_model(features)
        loss1 = loss_fn(preds, labels)   ## LOSS  

        Y_in = ((features[:,2:-1]*features_std[2:-1] + features_mean[2:-1])*0.1 + 1)**10
        Y_out = (((preds*labels_std + labels_mean) + (features[:,2:-1]*features_std[2:-1] + features_mean[2:-1]))*0.1 + 1)**10
        Y_target = (((labels*labels_std + labels_mean) + (features[:,2:-1]*features_std[2:-1] + features_mean[2:-1]))*0.1 + 1)**10
        loss2 = loss_fn(Y_out.sum(axis=1), Y_in.sum(axis=1))

        Y_out_total = torch.cat((Y_out, (1-Y_out.sum(axis=1)).reshape(Y_out.shape[0],1)), axis = 1)
        Y_target_total = torch.cat((Y_target, (1-Y_target.sum(axis=1)).reshape(Y_target.shape[0],1)), axis = 1)
        loss3 = loss_fn((formation_enthalpies*Y_out_total).sum(axis=1), (formation_enthalpies*Y_target_total).sum(axis=1))/time_step

        loss = loss1 + loss2 + loss3/1e+13
        loss.backward()
        optimizer.step()
        
        print("Epoch: {}, Loss1: {:4e}, Loss2: {:4e}, Loss3: {:4e}, Loss: {:4e}".format(epoch+1, loss1.item(), loss2.item(), loss3.item(), loss.item()))

    torch.save(
        {
            'net': demo_model.state_dict(),
            'data_in_mean': features_mean.cpu().numpy(),
            'data_in_std': features_std.cpu().numpy(),
            'data_target_mean': labels_mean.cpu().numpy(),
            'data_target_std': labels_std.cpu().numpy(),
        },
        output_path
    )
