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
    
    """
    Here is a simple demo of train script.

    Trains a neural network model to predict changes in thermochemical states based on input data.

    This function loads labeled data from a specified source file, initializes a chemical reaction model,
    and constructs a multi-layer perceptron (MLP) for training. The model learns to predict the changes 
    in species concentrations over time based on the input features. The training process includes 
    normalization of input and output data, computation of multiple loss functions, and optimization of 
    the model parameters.

    Parameters
    ----------
    mech_path : str
        Path to the mechanism file for the chemical model.
    source_file : str
        Path to the input data file containing labeled data.
    output_path : str
        Path to save the trained model and normalization parameters.
    time_step : float, optional
        Time step for the simulation, default is 1e-06 second.

    Returns
    -------
    np.ndarray
        Returns the trained model's output as a numpy array (if applicable).
    """

    labeled_data = np.load(source_file)

    gas = ct.Solution(mech_path)
    n_species = gas.n_species
    formation_enthalpies = formation_calculate(mech_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model instantiation (Use Float Precision for Speed)
    demo_model = MLP([2+n_species, 400, 400, 400, 400, n_species-1]).float().to(device)

    # Data loading
    thermochem_states1 = labeled_data[:, 0:2+n_species]
    thermochem_states2 = labeled_data[:, 2+n_species:]

    print(thermochem_states1.shape, thermochem_states2.shape)
    
    # Align with reference: Use np.abs to ensure non-negativity for all inputs
    thermochem_states1 = np.abs(thermochem_states1)
    thermochem_states2 = np.abs(thermochem_states2)

    # Apply BCT only to species (columns 2 onwards), keep T/P unchanged
    states_bct = thermochem_states1.copy()
    states_bct[:, 2:] = BCT(states_bct[:, 2:])
    # Use float32 for training speed
    features = torch.tensor(states_bct, dtype=torch.float32).to(device)
    
    # Labels: Delta of BCT-transformed species
    labels = torch.tensor(BCT(thermochem_states2[:, 2:-1]) - BCT(thermochem_states1[:, 2:-1]), dtype=torch.float32).to(device)

    features_mean = torch.mean(features, dim=0)
    features_std = torch.std(features, dim=0)
    # Prevent division by zero
    features_std[features_std == 0] = 1.0
    features = (features - features_mean) / features_std

    labels_mean = torch.mean(labels, dim=0)
    labels_std = torch.std(labels, dim=0)
    # Prevent division by zero
    labels_std[labels_std == 0] = 1.0
    labels = (labels - labels_mean) / labels_std

    formation_enthalpies = torch.tensor(formation_enthalpies, dtype=torch.float32).to(device)

    # Pre-slice constants for optimization
    f_std_species = features_std[2:-1]
    f_mean_species = features_mean[2:-1]

    # Training
    loss_fn = torch.nn.L1Loss()

    demo_model.train()  
    max_epochs = 1500
    initial_lr = 0.001
    lr_decay_epoch = 500
    batch_size = 20000
    optimizer = torch.optim.Adam(demo_model.parameters(), lr=initial_lr)


    for epoch in range(max_epochs):
        if epoch > 0 and epoch % lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        # 初始化损失值
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0
        total_loss3 = 0
        total_batches = 0

        for i in range(0, len(features), batch_size):
            batch_features = features[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            optimizer.zero_grad()

            preds = demo_model(batch_features)
            loss1 = loss_fn(preds, batch_labels)  

            # Optimization: Extract common sub-expression (Input species in physical space after Inverse BCT base calc)
            # Formula: ((NormInput * Std + Mean) * lambda + 1) ** (1/lambda)
            # Here lambda=0.1, so power is 10.
            
            # 1. De-normalize input species part
            input_species_part = batch_features[:, 2:-1] * f_std_species + f_mean_species
            
            # 2. Calculate common base term for Inverse BCT
            base_term_in = input_species_part * 0.1 + 1
            
            # 3. Y_in (Physical Species Input)
            Y_in = base_term_in ** 10
            
            # 4. Y_out (Predicted Physical Species)
            # Preds are delta. So Y_out_bct = Pred_unnorm + Input_bct
            # Y_out = (Y_out_bct * 0.1 + 1) ** 10
            preds_unnorm = preds * labels_std + labels_mean
            Y_out = ((preds_unnorm + input_species_part) * 0.1 + 1) ** 10
            
            # 5. Y_target (Target Physical Species for consistency check)
            labels_unnorm = batch_labels * labels_std + labels_mean
            Y_target = ((labels_unnorm + input_species_part) * 0.1 + 1) ** 10

            loss2 = loss_fn(Y_out.sum(axis=1), Y_in.sum(axis=1))

            Y_out_total = torch.cat((Y_out, (1 - Y_out.sum(axis=1)).reshape(Y_out.shape[0], 1)), axis=1)
            Y_target_total = torch.cat((Y_target, (1 - Y_target.sum(axis=1)).reshape(Y_target.shape[0], 1)), axis=1)

            loss3 = loss_fn((formation_enthalpies * Y_out_total).sum(axis=1), (formation_enthalpies * Y_target_total).sum(axis=1)) / time_step
            loss = loss1 + loss2 + loss3 / 1e+13

            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        total_loss1 /= (len(features) / batch_size)
        total_loss2 /= (len(features) / batch_size)
        total_loss3 /= (len(features) / batch_size)
        total_loss /= (len(features) / batch_size)

        print("Epoch: {}, Loss1: {:4e}, Loss2: {:4e}, Loss3: {:4e}, Loss: {:4e}".format(epoch+1, total_loss1, total_loss2, total_loss3, total_loss))

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
