import os
import numpy as np
import xarray as xr
import glob

from causaldynamics.score import score
from pathlib import Path
import torch

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt 

from causaldynamics.baselines import StructuredODEDiscovery


if __name__ == "__main__":

    loss = 'l1'

    assert torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert device.type == "cuda", "CUDA is not available. Please check your PyTorch installation and GPU configuration."

    tau_max = 1 # i.e. no delay 
    hyperparam_search_array = np.array([0.001, 0.0025, 0.005,  0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5])
    model_name = f"ode_sparse_discovery_{loss}"

    auroc_results = np.zeros((len(hyperparam_search_array), 1))
    auprc_results = np.zeros((len(hyperparam_search_array), 1))
    shd_results = np.zeros((len(hyperparam_search_array), 1))


    DATA_DIR = Path("../data/simple/noise=0.00_confounder=False/data")  # (you can change this to your own path)
    ds = xr.open_dataset(DATA_DIR / "Lorenz84_N10_T1000.nc")

    # # Extract timeseries and adjacency matrix as target # If not coupled
    timeseries = ds['time_series'].to_numpy().transpose(1, 0, 2) # shape of (N, T, D)
    adj_matrix = ds['adjacency_matrix'].to_numpy()

    n_systems = adj_matrix.shape[0]

    timeseries_stacked = timeseries[:, :, None]

    for j, lambda_m_hp in enumerate(hyperparam_search_array):
        print(f"Running hyperparameter search for lambda_m = {lambda_m_hp}")
        lowrank_adj_matrix = []
        for i in range(3):
            for x in timeseries_stacked:
                lowrank_model = StructuredODEDiscovery(
                    coupled=False,
                    t = tau_max,
                    n = n_systems,
                    L_lipschitz = 1, 
                    is_lipschitz = False,
                    hidden_dim = 8, 
                    n_layers = 2,
                    device = device,
                    beta_init = 0.33, 
                    beta_min = 0.33,
                    annealing_rate = 0.95,
                    annealing_epochs = 10,
                    normalize_grad = True,
                )
                trained_model, mask = lowrank_model.run(n_samples=3, lambda_grad=100, X=torch.tensor(x).to(device), l0_l1_l2=loss, th=0.5, lr=0.001, lambda_m = lambda_m_hp, n_inner_min_sparse = 100, batch_size=128, n_inner=4_000, patience = 200, save_fig_path="loss_hyperparam_search.png")
                lowrank_adj_matrix.append(mask.detach().cpu().numpy())
        
        results_scores = score(
            preds= np.array(lowrank_adj_matrix)[:, 0], #.transpose((0, 2, 1)), # Need to transpose
            labs= adj_matrix,
            name=model_name
        )

        auroc_results[j] = results_scores.loc["Joint AUROC", model_name]
        auprc_results[j] = results_scores.loc["Joint AUPRC", model_name]
        shd_results[j] = results_scores.loc["Joint SHD", model_name]

        print(f"Results scores for lambda_m = {lambda_m_hp}")
        print(results_scores)

    np.save(f"auroc_results_{loss}.npy", auroc_results)
    np.save(f"auprc_results_{loss}.npy", auprc_results)
    np.save(f"shd_results_{loss}.npy", shd_results)