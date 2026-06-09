import copy
import math
import os
import numpy as np
import xarray as xr
import glob

from causaldynamics.scm import create_scm_graph
from causaldynamics.plot import plot_scm
from causaldynamics.score import score

from causaldynamics.baselines import PCMCIPlus
from causaldynamics.baselines import FPCMCI
from causaldynamics.baselines import DYNOTEARS
from causaldynamics.baselines import VARLiNGAM
from causaldynamics.baselines import NGC_LSTM
from causaldynamics.baselines import TSCI
from causaldynamics.baselines import CUTSPlus
# from causaldynamics.baselines import RCD
# from causaldynamics.baselines import GIN
# from causaldynamics.baselines import GRASP
from causaldynamics.baselines import TCDF

# DYNOTEARS / FPCMCI + not sure about PCMCIPlus + VARLiNGAM

from tqdm import tqdm
from pathlib import Path
import torch

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt 

from causaldynamics.baselines import PICABU, StructuredODEDiscovery


tau_max = 1 # i.e. no delay 
hyperparam_search_array = np.array([0.01, 0.05, 0.1]) #0.001, 0.005
model_name = "ode_sparse_discovery"

# Hyperparameters to do search over: lambda_m only?
# Then save the plot inside each folder and report the best hyperparameter for each noise/confounder setting.

if __name__ == "__main__":

    assert torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert device.type == "cuda", "CUDA is not available. Please check your PyTorch installation and GPU configuration."

    for noise in [0.00, 0.50, 1.00, 1.50, 2.00]:
        for confounder in [False, True]:
            print(f"Running for noise {noise}, confounder {confounder}")

            # Load lorenz84 for hyperparameter search
            data_dir_noise = Path(f"/home/mila/j/julien.boussard/scratch/CausalDynamics/data/simple/noise={noise:.2f}_confounder={confounder}/data")
            eval_dir_noise = Path(f"/home/mila/j/julien.boussard/scratch/CausalDynamics/data/simple/noise={noise:.2f}_confounder={confounder}/eval/{model_name}")

            os.makedirs(eval_dir_noise, exist_ok=True)

            print(f"doing hyperparameter search on lorenz84, noise={noise}, confounder={confounder}")
            
            # NOT DOING HYPERPARAM SEARCH
            # hp_search_dataset = "Lorenz84_N10_T1000"
            # ds = xr.open_dataset(data_dir_noise / f"{hp_search_dataset}.nc")

            # timeseries = ds['time_series'].to_numpy().transpose(1, 0, 2) # shape of (N, T, D)
            # adj_matrix = ds['adjacency_matrix'].to_numpy()

            # N = timeseries.shape[-1]
            # T = timeseries.shape[1]
            # timeseries_stacked = np.stack([np.stack([timeseries[k, i:i + tau_max] for i in range(T - tau_max + 1)]) for k in range(timeseries.shape[0])])

            # assert not np.isnan(timeseries).any(), "Array contains NaN values"

            # auroc_res = np.zeros(len(hyperparam_search_array))
            # auprc_res = np.zeros(len(hyperparam_search_array))
            # shd_res = np.zeros(len(hyperparam_search_array))


            # for i, lambda_m in enumerate(hyperparam_search_array):

            #     print(f"Running for hyperparameter lambda_m: {lambda_m}")
            #     lowrank_adj_matrix = []
            #     for j, x in enumerate(timeseries_stacked):
            #         lowrank_model = StructuredODEDiscovery(
            #             t = tau_max,
            #             n = N,
            #             L_lipschitz = 1, 
            #             is_lipschitz = False,
            #             hidden_dim = 32, 
            #             n_layers = 3,
            #             device = device,
            #         )
            #         trained_model, mask = lowrank_model.run(X=torch.tensor(x).to(device), l0_l1_l2='l0', th=0.5, lr=0.001, lambda_grad=0.01, lambda_m = lambda_m, n_inner_min_sparse = 100, batch_size=32, n_inner=3_000, patience = 200, save_fig_path = eval_dir_noise / f"loss_components_{hp_search_dataset}_lambdam_{lambda_m}_ts_{j}.png")
            #         lowrank_adj_matrix.append(mask.detach().cpu().numpy())

            #     results_scores = score(
            #         preds= np.array(lowrank_adj_matrix)[:, 0], #.transpose((0, 2, 1)), # Need to transpose
            #         labs= adj_matrix,
            #         name=model_name
            #     )

            #     auroc = results_scores.loc["Joint AUROC", model_name]
            #     auprc = results_scores.loc["Joint AUPRC", model_name]
            #     shd = results_scores.loc["Joint SHD", model_name]

            #     auroc_res[i] = auroc
            #     auprc_res[i] = auprc
            #     shd_res[i] = shd

            # argmax = np.where(auroc_res == auroc_res.max())[0]
            # if len(argmax) > 1:
            #     argmax_bis = np.where(auprc[argmax] == auprc[argmax].max())[0]

            #     if len(argmax_bis) > 1:
            #         argmax_ter = np.where(shd_res[argmax[argmax_bis]] == shd_res[argmax[argmax_bis]].min())[0]

            #         if len(argmax_ter) > 1:
            #             argmax = argmax[argmax_bis][argmax_ter][-1]
            #         else:
            #             argmax = argmax[argmax_bis][argmax_ter]
            #     else:
            #         argmax = argmax[argmax_bis]
            # else:
            #     argmax = argmax[0]

            # print("Results for hyperparameter search: ")
            # print(f"AUROC: {auroc_res}")
            # print(f"AUPRC: {auprc_res}")
            # print(f"SHD: {shd_res}")
            # print(f"Best hyperparameter lambda_m: {hyperparam_search_array[argmax]}, AUROC: {auroc_res[argmax]}, AUPRC: {auprc_res[argmax]}, SHD: {shd_res[argmax]}")

            # results_dict = {}
            # results_dict["joint_shd"] = shd_res[argmax]
            # results_dict["joint_auroc"] = auroc_res[argmax]
            # results_dict["joint_auprc"] = auprc_res[argmax]
            # results_dict["any_nans"] = np.any(np.isnan(timeseries_stacked))

            # np.savez(eval_dir_noise / f'results_test_{hp_search_dataset}.npz', **results_dict)

            # final_lambda_m = hyperparam_search_array[argmax]

            final_lambda_m = 0.001

            all_files = glob.glob(str(data_dir_noise / "*.nc"))
            for file in all_files[50:]: # Just run on last 10 datasets
                name_system = file.split("/")[-1]
                name_psmodel = name_system.split(".")[0]
                print(f"Running method on {name_system}")
                name_eval =   eval_dir_noise / name_system


                ds = xr.open_dataset(data_dir_noise / name_system)
                timeseries = ds['time_series'].to_numpy().transpose(1, 0, 2) # shape of (N, T, D)
                adj_matrix = ds['adjacency_matrix'].to_numpy()

                N = timeseries.shape[-1]
                T = timeseries.shape[1]

                is_any_nan = np.any(np.isnan(timeseries))

                lowrank_adj_matrix = []
                for j, x in enumerate(timeseries[:, :, None]):
                    lowrank_model = StructuredODEDiscovery(
                        t = tau_max,
                        n = N,
                        L_lipschitz = 1, 
                        is_lipschitz = False,
                        hidden_dim = 32, 
                        n_layers = 3,
                        device = device,
                    )
                    trained_model, mask = lowrank_model.run(X=torch.tensor(x).to(device), l0_l1_l2='l0', th=0.5, lr=0.001, lambda_grad=100, lambda_m = final_lambda_m, n_inner_min_sparse = 100, batch_size=32, n_inner=4_000, patience = 200, save_fig_path = eval_dir_noise / f"loss_components_{name_psmodel}_{j}.png")
                    lowrank_adj_matrix.append(mask.detach().cpu().numpy())

                results_scores = score(
                    preds= np.array(lowrank_adj_matrix)[:, 0], #.transpose((0, 2, 1)), # Need to transpose
                    labs= adj_matrix,
                    name=model_name
                )

                results_dict = {}
                results_dict["joint_shd"] = results_scores.loc["Joint SHD", model_name]
                results_dict["joint_auroc"] = results_scores.loc["Joint AUROC", model_name]
                results_dict["joint_auprc"] = results_scores.loc["Joint AUPRC", model_name]
                results_dict["any_nans"] = is_any_nan

                np.savez(eval_dir_noise / f'results_test_{name_psmodel}.npz', **results_dict)

    print("Done")


