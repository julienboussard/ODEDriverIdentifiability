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


tau_max = 1 # i.e. no delay 
hyperparam_search_array = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01])
model_name = "ode_sparse_discovery"

# Hyperparameters to do search over: lambda_m only?
# Then save the plot inside each folder and report the best hyperparameter for each noise/confounder setting.

if __name__ == "__main__":

    assert torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert device.type == "cuda", "CUDA is not available. Please check your PyTorch installation and GPU configuration."

    for coupling_type in ["linear"]: #, "nonlinear", "periodic"
        for activation_type in ["none", "mixed"]:
            for systems_num in [3, 5, 10]:
            # Benchmark is only for each category, and noise is noise 2.00 
                for standardization_type, noise, confounder in [("False", 0.00, False), ("False", 0.00, True), ("False", 2.00, False), ("True", 0.00, False)]:
                # for standardization_type in ["False", "True"]:
                #     for noise in [0.00, 2.00]:
                #         for confounder in [False, True]:
                        print(f"Running for coupling={coupling_type}_noise={noise:.2f}_systems={systems_num}_confounder={confounder}_standardize={standardization_type}_timelag=0_activation={activation_type}")

                        # Load lorenz84 for hyperparameter search
                        data_dir_noise = Path(f"/home/mila/j/julien.boussard/scratch/CausalDynamics/data/coupled/coupling={coupling_type}_noise={noise:.2f}_systems={systems_num}_confounder={confounder}_standardize={standardization_type}_timelag=0_activation={activation_type}/data")
                        eval_dir_noise = Path(f"/home/mila/j/julien.boussard/scratch/CausalDynamics/data/coupled/coupling={coupling_type}_noise={noise:.2f}_systems={systems_num}_confounder={confounder}_standardize={standardization_type}_timelag=0_activation={activation_type}/eval/{model_name}")

                        os.makedirs(eval_dir_noise, exist_ok=True)

                        # print(f"doing hyperparameter search on lorenz84, noise={noise}, confounder={confounder}")
                        
                        # NOT DOING HYPERPARAM SEARCH, WAS ALREADY DONE, JUST RUNNING WITH BEST HYPERPARAM ON ALL DATASETS
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

                        final_lambda_m = 0.05

                        all_files = glob.glob(str(data_dir_noise / "*.nc"))
                        for file in all_files[:5]: # Just run on first 5 datasets
                            name_system = file.split("/")[-1]
                            name_psmodel = name_system.split(".")[0]

                            name_save_results = eval_dir_noise / f'results_test_{name_psmodel}.npz'
                            if name_save_results.exists():
                                print(f"Results already exist for {name_system}, skipping...")
                                continue

                            print(f"Running method on {name_system}")
                            name_eval =   eval_dir_noise / name_system


                            ds = xr.open_dataset(data_dir_noise / name_system)
                            timeseries = ds['time_series'].to_numpy()[..., 0].transpose(1, 0, 2) # shape of (N, T, D)
                            adj_matrix = ds['adjacency_matrix_summary'].to_numpy()

                            N = timeseries.shape[-1]
                            T = timeseries.shape[1]

                            is_any_nan = np.any(np.isnan(timeseries))

                            bool_nonans = False
                            lowrank_adj_matrix = []
                            for j, x in enumerate(timeseries[:, :, None]):
                                if np.isnan(x).any():
                                    print(f"Skipping timeseries {j} due to NaN values")
                                    continue
                                bool_nonans = True
                                lowrank_model = StructuredODEDiscovery(
                                    coupled=False,
                                    t = tau_max,
                                    n = N,
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
                                trained_model, mask = lowrank_model.run(n_samples=3, lambda_grad=100, X=torch.tensor(x).to(device), l0_l1_l2='l0', th=0.5, lr=0.001, lambda_m = 0.025, n_inner_min_sparse = 100, batch_size=128, n_inner=4_000, patience = 200, save_fig_path="couple_005_loss.png")
                                mask_numpy = mask.detach().cpu().numpy()

                                vec = (mask_numpy > 0.5).sum(1) == 0
                                if np.any(vec):
                                    idx = np.flatnonzero(vec)
                                    mask_numpy[0, idx, idx] = 1
                                
                                lowrank_adj_matrix.append(mask_numpy)

                            if bool_nonans:
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


