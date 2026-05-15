import numpy as np

import sys
sys.path.append('../src/')
import FSC as controller

import os
import pickle
import time as measure_time

# suppress numba parallel warnings
import warnings
from numba.core.errors import NumbaWarning
warnings.simplefilter('ignore', category=NumbaWarning)

dt = 1 / 40
train_percentage = 0.5

rat_names = ["T" + str(s) for s in [176, 219, 223]]

common_filename = "aggregation_n5"

M_array = np.arange(2, 8, 1)
NEpochs_MAPSO = 1500
NRestart_MAPSO = 10

sigma_MAPSO = 2.5

########################
##### DATA LOADING #####
########################
combined_rat_names = "_".join([rat_name for rat_name in rat_names])
data_path_aggregated = f"../data/RatTiger/dt{dt}/combined_trajectories/aggregated_{combined_rat_names}_train{train_percentage}/"

filename_train = data_path_aggregated + common_filename + f"_dt{dt}_trajectories_train.pkl"
filename_val = data_path_aggregated + common_filename + f"_dt{dt}_trajectories_val.pkl"

with open(filename_train, "rb") as f:
    trajectories_data_train = pickle.load(f)

with open(filename_val, "rb") as f:
    trajectories_data_val = pickle.load(f)

NTrain = int(len(trajectories_data_train))
NVal = len(trajectories_data_train)

actions_data_train = [trj_data["actions"] for trj_data in trajectories_data_train]
observations_data_train = [trj_data["observations"] for trj_data in trajectories_data_train]

ActSpace = np.unique(np.concatenate(actions_data_train))
ObsSpace = np.unique(np.concatenate(observations_data_train))

A = len(ActSpace)
Y = len(ObsSpace)

path_FSC = data_path_aggregated + f"FSC_params_complayer_fixedrho_{common_filename}/"
if not os.path.exists(path_FSC):
    os.makedirs(path_FSC)

FSC_params_in_path = [f for f in os.listdir(path_FSC) if f.startswith("FSC")]

########################

########################
##### FSC TRAINING #####
########################

idx_listen = np.where(ActSpace == "listen")[0][0]
idx_open_left = np.where(ActSpace == "open left")[0][0]
idx_open_right = np.where(ActSpace == "open right")[0][0]

idx_obs_left = np.where(ObsSpace == "left")[0][0]
idx_obs_silence = np.where(ObsSpace == "silence")[0][0]
idx_obs_right = np.where(ObsSpace == "right")[0][0]
idx_obs_end = np.where(ObsSpace == "end")[0][0]
idx_obs_na = np.where(ObsSpace == "na")[0][0]

for M in M_array:
    print("Training FSC with M =", M)

    FSC_params_in_path_M = [f for f in FSC_params_in_path if f"M{M}_" in f]
    # find the FSC with the largest idx_restart
    if len(FSC_params_in_path_M) > 0:
        import re
        idx_restart_array = np.array([int(re.search(r'_restart(\d+)', f).group(1)) for f in FSC_params_in_path_M])
        max_idx_restart = np.max(idx_restart_array)
        print(f"FSC with M = {M} already trained with {max_idx_restart} restarts. Starting training from restart {max_idx_restart + 1}.")
    else:
        max_idx_restart = 0

    
    tic = measure_time.time()

    if M < 4:
        MemSpace = np.array(["M-list-" + str(num) for num in range(M)])
        trainable_mask = {}
        trainable_mask_psi = np.ones(M, dtype=bool)
        trainable_mask_zeta = np.ones((A, M), dtype=bool)
        trainable_mask_theta = np.ones((Y, A, M, M), dtype=bool)

        psi_init = np.ones(M)
        zeta_init = np.ones((A, M))
        theta_init = np.ones((Y, A, M, M))

        trainable_mask["theta"] = trainable_mask_theta
        trainable_mask["psi"] = trainable_mask_psi
        trainable_mask["zeta"] = trainable_mask_zeta
    else:
        MemSpace_l1 = np.array(["M-list-" + str(num) for num in range(M - 2)])
        MemSpace_l2 = np.array(["M_open-" + str(num) for num in range(2)])
        MemSpace = np.concatenate([MemSpace_l1, MemSpace_l2])

        trainable_mask = {}

        ######## FIX PSI PARAMETERS ########
        trainable_mask_psi = np.zeros(M, dtype=bool)
        psi_init = np.zeros(M)

        if (M - 2) % 2 != 0:
            psi_init[1:] = -np.inf
        else:
            psi_init[2:] = -np.inf
        ####################################

        ######## FIX ZETA PARAMETERS ########
        trainable_mask_zeta = np.ones((A, M), dtype=bool)
        zeta_init = np.zeros((A, M))

        # only "open" actions can be taken from the second layer of the FSC memories
        trainable_mask_zeta[:, M - 2:] = False  
        zeta_init[:, M - 2:] = -np.inf

        zeta_init[idx_open_left, -2] = 1
        zeta_init[idx_open_right, -1] = 1

        # only "listen" action can be taken from the first layer of the FSC memories
        trainable_mask_zeta[idx_open_left, :M - 2] = False  
        trainable_mask_zeta[idx_open_right, :M - 2] = False
        zeta_init[idx_open_left, :M - 2] = -np.inf
        zeta_init[idx_open_right, :M - 2] = -np.inf
        ####################################

        ######## FIX THETA PARAMETERS ########
        trainable_mask_theta = np.ones((Y, A, M, M), dtype=bool)
        theta_init = np.ones((Y, A, M, M))

        # transitions from the second to the first layer are not trainable
        trainable_mask_theta[:, :, M - 2:, :M - 2] = False
        theta_init[:, :, :M - 2, M - 2:] = -np.inf
        # transitions from the first to the second layer are only possible after the end observation
        trainable_mask_theta[:, :, :M - 2, M - 2:] = False
        theta_init[:, :, M - 2:, :M - 2] = -np.inf

        trainable_mask_theta[idx_obs_end, :, :M - 2, M - 2:] = True
        theta_init[idx_obs_end, :, :M - 2, M - 2:] = 1

        # no transitions are trainable from the "na" observation, because "na" indicates that the observation is not available
        # this happens always after the "end" observation, and it's a dummy observation after the open action
        # for convention, we set the transition so that the memory does not change
        trainable_mask_theta[idx_obs_na, :, :, :] = False
        theta_init[idx_obs_na, :, :, :] = -np.inf

        for idx_M in range(M):
            theta_init[idx_obs_na, :, idx_M, idx_M] = 1

        # no transition are trainable after an open action, since the trajectory ends
        # for convention, we set the transition so that the memory does not change
        trainable_mask_theta[:, idx_open_left, :, :] = False
        trainable_mask_theta[:, idx_open_right, :, :] = False
        theta_init[:, idx_open_left, :, :] = -np.inf
        theta_init[:, idx_open_right, :, :] = -np.inf

        for idx_M in range(M):
            theta_init[:, idx_open_left, idx_M, idx_M] = 1
            theta_init[:, idx_open_right, idx_M, idx_M] = 1

        trainable_mask["theta"] = trainable_mask_theta

        trainable_mask["psi"] = trainable_mask_psi
        trainable_mask["zeta"] = trainable_mask_zeta


    for idx_restart in range(max_idx_restart, max_idx_restart + NRestart_MAPSO):
        custom_postname = f"{common_filename}_restart" + str(idx_restart + 1)

        tic = measure_time.time()
        FSC_curr = controller.FSC(M = M, A = A, Y = Y,
                            mode = "generation",
                            policy_model = "softmax",
                            policy_params = {"theta": theta_init.copy(), "zeta": zeta_init.copy()},
                            psi = psi_init.copy(),
                            ActSpace = ActSpace, MemSpace = MemSpace,
                            ObsSpace = ObsSpace)

        FSC_curr.set_inference_params(use_gradient = False, use_MAPSO = True,
                                        trainable_parameters = "elementwise_mask",
                                        trainable_mask = trainable_mask,
                                        n_particles_MAPSO = 50, NEpochs_MAPSO = NEpochs_MAPSO,
                                        dynamic_topology_MAPSO = True, num_neighbors_init_MAPSO = 50, num_neighbors_final_MAPSO = 50,
                                        num_neighbors_mid_MAPSO = 2,
                                        init_particles_MAPSO = {"distribution": "normal",
                                                            "mean": 0,
                                                            "std": sigma_MAPSO},
                                        init_velocities_MAPSO = {"distribution": "uniform",
                                                                "vmin": -0.01, "vmax": +0.01},
                                        print_params = False)

        _ = FSC_curr.fit(trajectories_data_train, verbose_MAPSO=False, verbose_epochs_MAPSO=False)
        FSC_curr.set_mode("generation")
        curr_val_loss = FSC_curr.compute_loss(trajectories_data_val)
        training_loss_epochs = FSC_curr.losses_epochs["train"]
        toc = measure_time.time()

        print(f"\t Validation loss for restart {idx_restart + 1}:", curr_val_loss)
        print(f"\t Elapsed time for restart {idx_restart + 1}: {round(toc - tic, 2)} seconds.")
        
        FSC_name = f"FSC_discrete_params_M{FSC_curr.M}_A{FSC_curr.A}_Y{FSC_curr.Y}"
        FSC_name += f"_trained_loss{np.round(FSC_curr.inferencer.best_loss, 6)}"
        FSC_name += f"_{custom_postname}"
        best_psi = FSC_curr.psi.copy()
        best_zeta = FSC_curr.GPModel.zeta.copy()
        best_theta = FSC_curr.GPModel.theta.copy()

        params_dict = {"psi": best_psi, "zeta": best_zeta, "theta": best_theta,
                        "training_loss": training_loss_epochs,
                        "ActSpace": FSC_curr.ActSpace, "ObsSpace": FSC_curr.ObsSpace, "MemSpace": FSC_curr.MemSpace}
        pickle.dump(params_dict, open(path_FSC + FSC_name + ".pkl", "wb"))
        print()