import numpy as np

import sys
sys.path.append('../src/')

import FSC as controller

import os
import time as measure_time
import pandas as pd

import pickle

probs_condition = '80-20' # P(high)-P(low)

data_path = "../data/MouseBandits/"
data = pd.read_csv(data_path + "bandit_data.csv")
data = data.loc[data.Condition==probs_condition] # segment out task condition

M_array = np.array([2, 4])

NEpochs_MAPSO = 2500
NRestart_MAPSO = 50

train_percentage = 0.8

path_name = data_path + probs_condition + f"_train{train_percentage}/"
if not os.path.exists(path_name):
    os.makedirs(path_name)

mouses = np.unique([s[:2] for s in np.unique(data["Session"])])

sessions_train = []
sessions_val = []
for idx, ms in enumerate(mouses):
    data_curr = np.unique(data["Session"][data["Session"].str.startswith(ms)])
    sessions_train.append(data_curr[:int(len(data_curr) * train_percentage)])
    sessions_val.append(data_curr[int(len(data_curr) * train_percentage):])

print("Processing experiment with", probs_condition, "reward probabilities.")
print("There are", np.sum([len(s) for s in sessions_train]), "sessions in total used for training.")
print("There are", np.sum([len(s) for s in sessions_val]), "sessions in total used for validation.")

for idx, ms in enumerate(mouses):
    print("\t There are", len(sessions_train[idx]) + len(sessions_val[idx]), "sessions for mouse", ms + str("."), end=" ")
    print(f"Training is going to use {len(sessions_train[idx])} sessions, validation {len(sessions_val[idx])} sessions.")

print("\n")

trajectories_data_train = pickle.load(open(path_name + f"trajectories_train{train_percentage}.pkl", "rb"))
trajectories_data_val = pickle.load(open(path_name + f"trajectories_val{train_percentage}.pkl", "rb"))

ActSpace = np.unique(trajectories_data_train[0]["actions"])
ObsSpace = np.unique(trajectories_data_train[0]["observations"])

A = len(ActSpace)
Y = len(ObsSpace)

FSC_params_in_path = [f for f in os.listdir(path_name) if f.startswith("FSC")]

# save training trajectories
with open(path_name + f"trajectories_train{train_percentage}.pkl", "wb") as f:
    pickle.dump(trajectories_data_train, f)

for M in M_array:
    tic = measure_time.time()
    MemSpace = np.array(["mem " + str(num) for num in range(M)])

    if M == 2:
        sigma = 5
    elif M == 4:
        sigma = 1.5
    
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

    for idx_restart in range(max_idx_restart, max_idx_restart + NRestart_MAPSO):
        custom_postname = f"traintest{train_percentage}_restart" + str(idx_restart + 1)
            
        print("\t Restart", idx_restart + 1, "of", max_idx_restart + NRestart_MAPSO)
        
        FSC_curr = controller.FSC(M = M, A = A, Y = Y,
                                mode = "inference",
                                policy_model = "softmax",
                                policy_params = {"theta": None, "zeta": None},
                                psi = None,
                                ActSpace = ActSpace, MemSpace = MemSpace,
                                ObsSpace = ObsSpace)
        
        FSC_curr.set_inference_params(use_gradient = False, use_MAPSO = True,
                                    trainable_parameters = "all",
                                    NEpochs_gradient = 25, NBatch_gradient = 20, lr_gradient = 1e-3,
                                    scheduler_gradient = {"type": "exponential", "decay_rate": 0.99},
                                    train_split_gradient = 1,
                                    n_particles_MAPSO = 50, NEpochs_MAPSO = NEpochs_MAPSO,
                                    dynamic_topology_MAPSO = True, num_neighbors_init_MAPSO = 50, num_neighbors_final_MAPSO = 50,
                                    num_neighbors_mid_MAPSO = 2,
                                    init_particles_MAPSO = {"distribution": "normal", "mean": 0, "std": sigma}, # 1
                                    init_velocities_MAPSO = {"distribution": "uniform", "vmin": -0.01, "vmax": +0.01},
                                    print_params = False)
        
        _ = FSC_curr.fit(trajectories_data_train, verbose_MAPSO=False, verbose_epochs_MAPSO=False)
        FSC_curr.set_mode("generation")
        print("\t Best validation loss:", FSC_curr.compute_loss(trajectories_data_val))

        

        params_dict = {"psi": FSC_curr.psi, "zeta": FSC_curr.GPModel.zeta, "theta": FSC_curr.GPModel.theta,
                "training_loss": FSC_curr.losses_epochs["train"],
                "ActSpace": FSC_curr.ActSpace, "ObsSpace": FSC_curr.ObsSpace, "MemSpace": FSC_curr.MemSpace}
        
        FSC_name = f"FSC_discrete_params_M{FSC_curr.M}_A{FSC_curr.A}_Y{FSC_curr.Y}"
        FSC_name += f"_trained_loss{np.round(FSC_curr.inferencer.best_loss, 6)}"
        FSC_name += f"_{custom_postname}"

        pickle.dump(params_dict, open(path_name + FSC_name + ".pkl", "wb"))
        print()

    toc = measure_time.time()
    print("FSC with M =", M, "completed in", round(toc - tic, 2), "seconds.")
    print("\n")