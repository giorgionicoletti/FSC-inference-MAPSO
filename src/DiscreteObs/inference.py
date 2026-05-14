import torch
from torch import nn
import numpy as np
import numba as nb
import random

import utils
import MAPSO

class InferenceDiscreteObs:
    def __init__(self, FSC):
        """
        """
        self.FSC = FSC

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")


        if not isinstance(self.FSC.psi, torch.Tensor):
            self.FSC.psi = torch.tensor(self.FSC.psi.astype(np.float32), device=self.device)

        if not isinstance(self.FSC.psi, nn.Parameter) and FSC.trained == False:
            self.FSC.psi = nn.Parameter(self.FSC.psi)

        for idx, param in enumerate(self.FSC.GPModel.params):
            param_name = self.FSC.GPModel.param_names[idx]

            if not isinstance(param, torch.Tensor):
                param = torch.tensor(param.astype(np.float32), device=self.device)

            if not isinstance(param, nn.Parameter) and FSC.trained == False:
                param = nn.Parameter(param)

            self.FSC.GPModel.__setattr__(param_name, param)

        self.InternalMemSpace = torch.arange(self.FSC.M)
        self.InternalActSpace = torch.arange(self.FSC.A)
        self.InternalObsSpace = torch.arange(self.FSC.Y)

        self.trajectories_loaded = False
        self.optimizer_initialized = False
        self.trained = False

    def get_policy_params(self):
        return tuple([self.FSC.GPModel.__getattribute__(param) for param in self.FSC.GPModel.param_names])

    def get_TMat(self):
        return self.FSC.GPModel.get_TMat_torch()
    
    def get_memory_transition(self):
        return self.FSC.GPModel.get_memory_transition_torch()
    
    def get_action_policy(self):
        return self.FSC.GPModel.get_action_policy_torch()


    def load_trajectories(self, trajectories):
        """
        Loads a set of trajectories to be used for training the FSC.

        Parameters:
        --- trajectories: list of dicts
            List of dictionaries containing the actions and observations for each trajectory.
        """
        self.ObsAct_trajectories = []
        self.observations_trajectories_np = []
        self.actions_trajectories_np = []
        self.n_trajectories = len(trajectories)

        self.pStart_ya_emp = np.zeros((self.FSC.Y, self.FSC.A))   


        for trajectory in trajectories:
            observations = self._map_obs_to_internal_space(trajectory["observations"])
            actions = self._map_act_to_internal_space(trajectory["actions"])
            self.observations_trajectories_np.append(observations)
            self.actions_trajectories_np.append(actions)

            self.ObsAct_trajectories.append([torch.tensor(observations), torch.tensor(actions)])
            y0 = observations[0]
            a0 = actions[0]

            self.pStart_ya_emp[y0, a0] += 1
        
        self.pStart_ya_emp /= np.sum(self.pStart_ya_emp)

        self.trajectories_loaded = True

    def _map_obs_to_internal_space(self, obs):
        """
        Helper method to map an observation sequence to the internal observation space.

        Parameters:
        --- obs: np.array
            Observation sequence to map.

        Returns:
        --- obs_internal: np.array
            Observation sequence in the internal observation space.
        """
        return np.array([self.get_obs_idx(o) for o in obs])
    
    def _map_act_to_internal_space(self, act):
        """
        Helper method to map an action sequence to the internal action space.

        Parameters:
        --- act: np.array
            Action sequence to map.

        Returns:
        --- act_internal: np.array
            Action sequence in the internal action space.
        """
        return np.array([self.get_act_idx(a) for a in act])
        
    def get_obs_idx(self, obs):
        """
        Helper method to get the index of an observation in the observation space.

        Parameters:
        --- obs: int
            Observation to get the index of.

        Returns:
        --- idx: int
            Index of the observation in the observation space.
        """

        return np.where(obs == self.FSC.ObsSpace)[0][0]
    
    def get_act_idx(self, act):
        """
        Helper method to get the index of an action in the action space.

        Parameters:
        --- act: int
            Action to get the index of.

        Returns:
        --- idx: int
            Index of the action in the action space.
        """

        return np.where(self.FSC.ActSpace == act)[0][0]
    
    def get_mem_idx(self, mem):
        """
        Helper method to get the index of a memory state in the memory space.

        Parameters:
        --- mem: int
            Memory state to get the index of.

        Returns:
        --- idx: int
            Index of the memory state in the memory space
        """

        return np.where(self.FSC.MemSpace == mem)[0][0]
    
    def evaluate_nloglikelihood(self, idx_traj, grad_required=False):
        """
        Wrapper method to evaluate the negative log-likelihood of a given trajectory.
        It distinguishes between the case of custom observation and action spaces and the case
        of default observation and action spaces.

        Parameters:
        --- idx_traj: int
            Index of the trajectory to evaluate.
        --- grad_required: bool (default = False)
            Flag indicating whether the gradient is required or not.

        Returns:
        --- nLL: float
            Negative log-likelihood of the trajectory.
        """
        observations, actions = self.ObsAct_trajectories[idx_traj]
        
        return self.loss(observations, actions, grad_required = grad_required)
            
    def loss(self, observations, actions, grad_required=True):
        """
        Method to compute the negative log-likelihood of a given trajectory with default observation and action spaces.
        The gradients of the loss are computed if the grad_required flag is set to True.

        Parameters:
        --- observations: torch.tensor
            Array of observations.
        --- actions: torch.tensor
            Array of actions.
        --- grad_required: bool (default = True)
            Flag indicating whether the gradient is required or not.

        Returns:
        --- nLL: float
            Negative log-likelihood of the trajectory.
        """
        nLL = torch.tensor(0.0, requires_grad = grad_required)

        TMat = self.FSC.GPModel.get_TMat_torch()

        if self.FSC._init_memory_obs_dependent:
            rho = self.FSC.rho[observations[0]]
        else:
            rho = self.FSC.rho

        for t in range(observations.size(0)):
            idx_a = actions[t]
            idx_obs = observations[t]

            transition_probs = TMat[idx_obs, idx_a].T

            if torch.sum(transition_probs) == 0:
                break

            if t == 0:
                if transition_probs.device.type == 'mps':
                    # MPS-specific workaround
                    transition_probs_safe = transition_probs.clone().detach().requires_grad_(transition_probs.requires_grad)
                    rho_safe = rho.clone().detach().requires_grad_(rho.requires_grad)
                    m = torch.matmul(transition_probs_safe, rho_safe)
                else:
                    m = torch.matmul(transition_probs, rho)
            else:
                m = torch.matmul(transition_probs, m)

            if torch.sum(m) == 0:
                break

            mv = torch.sum(m)
            nLL = nLL - torch.log(mv)
            m /= mv

        
        if torch.sum(m) == 0:
            return nLL
        else:
            return nLL - torch.log(torch.sum(m))


    def optimize_w_MAPSO(self, trainable_params, trainable_params_mask,
                        n_particles, NEpochs,
                        init_particles, init_velocities,
                        c1_init, c2_init, w_init,
                        sigma_min, sigma_max,
                        dynamic_topology, n_neighbors_init, n_neighbors_final, num_neighbors_mid,
                        verbose, verbose_epochs):
        
        assert self.trajectories_loaded, "No trajectories have been loaded. Load trajectories with the load_trajectories method."
        assert not self.trained, "The model has already been trained. If you want to train it again, reinitialize it or set the flag self.trained to False."

        spacedim = sum([param.numel() for param in self.get_policy_params()]) + self.FSC.psi.numel()

        trainable_mask = np.zeros(spacedim, dtype=bool)
        init_pos = np.zeros((n_particles, spacedim))
        init_vel = np.zeros((n_particles, spacedim))

        if init_particles["distribution"] == "multivariate_normal":
            random_pos_mv = np.random.multivariate_normal(init_particles["mean"], init_particles["cov"],
                                                         n_particles)

        start_idx = 0
        for idx, param in enumerate(self.get_policy_params()):
            param_name = self.FSC.GPModel.param_names[idx]
            end_idx = start_idx + param.numel()
            if trainable_params_mask[param_name] is not None:
                trainable_mask[start_idx:end_idx] = trainable_params_mask[param_name].flatten()
            else:
                trainable_mask[start_idx:end_idx] = trainable_params[param_name]

            flatten_param = self.FSC.GPModel.__getattribute__(self.FSC.GPModel.param_names[idx]).detach().cpu().numpy().flatten()
            init_pos[:, start_idx:end_idx] = np.tile(flatten_param, (n_particles, 1)) 
            init_vel[:, start_idx:end_idx] = np.zeros((n_particles, param.numel()))
            
            if trainable_params[param_name]:
                num_trainable = np.sum(trainable_mask[start_idx:end_idx])

                if init_particles["distribution"] == "uniform":
                    random_pos = np.random.uniform(init_particles["xmin"], init_particles["xmax"],
                                                   (n_particles, num_trainable))    
                elif init_particles["distribution"] == "normal":
                    random_pos = np.random.normal(init_particles["mean"], init_particles["std"],
                                                 (n_particles, num_trainable))
                elif init_particles["distribution"] == "multivariate_normal":
                    random_pos = random_pos_mv[:, start_idx:end_idx][:, trainable_mask[start_idx:end_idx]]
                elif init_particles["distribution"] == "uniform_with_biases":
                    random_pos = np.random.uniform(init_particles["xmin"], init_particles["xmax"],
                                                   (n_particles, num_trainable))
                    random_pos += init_particles["biases"][start_idx:end_idx][trainable_mask[start_idx:end_idx]]                                            
                else:
                    raise ValueError("Invalid position distribution.")
                
                if init_velocities["distribution"] == "uniform":
                    random_vel = np.random.uniform(init_velocities["vmin"], init_velocities["vmax"],
                                                   (n_particles, num_trainable))
                elif init_velocities["distribution"] == "normal":
                    random_vel = np.random.normal(init_velocities["mean"], init_velocities["std"],
                                                  (n_particles, num_trainable))
                else:
                    raise ValueError("Invalid velocity distribution.")

                init_pos[:, start_idx:end_idx][:, trainable_mask[start_idx:end_idx]] = random_pos
                init_vel[:, start_idx:end_idx][:, trainable_mask[start_idx:end_idx]] = random_vel

            start_idx = end_idx
        
        if trainable_params_mask["psi"] is not None:
            trainable_mask[-self.FSC.psi.numel():] = trainable_params_mask["psi"].flatten()
        else:
            trainable_mask[-self.FSC.psi.numel():] = trainable_params["psi"]

        #print(self.FSC.psi.shape, self.FSC.psi.numel())
        init_pos[:, -self.FSC.psi.numel():] = np.tile(self.FSC.psi.detach().cpu().numpy().flatten(), (n_particles, 1))
        init_vel[:, -self.FSC.psi.numel():] = np.zeros((n_particles, self.FSC.psi.numel()))
        
        if trainable_params["psi"]:
            num_trainable = np.sum(trainable_mask[-self.FSC.psi.numel():])

            if init_particles["distribution"] == "uniform":
                    random_pos = np.random.uniform(init_particles["xmin"], init_particles["xmax"],
                                                   (n_particles, num_trainable))    
            elif init_particles["distribution"] == "normal":
                random_pos = np.random.normal(init_particles["mean"], init_particles["std"],
                                                (n_particles, num_trainable))
            elif init_particles["distribution"] == "multivariate_normal":
                random_pos = random_pos_mv[:, -self.FSC.psi.numel():][:, trainable_mask[-self.FSC.psi.numel():]]
            elif init_particles["distribution"] == "uniform_with_biases":
                random_pos = np.random.uniform(init_particles["xmin"], init_particles["xmax"],
                                                (n_particles, num_trainable))
                random_pos += init_particles["biases"][-self.FSC.psi.numel():][trainable_mask[-self.FSC.psi.numel():]]                                            
            else:
                raise ValueError("Invalid position distribution.")
            
            if init_velocities["distribution"] == "uniform":
                random_vel = np.random.uniform(init_velocities["vmin"], init_velocities["vmax"],
                                                (n_particles, num_trainable))
            elif init_velocities["distribution"] == "normal":
                random_vel = np.random.normal(init_velocities["mean"], init_velocities["std"],
                                                (n_particles, num_trainable))
            else:
                raise ValueError("Invalid velocity distribution.")

            init_pos[:, -self.FSC.psi.numel():][:, trainable_mask[-self.FSC.psi.numel():]] = random_pos
            init_vel[:, -self.FSC.psi.numel():][:, trainable_mask[-self.FSC.psi.numel():]] = random_vel


        if dynamic_topology:
            gbests, gbest_values = MAPSO.particle_swarm_optimization_discrete_kNN(self.FSC.GPModel._nb_get_TMat_flat,
                                                                                 trainable_mask,
                                                                                 spacedim, n_particles, NEpochs,
                                                                                 self.FSC.M, self.FSC.A, self.FSC.Y,
                                                                                 self.observations_trajectories_np,
                                                                                 self.actions_trajectories_np,
                                                                                 init_pos, init_vel,
                                                                                 num_neighbors_init = n_neighbors_init,
                                                                                 num_neighbors_final = n_neighbors_final,
                                                                                 num_neighbors_mid = num_neighbors_mid,
                                                                                 c1 = c1_init, c2 = c2_init, w = w_init,
                                                                                 sigma_min = sigma_min, sigma_max = sigma_max,
                                                                                 verbose = verbose, verbose_epochs = verbose_epochs,
                                                                                 init_memory_obs_dependent = self.FSC._init_memory_obs_dependent)
        else:
            gbests, gbest_values = MAPSO.particle_swarm_optimization_discrete(self.FSC.GPModel._nb_get_TMat_flat,
                                                                             trainable_mask,
                                                                             spacedim, n_particles, NEpochs,
                                                                             self.FSC.M, self.FSC.A, self.FSC.Y,
                                                                             self.observations_trajectories_np,
                                                                             self.actions_trajectories_np,
                                                                             init_pos, init_vel,
                                                                             c1 = c1_init, c2 = c2_init, w = w_init,
                                                                             sigma_min = sigma_min, sigma_max = sigma_max,
                                                                             verbose = verbose, verbose_epochs = verbose_epochs,
                                                                             init_memory_obs_dependent = self.FSC._init_memory_obs_dependent)

        for idx, param in enumerate(self.get_policy_params()):
            param_name = self.FSC.GPModel.param_names[idx]
            start_idx = 0
            for idx, param in enumerate(self.get_policy_params()):
                param_name = self.FSC.GPModel.param_names[idx]
                end_idx = start_idx + param.numel()
                new_param = torch.tensor(gbests[-1, start_idx:end_idx].reshape(param.shape).astype(np.float32), device=self.device)
                self.FSC.GPModel.__setattr__(param_name,
                                             nn.Parameter(new_param))
                start_idx = end_idx
        new_psi = gbests[-1, -self.FSC.psi.numel():].astype(np.float32)
        new_psi = new_psi.reshape(self.FSC.psi.shape)
        self.FSC.psi = nn.Parameter(torch.tensor(new_psi, device=self.device))

        return gbest_values

    def optimize_w_gradient(self, use_ccopt,
                            trainable_params, trainable_params_mask, any_masked,
                            NEpochs, NBatch, lr,
                            train_split, optimizer, scheduler_dict,
                            maxiter, rho0, th, c_gauge,
                            verbose, verbose_epochs):
        assert self.trajectories_loaded, "No trajectories have been loaded. Load trajectories with the load_trajectories method."
        assert not self.trained, "The model has already been trained. If you want to train it again, reinitialize it or set the flag self.trained to False."

        lr_dict = {}

        if isinstance(lr, float):
            single_lr = True
            for param in self.FSC.GPModel.param_names:
                lr_dict[param] = lr
            lr_dict["psi"] = lr
        elif isinstance(lr, dict):
            for param in self.FSC.GPModel.param_names:
                if param != "psi" and trainable_params[param]:
                    if param in lr:
                        lr_dict[param] = lr[param]
                    else:
                        raise ValueError(f"Missing learning rate for parameter {param}.")
            if "psi" not in lr and trainable_params["psi"] and not use_ccopt:
                raise ValueError("Missing learning rate for psi.")
            else:
                lr_dict["psi"] = lr["psi"]
        else:
            raise ValueError("Invalid learning rate. The learning rate must be a float or a dictionary with the parameters as keys.")

        parkey_optimizer = []
        for idx, param in enumerate(self.FSC.GPModel.param_names):
            if trainable_params[param]:
                parkey_optimizer.append({"params": self.FSC.GPModel.__getattribute__(param), 'lr': lr_dict[param]})
        if trainable_params["psi"] and not use_ccopt:
            parkey_optimizer.append({'params': self.FSC.psi, 'lr': lr_dict["psi"]})

        if optimizer == "ADAM":
            self.optimizer = torch.optim.Adam(parkey_optimizer)
        elif optimizer == "SDG":
            self.optimizer = torch.optim.SGD(parkey_optimizer)
        
        if scheduler_dict["type"] == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=scheduler_dict["decay_rate"])
        elif scheduler_dict["type"] == "fixed":
            scheduler = None
        else:
            raise ValueError("Invalid scheduler.")

        NTrain = int(train_split * len(self.ObsAct_trajectories))
        NVal = len(self.ObsAct_trajectories) - NTrain

        trjs_train = self.ObsAct_trajectories[:NTrain]
        best_params = [self.FSC.GPModel.__getattribute__(param) for param in self.FSC.GPModel.param_names]
        best_psi = self.FSC.psi
        best_epoch = 0

        losses_train = []

        init_loss = 0
        for idx_traj in range(NTrain):
            init_loss += self.loss(trjs_train[idx_traj][0], trjs_train[idx_traj][1], grad_required=False).item()
        losses_train.append(init_loss / NTrain)

        init_msg = f"Training with {NTrain} trajectories"
        if NVal != 0:
            trjs_val = self.FeatAct_trajectories[NTrain:]
            losses_val = []
            
            init_loss_val = 0
            for idx_traj in range(NVal):
                init_loss_val += self.loss(trjs_val[idx_traj][0], trjs_val[idx_traj][1], grad_required=False).item()
            losses_val.append(init_loss_val / NVal)

            init_msg += f" and validating with {NVal} trajectories."
            init_msg += " Initial training loss: " + str(losses_train[0]) + ". Initial validation loss: " + str(losses_val[0]) + "."
        else:
            init_msg += ". Initial loss: " + str(losses_train[0]) + "."
        
        if verbose_epochs:
            if single_lr:
                print(init_msg + f" Using a single learning rate of {lr}.")
            else:
                for idx, param in enumerate(self.FSC.GPModel.param_names):
                    print(init_msg + f" Using learning rate {lr[param]} for {param}.")
                print(init_msg + f" Using learning rate {lr['psi']} for psi.")

        for epoch in range(NEpochs):
            running_loss = 0.0
            running_count = 0

            random.shuffle(trjs_train)

            for idx in range(0, NTrain, NBatch):
                if any_masked:
                    pre_loss_params = {}
                    for key in self.FSC.GPModel.param_names:
                        mask = trainable_params_mask[key]
                        if mask is not None:
                            pre_loss_params[key] = self.FSC.GPModel.__getattribute__(key).detach().clone()
                    if trainable_params_mask["psi"] is not None:
                        pre_loss_psi = self.FSC.psi.detach().clone()

                self.optimizer.zero_grad()
                loss = torch.tensor(0.0, requires_grad=True)

                TMat = self.FSC.GPModel.get_TMat_torch()

                if use_ccopt and not self.FSC._init_memory_obs_dependent and trainable_params["psi"]:
                    if rho0 is None:
                        rho0 = np.ones(self.FSC.M)/self.FSC.M

                    rho, _ = InferenceDiscreteObs.optimize_rho(self.FSC.Y, self.FSC.M, self.FSC.A,
                                                               TMat.detach().cpu().numpy(), self.pStart_ya_emp,
                                                               rho0, maxiter, th = th)
                    
                    rho = torch.tensor(rho.astype(np.float32), device = self.device)
                    self.FSC.psi = nn.Parameter(torch.log(rho) + c_gauge)

                count = 0
                for idx_traj in range(idx, idx + NBatch):
                    if idx_traj < NTrain:
                        loss_traj = self.loss(trjs_train[idx_traj][0], trjs_train[idx_traj][1])
                        if torch.isnan(loss_traj):
                            continue
                        loss = loss + loss_traj
                        count += 1

                if count == 0:
                    err_msg = "Gradient optimization failed because no valid trajectories were found in a batch. This means that the loss could not be evaluated due to forbidden transition, and that the current parameters are not compatible with some trajectories."
                    err_msg += " Either improve initialization or choose a smaller learning rate. Overwriting with the best parameters found so far."

                    self.FSC.psi = best_psi
                    for idx, param in enumerate(self.FSC.GPModel.param_names):
                        self.FSC.GPModel.__setattr__(param, nn.Parameter(best_params[idx]))

                    raise RuntimeError(err_msg)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_count += count

                if any_masked:
                    for key in self.FSC.GPModel.param_names:
                        mask = ~trainable_params_mask[key]
                        if mask is not None:
                            self.FSC.GPModel.__getattribute__(key).data[mask] = pre_loss_params[key].data[mask].clone()
                    mask_psi = ~trainable_params_mask["psi"]
                    if mask_psi is not None:
                        self.FSC.psi.data[mask_psi] = pre_loss_psi.data[mask_psi].clone()

            running_loss = running_loss / running_count
            losses_train.append(running_loss)

            if NVal != 0:
                running_loss_val = 0.0

                for idx_traj in range(NVal):
                    loss_val = torch.tensor(0.0, requires_grad=False)

                    loss_traj_val = self.loss(trjs_val[idx_traj][0], trjs_val[idx_traj][1], grad_required=False)
                    loss_val = loss_val + loss_traj_val

                    running_loss_val += loss_val.item()

                running_loss_val = running_loss_val / NVal
                losses_val.append(running_loss_val)

                if running_loss_val < min(losses_val[:-1]):
                    best_params = [self.FSC.GPModel.__getattribute__(param).detach().clone() for param in self.FSC.GPModel.param_names]
                    best_psi = self.FSC.psi.detach().clone()
                    best_epoch = epoch + 1

                if verbose_epochs:
                    print(f"Epoch {epoch + 1} - Training loss: {round(running_loss, 5)}, Validation loss: {round(running_loss_val, 5)} - Learning rate: {round(self.optimizer.param_groups[0]['lr'], 5)}")
            
            else:
                if running_loss < min(losses_train[:-1]):
                    best_params = [self.FSC.GPModel.__getattribute__(param).detach().clone() for param in self.FSC.GPModel.param_names]
                    best_psi = self.FSC.psi.detach().clone()
                    best_epoch = epoch + 1

                if verbose_epochs:
                    print(f"Epoch {epoch + 1} - Training loss: {round(running_loss, 5)} - Learning rate: {round(self.optimizer.param_groups[0]['lr'], 5)}")

            if scheduler is not None:
                scheduler.step()

        if verbose_epochs:
            print("Training complete. Best parameters found at epoch", best_epoch)

        self.FSC.psi = best_psi
        for idx, param in enumerate(self.FSC.GPModel.param_names):
            self.FSC.GPModel.__setattr__(param, nn.Parameter(best_params[idx]))

        if NVal != 0:
            return losses_train, losses_val
        else:
            return losses_train, None

    def optimize(self, inference_params, verbose, verbose_epochs):
        
        assert self.trajectories_loaded, "No trajectories have been loaded. Load trajectories with the load_trajectories method."
        assert not self.trained, "The model has already been trained. If you want to train it again, reinitialize it or set the flag self.trained to False."

        loss_epochs = {}

        trainable_params = inference_params["trainable_parameters"]
        trainable_mask = inference_params["trainable_mask"]

        # check if any value of the trainable_mask dictionary is not None
        any_masked = False
        for key, val in trainable_mask.items():
            if val is not None:
                any_masked = True
                break

        if inference_params["use_MAPSO"]:
            n_particles = inference_params['n_particles_MAPSO']
            NEpochs = inference_params['NEpochs_MAPSO']

            c1_init = inference_params['c1_init_MAPSO']
            c2_init = inference_params['c2_init_MAPSO']
            w_init = inference_params['w_init_MAPSO']

            sigma_min = inference_params['sigma_min_MAPSO']
            sigma_max = inference_params['sigma_max_MAPSO']

            dynamic_topology = inference_params['dynamic_topology_MAPSO']
            num_neighbors_init = inference_params['num_neighbors_init_MAPSO']
            num_neighbors_final = inference_params['num_neighbors_final_MAPSO']
            num_neighbors_mid = inference_params['num_neighbors_mid_MAPSO']

            init_particles = inference_params['init_particles_MAPSO']
            init_velocities = inference_params['init_velocities_MAPSO']


            loss_MAPSO = self.optimize_w_MAPSO(trainable_params, trainable_mask,
                                             n_particles, NEpochs,
                                             init_particles, init_velocities,
                                             c1_init, c2_init, w_init,
                                             sigma_min, sigma_max,
                                             dynamic_topology,
                                             num_neighbors_init, num_neighbors_final, num_neighbors_mid,
                                             verbose, verbose_epochs)
            loss_epochs["MAPSO"] = loss_MAPSO

        if inference_params["use_gradient"]:

            NEpochs = inference_params['NEpochs_gradient']
            NBatch = inference_params['NBatch_gradient']
            lr = inference_params['lr_gradient']

            train_split = inference_params['train_split_gradient']
            scheduler = inference_params['scheduler_gradient']
            optimizer = inference_params['optimizer_gradient']
            
            if inference_params["use_ccopt"]:
                if self.FSC._init_memory_obs_dependent:
                    raise ValueError("CCOpt cannot be used with memory observation dependent initialization.")
                
                maxiter = inference_params['maxiter_ccopt']
                rho0 = inference_params['rho0_ccopt']
                th = inference_params['th_ccopt']
                c_gauge = inference_params['c_gauge_ccopt']
            else:
                maxiter = None
                rho0 = None
                th = None
                c_gauge = None

            losses_gradient = self.optimize_w_gradient(inference_params["use_ccopt"],
                                                       trainable_params, trainable_mask, any_masked,
                                                       NEpochs, NBatch, lr,
                                                       train_split, optimizer, scheduler,
                                                       maxiter, rho0, th, c_gauge,
                                                       verbose, verbose_epochs)
            
            if inference_params["use_MAPSO"]:
                loss_epochs["train"] = np.concatenate([loss_epochs["MAPSO"], losses_gradient[0][1:]])
                if train_split < 1.0:
                    loss_epochs["val"] = np.concatenate([loss_epochs["MAPSO"], losses_gradient[1][1:]])
                    self.best_loss = np.min(loss_epochs["val"])
                else:
                    self.best_loss = np.min(loss_epochs["train"])
            else:
                loss_epochs["train"] = losses_gradient[0]
                if train_split < 1.0:
                    loss_epochs["val"] = losses_gradient[1]
                    self.best_loss = np.min(loss_epochs["val"])
                else:
                    self.best_loss = np.min(loss_epochs["train"])
        else:
            loss_epochs["train"] = loss_MAPSO
            self.best_loss = np.min(loss_epochs["train"])

        self.trained = True

        return loss_epochs
    
    def get_inferred_policy_params(self):
        return [self.FSC.GPModel.__getattribute__(param) for param in self.FSC.GPModel.param_names]


    @staticmethod
    @nb.njit
    def optimize_rho(Y, M, A, TMat, pya, rhok, maxiter, th):
        TMat = np.transpose(TMat, (0, 2, 3, 1))
        wVec = np.zeros((Y, A, M))
        for y in range(Y):
            for a in range(A):
                for m in range(M):
                    wVec[y, a, m] = np.sum(TMat[y, m, :, a])

        for _ in range(maxiter):
            wsumexp_test_k = np.zeros((Y, A))
            
            for y in range(Y):
                for a in range(A):
                    wsumexp_test_k[y, a] = np.sum(wVec[y, a] * rhok)
            
            grad = wVec * rhok / wsumexp_test_k[..., None]

            rhok_new = np.zeros(M)

            for y in range(Y):
                for a in range(A):
                    rhok_new += pya[y, a] * grad[y, a]
            
            if np.linalg.norm(rhok_new - rhok) < th:
                break

            rhok = rhok_new

        return rhok, np.linalg.norm(rhok_new - rhok)