import numpy as np
import parametrizations
import torch
import utils
from torch import nn

from DiscreteObs.generation import GenerationDiscreteObs
from DiscreteObs.inference import InferenceDiscreteObs

import warnings

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import pickle
import os

import parametrizations
from environments.base_environment_model import BaseEnvironmentModel

class FSC:
    def __init__(self, mode,
                 M, A, Y = None,
                 policy_model = "split",
                 policy_params = {"theta": None, "zeta": None},
                 psi = None, seed = None,
                 ObsSpace = None, ActSpace = None, MemSpace = None,
                 init_memory_obs_dependent = False):
        
        self.__init_supported_objects()

        self.__check_and_init_observations(Y)
        self.__initialize_structure(M, A, Y)

        self._init_memory_obs_dependent = init_memory_obs_dependent
        self.__init_generalized_policy_model(policy_model,
                                             policy_params, seed)
        self.__check_parameters_consistency(psi)

        if psi is not None:
            self.psi = psi
        else:
            self.__initialize_psi(seed)

        self.__initialize_spaces(ObsSpace, ActSpace, MemSpace)

        self.mode = mode

        self.__loaded_trajectories_inference = False
        self.__trained = False

        if mode == 'generation':
            self.generator = GenerationDiscreteObs(self)
        elif mode == 'inference':
            self.inferencer = InferenceDiscreteObs(self)
        else:
            raise ValueError("Mode must be either 'generation' or 'inference'")
        
        self.__policy_model_name = policy_model

    def get_rho(self):
        return self.rho


    def get_TMat(self):
        if self.mode == 'generation':
            return self.GPModel.get_TMat_numpy().transpose(0, 2, 3, 1)
        elif self.mode == 'inference':
            return self.GPModel.get_TMat_torch().permute(0, 2, 3, 1)
            
    def get_action_policy(self, obs = None):
        if obs is not None:
            raise Warning("Explicit observation values are not used in the policy model for discrete observations.")
        if self.mode == 'generation':
            return self.generator.get_action_policy()
        elif self.mode == 'inference':
            return self.inferencer.get_action_policy()
            
    def get_memory_transitions(self, obs = None):
        if obs is not None:
            raise Warning("Explicit observation values are not used in the policy model for discrete observations.")
        if self.mode == 'generation':
            return self.generator.get_memory_transition()
        elif self.mode == 'inference':
            return self.inferencer.get_memory_transition()
                        
    def save(self, directory, filename = None,
             custom_postname = ""):
        if filename is None:
            filename = f"FSC_discrete_{self.__policy_model_name}_M{self.M}_A{self.A}"
            filename += f"_Y{self.Y}"

            if self.trained:
                filename += f"_trained_loss{np.round(self.inferencer.best_loss, 6)}"
            
            if self._init_memory_obs_dependent:
                filename += "_initmemobs"

            if custom_postname != "":
                filename += f"_{custom_postname}"
                
            filename += ".pkl"
        
        elif not filename.endswith(".pkl"):
            filename += ".pkl"

        if not directory.endswith("/"):
            directory += "/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(directory + filename, 'wb') as file:
            pickle.dump(self, file)

        print(f"FSC saved in {directory + filename}")


    ##############################################################################################################################
    ############### Methods to handle the conversion between generation and inference ############################################
    ##############################################################################################################################

    def set_mode(self, mode):
        if mode not in ['generation', 'inference']:
            raise ValueError("Mode must be either 'generation' or 'inference'")
        
        if mode == 'generation':
            for key in self.GPModel.param_names:
                if getattr(self.GPModel, key) is not None:
                    param = getattr(self.GPModel, key)
                    if isinstance(param, torch.nn.Parameter):
                        setattr(self.GPModel, key, param.detach().cpu().numpy().astype(np.float64))
                    elif isinstance(param, torch.Tensor):
                        setattr(self.GPModel, key, param.cpu().numpy().astype(np.float64))
                    else:
                        setattr(self.GPModel, key, param.astype(np.float64))
            
            if isinstance(self.psi, torch.nn.Parameter):
                self.psi = self.psi.detach().cpu().numpy().astype(np.float64)
            elif isinstance(self.psi, torch.Tensor):
                self.psi = self.psi.cpu().numpy().astype(np.float64)
            elif isinstance(self.psi, np.ndarray):
                self.psi = self.psi.astype(np.float64)

            # check if observations have been loaded
            if hasattr(self, '_fitted_observations_numpy'):
                self.convert_observations_to_numpy()

            self.generator = GenerationDiscreteObs(self)
            if hasattr(self, '_fitted_observations_numpy'):
                self.generator.load_observations(self._fitted_observations_numpy)
            
        elif mode == 'inference':
            self.inferencer = InferenceDiscreteObs(self)

        self.mode = mode

    def convert_observations_to_numpy(self):
        if not self.__loaded_trajectories_inference:
            raise ValueError("No trajectories to fit have been loaded.")
        
        self._fitted_observations_numpy = []
        for obs, _ in self.inferencer.ObsAct_trajectories:
            obs_original_space = np.array([self.ObsSpace[o.item()] for o in obs])
            self._fitted_observations_numpy.append(obs_original_space)

    ##############################################################################################################################



    ##############################################################################################################################
    ############### Methods to define and compute the loss #######################################################################
    ##############################################################################################################################

    def compute_loss(self, trajectories):
        nLL = 0.0

        if self.mode == 'inference':
            for trj in trajectories:
                fo = trj["observations"]
                fo = self.inferencer._map_obs_to_internal_space(fo)
                fo = torch.tensor(fo)

                actions = self.inferencer._map_act_to_internal_space(trj["actions"])
                actions = torch.tensor(actions)

                nLL += self.inferencer.loss(fo, actions, grad_required=False)
        else:
            for trj in trajectories:
                nLL += self.generator.evaluate_nloglikelihood(trj)

        return nLL / len(trajectories)
    
    def evaluate_nloglikelihood_for_trajectory(self, trajectory_idx):
        if self.mode != 'inference':
            raise ValueError("Mode must be 'inference' to evaluate the negative log-likelihood.")
        if not self.__loaded_trajectories_inference:
            raise ValueError("No trajectories to fit have been loaded.")
        return self.inferencer.evaluate_nloglikelihood(trajectory_idx)

    ##############################################################################################################################



    ##############################################################################################################################
    ############### Methods to handle the inference procedure ####################################################################
    ##############################################################################################################################

    def reset_trained_status(self):
        self.__trained = False
    
    def set_trained_status(self, val):
        self.__trained = val

    def initialize_for_inference(self):
        self.set_mode('inference')
        self.__loaded_trajectories_inference = False

    def set_inference_params(self,
                             use_gradient = True, use_MAPSO = True, use_ccopt = False,
                             trainable_parameters = "all", trainable_mask = None,
                             NEpochs_gradient = None, NBatch_gradient = None, lr_gradient = None,
                             optimizer_gradient = "ADAM",
                             scheduler_gradient = {"type": "exponential", "decay_rate": 0.8},
                             train_split_gradient = 1,
                             n_particles_MAPSO = None, NEpochs_MAPSO = None,
                             init_particles_MAPSO = {"distribution": "uniform", "xmin": -5, "xmax": +5},
                             init_velocities_MAPSO = {"distribution": "uniform", "vmin": 0, "vmax": 0},
                             c1_init_MAPSO = 2.0, c2_init_MAPSO = 2.0, w_init_MAPSO = 0.9,
                             sigma_min_MAPSO = 0.1, sigma_max_MAPSO = 1,
                             dynamic_topology_MAPSO = False,
                             num_neighbors_init_MAPSO = None, num_neighbors_final_MAPSO = None,
                             num_neighbors_mid_MAPSO = None,
                             maxiter_ccopt = 1000, rho0_ccopt = None,
                             th_ccopt = 1e-6, c_gauge_ccopt = 0,
                             print_params = True):


        self.inference_params = {}
        trainable_params_dict = {}
        trainable_mask_dict = {}

        if trainable_parameters != "all" and trainable_parameters != "elementwise_mask":
            if not isinstance(trainable_parameters, list):
                raise ValueError("trainable_parameters must be a list of strings.")
            for key in trainable_parameters:
                if key not in self.GPModel.param_names and key != "psi":
                    raise ValueError(f"Parameter {key} in trainable_parameters is not a valid parameter. Valid parameters are among {self.GPModel.param_names + ['psi']} or 'all'.")
                
            for key in self.GPModel.param_names:
                trainable_mask_dict[key] = None
                if key in trainable_parameters:
                    trainable_params_dict[key] = True
                else:
                    trainable_params_dict[key] = False
            
            trainable_mask_dict["psi"] = None
            if "psi" in trainable_parameters:
                trainable_params_dict["psi"] = True
            else:
                trainable_params_dict["psi"] = False

        elif trainable_parameters == "all":
            for key in self.GPModel.param_names:
                trainable_params_dict[key] = True
                trainable_mask_dict[key] = None

            trainable_params_dict["psi"] = True
            trainable_mask_dict["psi"] = None

        elif trainable_parameters == "elementwise_mask":
            # if there is a trainable mask, all parameters are considered generally trainable, and elementwise masks are applied
            if trainable_mask is None:
                raise ValueError("trainable_params_mask must be provided if trainable_parameters is 'elementwise_mask'.")
            
            for key in self.GPModel.param_names:
                trainable_params_dict[key] = True

                if key not in trainable_mask:
                    trainable_mask_dict[key] = None
                if isinstance(trainable_mask[key], np.ndarray):
                    if trainable_mask[key].shape != getattr(self.GPModel, key).shape:
                        raise ValueError(f"Shape of {key} in trainable_params_mask is not the same as the shape of the parameter.")
                    trainable_mask_dict[key] = trainable_mask[key]
                elif isinstance(trainable_mask[key], torch.Tensor):
                    if trainable_mask[key].shape != getattr(self.GPModel, key).shape:
                        raise ValueError(f"Shape of {key} in trainable_params_mask is not the same as the shape of the parameter.")
                    trainable_mask_dict[key] = trainable_mask[key].cpu().numpy()
                else:
                    raise ValueError(f"trainable_params_mask must be a numpy array or a torch tensor.")

            trainable_params_dict["psi"] = True

            if "psi" not in trainable_mask:
                trainable_mask_dict["psi"] = None
            elif isinstance(trainable_mask["psi"], np.ndarray):
                if trainable_mask["psi"].shape != self.psi.shape:
                    raise ValueError(f"Shape of psi in trainable_params_mask is not the same as the shape of the parameter.")
                trainable_mask_dict["psi"] = trainable_mask["psi"]
            elif isinstance(trainable_mask["psi"], torch.Tensor):
                if trainable_mask["psi"].shape != self.psi.shape:
                    raise ValueError(f"Shape of psi in trainable_params_mask is not the same as the shape of the parameter.")
                trainable_mask_dict["psi"] = trainable_mask["psi"].cpu().numpy()
            else:
                raise ValueError(f"trainable_params_mask must be a numpy array or a torch tensor.")

        # check if the dictionary is empty
        if not trainable_params_dict:
            raise ValueError("No trainable parameters have been selected. At least one parameter must be selected")

        self.inference_params['trainable_parameters'] = trainable_params_dict
        self.inference_params['trainable_mask'] = trainable_mask_dict

        if use_gradient is False and use_MAPSO is False:
            raise ValueError("At least one of use_gradient or use_MAPSO must be True.")
        else:
            self.inference_params['use_gradient'] = use_gradient
            self.inference_params['use_MAPSO'] = use_MAPSO
            self.inference_params['use_ccopt'] = use_ccopt
        
        if use_gradient is True:
            assert NEpochs_gradient is not None, "Number of epochs must be provided if gradient descent is used."
            assert NEpochs_gradient > 0, "Number of epochs must be greater than 0."
            assert NBatch_gradient is not None, "Number of batches must be provided if MAPSO is not used."
            assert NBatch_gradient > 0, "Number of batches must be greater than 0."
            assert lr_gradient is not None, "Learning rate must be provided if MAPSO is not used."

            self.inference_params['NEpochs_gradient'] = NEpochs_gradient
            self.inference_params['NBatch_gradient'] = NBatch_gradient
            self.inference_params['lr_gradient'] = lr_gradient

            if optimizer_gradient not in self.supported_optimizers:
                err_msg = "Optimizer not supported. Supported optimizers are: "
                err_msg += ", ".join(self.supported_optimizers)
                raise ValueError(err_msg)
            
            assert isinstance(scheduler_gradient, dict), "Scheduler must be a dictionary."
            if 'type' not in scheduler_gradient:
                raise ValueError("Type of scheduler must be provided in scheduler_gradient.")
            
            if scheduler_gradient["type"] not in self.supported_schedulers:
                err_msg = "Scheduler not supported. Supported schedulers are: "
                err_msg += ", ".join(self.supported_schedulers)
                raise ValueError(err_msg)

            self.inference_params['optimizer_gradient'] = optimizer_gradient
            self.inference_params['scheduler_gradient'] = scheduler_gradient
            self.inference_params['train_split_gradient'] = train_split_gradient
            
        if use_MAPSO:
            assert n_particles_MAPSO is not None, "Number of particles for MAPSO must be provided if MAPSO is used."
            assert NEpochs_MAPSO is not None, "Number of iterations for MAPSO must be provided if MAPSO is used."

            assert NEpochs_MAPSO is not None, "Number of iterations for MAPSO must be provided if MAPSO is used."
            assert NEpochs_MAPSO > 0, "Number of iterations for MAPSO must be greater than 0."
            assert n_particles_MAPSO is not None, "Number of particles for MAPSO must be provided if MAPSO is used."
            assert n_particles_MAPSO > 0, "Number of particles for MAPSO must be greater than 0."

            self.inference_params['NEpochs_MAPSO'] = NEpochs_MAPSO
            self.inference_params['n_particles_MAPSO'] = n_particles_MAPSO

            assert isinstance(init_particles_MAPSO, dict), "init_particles_MAPSO must be a dictionary."
            
            if 'distribution' not in init_particles_MAPSO:
                err_msg = "Distribution for initial particles must be provided in init_particles_MAPSO. Supported distributions are: "
                err_msg += ", ".join(self.supported_particle_distributions)
                raise ValueError(err_msg)
            
            if init_particles_MAPSO['distribution'] == "uniform":
                assert 'xmin' in init_particles_MAPSO, "xmin must be provided for an initial uniform distribution of particles in MAPSO."
                assert 'xmax' in init_particles_MAPSO, "xmax must be provided for an initial uniform distribution of particles in MAPSO."
            elif init_particles_MAPSO['distribution'] == "normal":
                assert 'mean' in init_particles_MAPSO, "mean must be provided for an initial gaussian distribution of particles in MAPSO."
                assert 'std' in init_particles_MAPSO, "std must be provided for an initial gaussian distribution of particles in MAPSO."
            elif init_particles_MAPSO['distribution'] == "uniform_with_biases":
                assert 'xmin' in init_particles_MAPSO, "xmin must be provided for an initial uniform distribution of particles in MAPSO."
                assert 'xmax' in init_particles_MAPSO, "xmax must be provided for an initial uniform distribution of particles in MAPSO."
                assert 'biases' in init_particles_MAPSO, "biases must be provided for an initial uniform distribution of particles in MAPSO."
                if isinstance(init_particles_MAPSO['biases'], list):
                    init_particles_MAPSO['biases'] = np.array(init_particles_MAPSO['biases'])
                if isinstance(init_particles_MAPSO['biases'], float) or init_particles_MAPSO['biases'].size == 1:
                    init_particles_MAPSO['biases'] = np.ones(self.GPModel.dim_params + self.M) * init_particles_MAPSO['biases']
                else:
                    if init_particles_MAPSO['biases'].shape != (self.GPModel.dim_params + self.M,):
                        raise ValueError(f"Biases must be a vector of size {self.GPModel.dim_params + self.M}.")
            elif init_particles_MAPSO['distribution'] == "multivariate_normal":
                assert 'mean' in init_particles_MAPSO, "mean must be provided for an initial multivariate normal distribution of particles in MAPSO."
                assert 'cov' in init_particles_MAPSO, "cov must be provided for an initial multivariate normal distribution of particles in MAPSO."
                if isinstance(init_particles_MAPSO['mean'], list):
                    init_particles_MAPSO['mean'] = np.array(init_particles_MAPSO['mean'])
                if isinstance(init_particles_MAPSO['cov'], list):
                    init_particles_MAPSO['cov'] = np.array(init_particles_MAPSO['cov'])
                assert type(init_particles_MAPSO['mean']) in [np.ndarray, torch.Tensor], "Mean of particle initialization must be a numpy array or a torch tensor."
                assert type(init_particles_MAPSO['cov']) in [np.ndarray, torch.Tensor], "Covariance of particle initialization must be a numpy array or a torch tensor."
                assert init_particles_MAPSO['mean'].shape == (self.GPModel.dim_params + self.M,), \
                    f"Mean must be a vector of size {self.GPModel.dim_params + self.M}."
                assert init_particles_MAPSO['cov'].shape == (self.GPModel.dim_params + self.M, self.GPModel.dim_params + self.M), \
                    f"Covariance must be a matrix of size {(self.GPModel.dim_params + self.M, self.GPModel.dim_params + self.M)}."

            self.inference_params['init_particles_MAPSO'] = init_particles_MAPSO

            assert isinstance(init_velocities_MAPSO, dict), "init_velocities_MAPSO must be a dictionary."

            if 'distribution' not in init_velocities_MAPSO:
                err_msg = "Distribution for initial velocities must be provided in init_velocities_MAPSO. Supported distributions are: "
                err_msg += ", ".join(self.supported_velocity_distributions)
                raise ValueError(err_msg)
            
            if init_velocities_MAPSO['distribution'] == "uniform":
                assert 'vmin' in init_velocities_MAPSO, "vmin must be provided for an initial uniform distribution of velocities in MAPSO."
                assert 'vmax' in init_velocities_MAPSO, "vmax must be provided for an initial uniform distribution of velocities in MAPSO."
            elif init_velocities_MAPSO['distribution'] == "normal":
                assert 'mean' in init_velocities_MAPSO, "mean must be provided for an initial gaussian distribution of velocities in MAPSO."
                assert 'std' in init_velocities_MAPSO, "std must be provided for an initial gaussian distribution of velocities in MAPSO."

            self.inference_params['init_velocities_MAPSO'] = init_velocities_MAPSO

            self.inference_params['c1_init_MAPSO'] = c1_init_MAPSO
            self.inference_params['c2_init_MAPSO'] = c2_init_MAPSO
            self.inference_params['w_init_MAPSO'] = w_init_MAPSO
            self.inference_params['sigma_min_MAPSO'] = sigma_min_MAPSO
            self.inference_params['sigma_max_MAPSO'] = sigma_max_MAPSO

            if dynamic_topology_MAPSO:
                assert num_neighbors_init_MAPSO is not None, "Initial number of neighbors for dynamic topology must be provided if dynamic topology is used. Specify num_neighbors_init_MAPSO."
                assert num_neighbors_final_MAPSO is not None, "Final number of neighbors for dynamic topology must be provided if dynamic topology is used. Specify num_neighbors_final_MAPSO."
                assert num_neighbors_mid_MAPSO is not None, "Midpoint number of neighbors for dynamic topology must be provided if dynamic topology is used. Specify num_neighbors_mid_MAPSO."

                assert num_neighbors_init_MAPSO > 0, "Initial number of neighbors for dynamic topology must be greater than 0."
                assert num_neighbors_final_MAPSO > 0, "Final number of neighbors for dynamic topology must be greater than 0."
                assert num_neighbors_mid_MAPSO > 0, "Midpoint number of neighbors for dynamic topology must be greater than 0."

                assert num_neighbors_init_MAPSO <= n_particles_MAPSO, "Initial number of neighbors for dynamic topology must be less than or equal to the number of particles."
                assert num_neighbors_final_MAPSO <= n_particles_MAPSO, "Final number of neighbors for dynamic topology must be less than or equal to the number of particles."
                assert num_neighbors_mid_MAPSO <= n_particles_MAPSO, "Midpoint number of neighbors for dynamic topology must be less than or equal to the number of particles."

                self.inference_params['num_neighbors_init_MAPSO'] = num_neighbors_init_MAPSO
                self.inference_params['num_neighbors_final_MAPSO'] = num_neighbors_final_MAPSO
                self.inference_params['num_neighbors_mid_MAPSO'] = num_neighbors_mid_MAPSO
            else:
                self.inference_params['num_neighbors_init_MAPSO'] = None
                self.inference_params['num_neighbors_final_MAPSO'] = None
                self.inference_params['num_neighbors_mid_MAPSO'] = None

            self.inference_params['dynamic_topology_MAPSO'] = dynamic_topology_MAPSO

        if use_ccopt:
            self.inference_params['maxiter_ccopt'] = maxiter_ccopt
            self.inference_params['rho0_ccopt'] = rho0_ccopt
            self.inference_params['th_ccopt'] = th_ccopt
            self.inference_params['c_gauge_ccopt'] = c_gauge_ccopt

        if print_params:
            for key, value in self.inference_params.items():
                if key == "trainable_parameters":
                    print("===========================================")
                    print("=== Trainable parameters for inference ===")
                    print("===========================================")
                    for parname, trainable in value.items():
                        print(f"{parname}: {trainable}")
                    print("===========================================")
                    print()
                if key == "trainable_mask":
                    print("================================================")
                    print("=== Trainable parameters masks for inference ===")
                    print("================================================")
                    for parname, mask in value.items():
                        if mask is not None:
                            print(f"{parname}: {mask}")
                        else:
                            print(f"{parname}: None")
                    print("===========================================")
                    print()
                if key == "NEpochs_gradient" and use_gradient:
                    print("===========================================")
                    print("=== Gradient-based inference parameters ===")
                    print("===========================================")

                if key == "NEpochs_MAPSO" and use_MAPSO:
                    print("========================================")
                    print("=== Particle swarm (MAPSO) parameters ===")
                    print("========================================")

                if key == "maxiter_ccopt":
                    print("=======================================================")
                    print("=== Discrete convex-concave optimization parameters ===")
                    print("=======================================================")

                if key != "use_gradient" and key != "use_MAPSO" and key != "use_ccopt" and key != "trainable_parameters" and key != "trainable_mask":
                    print(f"{key}: {value}")

                if key == "train_split_gradient":
                    print("===========================================")
                    # if it's not the last key, add a separator
                    if key != list(self.inference_params.keys())[-1]:
                        print()

                if key == "dynamic_topology_MAPSO":
                    print("========================================")
                    if key != list(self.inference_params.keys())[-1]:
                        print()

                if key == "c_gauge_ccopt":
                    print("=======================================================")
                    if key != list(self.inference_params.keys())[-1]:
                        print()

    def fit(self, trajectories, overwrite = True, verbose_MAPSO = True, verbose_epochs_MAPSO = True):
        assert not self.trained, "The model has already been trained. If you want to train it again, reinitialize it or call the method reset_trained_status()."

        self.initialize_for_inference()
        self.load_trajectories_tofit(trajectories)
        
        self.__check_ready_for_inference()


        if not hasattr(self, 'inference_params'):
            raise ValueError("Inference parameters must be set before fitting. Use set_inference_params method to set the parameters.")
        
        losses_epochs = self.inferencer.optimize(self.inference_params, verbose = verbose_MAPSO, verbose_epochs = verbose_epochs_MAPSO)
                
        if overwrite:
            inferred_params = {}
            for idx, p in enumerate(self.inferencer.get_inferred_policy_params()):
                inferred_params[self.GPModel.param_names[idx]] = p.detach()#.cpu().numpy().astype(np.float64)

            self.GPModel._load_params(inferred_params)

            self.psi = self.psi.detach()#.cpu().numpy().astype(np.float64)

            self.losses_epochs = losses_epochs
            self.best_loss = self.inferencer.best_loss
            self.__trained = True

        return losses_epochs

    ##############################################################################################################################



    ##############################################################################################################################
    ############### Methods to load parameters and trajectories ##################################################################
    ##############################################################################################################################
    
    def load_parameters(self, policy_params, psi):

        self.__check_parameters_consistency(psi)
        self.GPModel._load_params(policy_params)

        if self.mode == 'inference':
            for parname in self.GPModel.param_names:
                if isinstance(getattr(self.GPModel, parname), np.ndarray):
                    setattr(self.GPModel, parname, torch.tensor(getattr(self.GPModel, parname), dtype=torch.float32))

            if isinstance(psi, np.ndarray):
                psi = torch.tensor(psi, dtype=torch.float32)

        elif self.mode == 'generation':
            for parname in self.GPModel.param_names:
                if isinstance(getattr(self.GPModel, parname), torch.Tensor):
                    setattr(self.GPModel, parname, getattr(self.GPModel, parname).detach().cpu().numpy().astype(np.float64))

            if isinstance(psi, torch.Tensor):
                psi = psi.detach().cpu().numpy().astype(np.float64)

        self.psi = psi

    def load_observations_for_generation(self, observations):
        if self.mode != 'generation':
            raise ValueError("Mode must be 'generation' to load observations.")

        self.generator.load_observations(observations)

    def load_trajectories_tofit(self, trajectories):

        if self.mode != 'inference':
            raise ValueError("Mode must be 'inference' to load trajectories to be fitted.")
        
        trajectories_to_fit = []

        for trajectory in trajectories:
            assert isinstance(trajectory, dict), "Trajectory must be a dictionary."
            assert "observations" in trajectory, "Trajectory must contain the key 'observations'."
            assert "actions" in trajectory, "Trajectory must contain the key 'actions'."
            assert len(trajectory["observations"]) == len(trajectory["actions"]), "Trajectory must contain the same number of observations and actions."

            trajectories_to_fit.append(trajectory)
        
        self.inferencer.load_trajectories(trajectories_to_fit)
        self.__loaded_trajectories_inference = True

    ##############################################################################################################################



    ##############################################################################################################################
    ############### Methods to print and get information about the FSC ###########################################################
    ##############################################################################################################################

    def show_supported(self):
        print("=============================")
        print("Supported policy models: ", self.supported_policy_models)
        print("Supported optimizers for gradient descent: ", self.supported_optimizers)
        print("Supported schedulers for gradient descent: ", self.supported_schedulers)
        print("Supported particle distributions for MAPSO: ", self.supported_particle_distributions)
        print("Supported velocity distributions for MAPSO: ", self.supported_velocity_distributions)
        print("=============================")

    def show_summary(self):
        print("======================")
        print("=== FSC properties ===")
        print("======================")
        print("Number of memory states: ", self.M)
        print("Number of actions: ", self.A)
        print("Number of observations: ", self.Y)
        print("Policy model: ", self.policy_type)
        print("Mode: ", self.mode)
        print("Trained: ", self.trained)
        print("======================")
        print()
        print("===================================")
        print("=== Policy and memory parameters ===")
        print("===================================")
        if self._init_memory_obs_dependent:
            print("Initial memory state is observation-dependent.")
        else:
            print("Initial memory state is not observation-dependent.")
        print()
        for key in self.GPModel.param_names:
            print(f"{key}: {getattr(self.GPModel, key)}")
        print("Psi: ", self.psi)
        if not self._init_memory_obs_dependent:
            print("Initial memory distribution: ", self.rho)
        print("===================================")

    ##############################################################################################################################



    ##############################################################################################################################
    ############### Methods for generating trajectories ##########################################################################
    ##############################################################################################################################

    def generate_single_trajectory(self,
                                   environment_model=None,
                                   NSteps=None, observations=None,
                                   idx_observation=None, initial_state=None,
                                   verbose=False):
        if self.mode != 'generation':
            raise ValueError("Mode must be 'generation' to generate a trajectory.")
        
        use_environment_model = False
        use_user_observations = False

        if environment_model is not None:
            if idx_observation is not None or observations is not None:
                raise ValueError("environment_model cannot be used when generating a trajectory from observations.")
            else:
                if NSteps is None:
                    raise ValueError("NSteps must be provided when using environment_model.")

                if environment_model.A != self.A:
                    raise ValueError("The number of actions in the environment model must be the same as the number of actions in the FSC.")
                if np.all(environment_model.ActSpace != self.ActSpace):
                    raise ValueError("The action space in the environment model must be the same as the action space in the FSC.")
                if np.all(environment_model.obs_type != "discrete"):
                    raise ValueError("The observation type in the environment model must be the same as the observation type in the FSC.")

                use_environment_model = True
        elif observations is not None:
            use_user_observations = True

        if use_environment_model:
            gen_trj = self.generator.generate_single_trajectory_from_environment(environment_model, NSteps, initial_state=initial_state)
        else:
            if use_user_observations:
                obs_vals = observations
            else:
                obs_vals = None

            gen_trj = self.generator.generate_single_trajectory(NSteps, obs_vals, idx_observation)

            if use_user_observations:
                gen_trj["observations"] = observations
            else:
                gen_trj["observations"] = self.generator.observations[0]

        return gen_trj

    def generate_trajectories(self,
                              environment_model = None,
                              obs_from_act = False,
                              NSteps = None, observations = None,
                              idx_observation = None, NTraj = None,
                              initial_states = None,
                              verbose = False):
        if self.mode != 'generation':
            raise ValueError("Mode must be 'generation' to generate trajectories.")
        
        use_environment_model = False
        use_user_observations = False

        if environment_model is not None:
            if not isinstance(environment_model, BaseEnvironmentModel):
                raise ValueError("environment_model must be an instance of BaseEnvironmentModel.")
            
            if np.all(environment_model.observation_type != "discrete"):
                raise ValueError("The observation type in the environment model must be the same as the observation type in the FSC.")

            if environment_model.depends_on_action:
                if environment_model.A != self.A:
                    raise ValueError("The number of actions in the environment model must be the same as the number of actions in the FSC.")
                if np.all(environment_model.ActSpace != self.ActSpace):
                    raise ValueError("The action space in the environment model must be the same as the action space in the FSC.")

            if idx_observation is not None or observations is not None:
                raise ValueError("environment_model cannot be used when generating trajectories from observations.")
            else:
                if NSteps is None:
                    raise ValueError("NSteps must be provided when using environment_model.")
                if NTraj is None:
                    raise ValueError("NTraj must be provided when using environment_model. If you want to generate a single trajectory, use the method generate_single_trajectory.")

            use_environment_model = True

        elif observations is not None:
            use_user_observations = True


        if use_environment_model:
            if obs_from_act:
                gen_trj = self.generator.generate_trajectories_from_environment_obs_from_act(environment_model, NSteps, NTraj, initial_states = initial_states)
            else:
                gen_trj = self.generator.generate_trajectories_from_environment(environment_model, NSteps, NTraj, initial_states = initial_states)
        else:
            if use_user_observations:
                obs_vals = observations
            else:
                obs_vals = None

            gen_trj = self.generator.generate_trajectories(NSteps, obs_vals, idx_observation, NTraj, verbose)
                
            if use_user_observations:
                for i in range(len(gen_trj)):
                    if "observations" not in gen_trj[i]:
                        gen_trj[i]["observations"] = observations[i]
            else:
                for i in range(len(gen_trj)):
                    if "observations" not in gen_trj[i]:
                        curr_obs = self.generator.observations[i]
                        gen_trj[i]["observations"] = [self.ObsSpace[obs] for obs in curr_obs]
                        
        return gen_trj
    ##############################################################################################################################



    ##############################################################################################################################
    ############### Methods for plotting results #################################################################################
    ##############################################################################################################################

    def plot_losses(self, ax = None, figsize = (5, 3), return_ax = False):
        assert self.trained, "The model has not been trained yet. Call the fit method to train the model."

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize = figsize)

        # check if losses_epochs has the key "MAPSO"
        if self.inference_params['use_MAPSO'] and self.inference_params['use_gradient']:
            ax.axvline(x = self.inference_params['NEpochs_MAPSO'], c = 'gray', ls = '--', lw = 1, label = "MAPSO end", zorder = -1)
        
        ax.plot(np.arange(1, len(self.losses_epochs["train"]) + 1, 1), self.losses_epochs["train"], c = 'k', lw = 2)
        ax.axvline(x = np.argmin(self.losses_epochs["train"]) + 1, c = 'k', ls = '--', lw = 1, label = "Best training loss", zorder = -1)

        if "val" in self.losses_epochs:
            ax.plot(np.arange(1, len(self.losses_epochs["train"]) + 1, 1), self.losses_epochs["val"], c = 'darkred', lw = 2)
            
            ax.axvline(x = np.argmin(self.losses_epochs["val"]),
                       c = 'darkred', ls = '--', lw = 1, label = "Best validation loss", zorder = -1)

        ax.legend(loc = 'upper right')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Negative log-likelihood")
        ax.set_title("Negative log-likelihood during training")

        if return_ax:
            return fig, ax

    def plot_FSC(self, observation_node_colors,
                 memory_ordering = None,
                 **kwargs):
        """
        Plot the FSC structure.
        
        Args:
            observation_node_colors: Colors for observation nodes (required)
            **kwargs: All other plotting parameters (see utils.draw_FSC_complex_network for options)
        """
        
        # Set sensible defaults
        defaults = {
            'th_mem_transitions': 1e-3,
            'th_action_probs': 1e-32,
            'arrangement': "horizontal",
            'spacing': 6,
            'mem_node_size': 5000,
            'action_node_size': 1500,
            'obs_node_size': 500,
            'action_node_yoffset': 2.5,
            'min_weightsum_threshold': 0.01,
            'max_action_width': 5,
            'min_action_width': 0.,
            'max_line_width': 5,
            'min_line_width': 0.,
            'suppress_zero_action_transitions': True,
            'action_prob_threshold': 1e-10,
            'fade_no_incoming': True,
            'no_incoming_alpha': 0.1,
            'arrowhead_distance': 35,
            'arrowhead_distance_actions': 20,
            'action_edge_color': "black",
            'memory_node_color': 'lightgray',
            'action_node_color': 'darkgray',
            'return_graph': False,
        }

        if self.mode != 'generation':
            raise ValueError("Mode must be 'generation' to plot the FSC structure.")
        
        # Merge with user inputs
        config = {**defaults, **kwargs}
        
        # Validate required parameter
        assert len(observation_node_colors) == self.Y, \
            "Length of observation_node_colors must be equal to the number of observations."
        
        # Set name defaults if not provided
        config.setdefault('action_names', [str(a) for a in self.ActSpace])
        config.setdefault('observation_names', [str(o) for o in self.ObsSpace])
        
        memory_names = [str(m) for m in self.MemSpace]
        if memory_ordering is not None:
            if len(memory_ordering) != self.M:
                raise ValueError("Length of memory_ordering must be equal to the number of memory states.")
            memory_names = [memory_names[i] for i in memory_ordering]
        config.setdefault('memory_names', memory_names)
        
        # Compute transition weights and action probabilities
        transition_weights = self.get_memory_transitions()
        transition_weights[transition_weights < config['th_mem_transitions']] = 0
        
        action_probabilities = self.get_action_policy()
        action_probabilities[action_probabilities < config['th_action_probs']] = 0
        
        # Extract drawing parameters and call the function
        draw_params = {k: v for k, v in config.items() 
                    if k not in ['th_mem_transitions', 'th_action_probs', 'return_graph']}
        
        G, ax = utils.draw_FSC_complex_network(
            transition_weights, 
            action_probabilities=action_probabilities,
            observation_node_colors=observation_node_colors,
            memory_ids=memory_ordering,
            **draw_params
        )
        
        if config['return_graph']:
            return G, ax
        else:
            return ax


    def plot_bipartite_FSC(self, observation_node_colors,
                           memory_ordering = None,
                           **kwargs):        
        # Set sensible defaults
        defaults = {
            'th_mem_transitions': 1e-3,
            'th_action_probs': 1e-2,
            'arrangement': "horizontal",
            'spacing': 20,
            'layer_spacing': 10,
            'mem_node_size': 5000,
            'action_node_size': 1500,
            'obs_node_size': 500,
            'obs_node_xoffset': 2.5,
            'obs_node_yoffset': 0.5,
            'action_node_xoffset': 10,
            'action_node_yoffset': 10.5,
            'min_weightsum_threshold': 0.01,
            'max_action_width': 5,
            'min_action_width': 0.,
            'max_line_width': 5,
            'min_line_width': 0.,
            'suppress_zero_action_transitions': False,
            'action_prob_threshold': 1e-10,
            'fade_no_incoming': True,
            'no_incoming_alpha': 0.1,
            'arrowhead_distance': 35,
            'arrowhead_distance_actions': 20,
            'action_edge_color': "black",
            'memory_node_color': 'lightgray',
            'action_node_color': 'darkgray',
            "reverse_obs_below": False,
            "hide_unused_actions": True,
            "AllowedObsFromAct": None,
            "obs_rotation": np.pi/4 * 3,
            'return_graph': False,
        }

        if self.mode != 'generation':
            raise ValueError("Mode must be 'generation' to plot the FSC structure.")
        
        # Merge with user inputs
        config = {**defaults, **kwargs}
        
        # Validate required parameter
        assert len(observation_node_colors) == self.Y, \
            "Length of observation_node_colors must be equal to the number of observations."
        
        # Set name defaults if not provided
        config.setdefault('action_names', [str(a) for a in self.ActSpace])
        config.setdefault('observation_names', [str(o) for o in self.ObsSpace])

        memory_names = [str(m) for m in self.MemSpace]
        if memory_ordering is not None:
            if len(memory_ordering) != self.M:
                raise ValueError("Length of memory_ordering must be equal to the number of memory states.")
            memory_names = [memory_names[i] for i in memory_ordering]
        config.setdefault('memory_names', memory_names)

        # Compute transition weights and action probabilities
        transition_weights = self.get_memory_transitions()
        transition_weights[transition_weights < config['th_mem_transitions']] = 0
        
        action_probabilities = self.get_action_policy()
        action_probabilities[action_probabilities < config['th_action_probs']] = 0
        
        # Extract drawing parameters and call the function
        draw_params = {k: v for k, v in config.items() 
                    if k not in ['th_mem_transitions', 'th_action_probs', 'return_graph']}

        G, ax = utils.draw_bipartite_FSC_network(
            transition_weights, 
            action_probabilities=action_probabilities,
            observation_node_colors=observation_node_colors,
            memory_ids=memory_ordering,
            **draw_params
        )
        
        if config['return_graph']:
            return G, ax
        else:
            return ax


    def plot_twolayers_FSC(self, observation_node_colors,
                           memory_ordering = None,
                           second_layer_nodes = None,
                           **kwargs):        
        """
        Plot the FSC structure with a two-layer arrangement.
        
        Args:
            observation_node_colors: Colors for observation nodes (required)
            memory_ordering: Order of memory nodes (optional)
            second_layer_nodes: List of 2 node indices for the second layer (optional, defaults to last two)
            **kwargs: All other plotting parameters (see utils.draw_twolayers_FSC_network for options)
        """
        # Set sensible defaults
        defaults = {
            'th_mem_transitions': 1e-3,
            'th_action_probs': 1e-2,
            'arrangement': "horizontal",
            'spacing': 20,
            'layer_spacing': 10,
            'mem_node_size': 5000,
            'action_node_size': 1500,
            'obs_node_size': 500,
            'obs_node_xoffset': 2.5,
            'obs_node_yoffset': 0.5,
            'action_node_xoffset': 10,
            'action_node_yoffset': 10.5,
            'min_weightsum_threshold': 0.01,
            'max_action_width': 5,
            'min_action_width': 0.,
            'max_line_width': 5,
            'min_line_width': 0.,
            'suppress_zero_action_transitions': False,
            'action_prob_threshold': 1e-10,
            'fade_no_incoming': True,
            'no_incoming_alpha': 0.1,
            'arrowhead_distance': 35,
            'arrowhead_distance_actions': 20,
            'action_edge_color': "black",
            'memory_node_color': 'lightgray',
            'action_node_color': 'darkgray',
            "reverse_obs_below": False,
            "hide_unused_actions": True,
            "AllowedObsFromAct": None,
            "obs_rotation": np.pi/4 * 3,
            'return_graph': False,
        }

        if self.mode != 'generation':
            raise ValueError("Mode must be 'generation' to plot the FSC structure.")
        
        # Merge with user inputs
        config = {**defaults, **kwargs}
        
        # Validate required parameter
        assert len(observation_node_colors) == self.Y, \
            "Length of observation_node_colors must be equal to the number of observations."
        
        # Set name defaults if not provided
        config.setdefault('action_names', [str(a) for a in self.ActSpace])
        config.setdefault('observation_names', [str(o) for o in self.ObsSpace])

        memory_names = [str(m) for m in self.MemSpace]
        if memory_ordering is not None:
            if len(memory_ordering) != self.M:
                raise ValueError("Length of memory_ordering must be equal to the number of memory states.")
            memory_names = [memory_names[i] for i in memory_ordering]
        config.setdefault('memory_names', memory_names)

        # Compute transition weights and action probabilities
        transition_weights = self.get_memory_transitions()
        transition_weights[transition_weights < config['th_mem_transitions']] = 0
        
        action_probabilities = self.get_action_policy()
        action_probabilities[action_probabilities < config['th_action_probs']] = 0
        
        # Extract drawing parameters and call the function
        draw_params = {k: v for k, v in config.items() 
                    if k not in ['th_mem_transitions', 'th_action_probs', 'return_graph']}

        G, ax = utils.draw_twolayers_FSC_network(
            transition_weights, 
            action_probabilities=action_probabilities,
            observation_node_colors=observation_node_colors,
            memory_ids=memory_ordering,
            second_layer_nodes=second_layer_nodes,
            **draw_params
        )
        
        if config['return_graph']:
            return G, ax
        else:
            return ax

    ##############################################################################################################################



    ##############################################################################################################################
    ############### Private methods and initialization functions #################################################################
    ##############################################################################################################################

    def __check_parameters_consistency(self, psi):
        if self._init_memory_obs_dependent:
            if psi is not None:
                if psi.shape != (self.Y, self.M):
                    raise ValueError("psi must have shape (Y, M) where M is the number of states and Y is the number of observations.")
        else:
            if psi is not None:
                if psi.shape != (self.M,):
                    raise ValueError("psi must have shape (M,) where M is the number of states.")
        
    def __check_and_init_observations(self, Y):
        
        if Y is None:
            raise ValueError("Please provide the number of possible observations Y.")
        if Y <= 0:
            raise ValueError("Y must be a positive integer.")
        if Y != int(Y):
            raise ValueError("Y must be an integer.")            

    def __init_generalized_policy_model(self, policy_model,
                                        policy_params, seed):        
        if policy_model not in self.supported_policy_models:
            raise ValueError("Transition model not supported. Supported policy models are: " + ", ".join(self.supported_policy_models))
            
        if policy_model == "softmax":
            self.GPModel = parametrizations.SoftmaxDiscreteObs.GeneralizedPolicyModel("discrete",
                                                                                    M = self.M, A = self.A, Y = self.Y,
                                                                                    seed = seed,
                                                                                    policy_params = policy_params)

        else:
            raise ValueError("Transition model not supported for the given observation type.")

    def __init_supported_objects(self):
        self.__supported_policy_models_discrete = ["softmax"]

        self.__supported_optimizers = ["ADAM", "SGD"]
        self.__supported_schedulers = ["exponential", "fixed"]

        self.__supported_particle_distributions = ["uniform", "normal", "multivariate_normal", "uniform_with_biases"]
        self.__supported_velocity_distributions = ["uniform", "normal"]


    def __initialize_structure(self, M, A, Y):
        self.M = M
        self.A = A
        self.Y = Y

    def __initialize_psi(self, seed):
        if seed is not None:
            np.random.seed(seed)
        if self._init_memory_obs_dependent:
            self.psi = np.random.randn(self.Y, self.M)
        else:
            self.psi = np.random.randn(self.M)

    def __check_ready_for_inference(self):
        if self.mode != 'inference':
            raise ValueError("Mode must be 'inference' to perform inference.")
        if not self.__loaded_trajectories_inference:
            raise ValueError("No trajectories to fit have been loaded.")
        if not hasattr(self, 'inferencer'):
            raise ValueError("Inferencer has not been initialized.")
        
    def __initialize_spaces(self, ObsSpace, ActSpace, MemSpace):
        if ObsSpace is not None:
            if len(ObsSpace) != self.Y:
                raise ValueError("The number of observations in ObsSpace must match the number of observations.")
            self.ObsSpace = np.array(ObsSpace)
            self.custom_obs_space = True
        else:
            self.ObsSpace = np.arange(self.Y)
            self.custom_obs_space = False

        if ActSpace is not None:
            if len(ActSpace) != self.A:
                raise ValueError("The number of actions in ActSpace must match the number of actions.")
            self.ActSpace = np.array(ActSpace)
            self.custom_act_space = True
        else:
            self.ActSpace = np.arange(self.A)
            self.custom_act_space = False

        if MemSpace is not None:
            if len(MemSpace) != self.M:
                raise ValueError("The number of states in MemSpace must match the number of states.")
            self.MemSpace = np.array(MemSpace)
            self.custom_mem_space = True
        else:
            self.MemSpace = np.arange(self.M)
            self.custom_mem_space = False

    ##############################################################################################################################



    ##############################################################################################################################
    ############### Properties ###################################################################################################
    ##############################################################################################################################

    @property
    def supported_policy_models(self):
        return self.__supported_policy_models_discrete
        
    @property
    def supported_optimizers(self):
        return self.__supported_optimizers
    
    @property
    def supported_schedulers(self):
        return self.__supported_schedulers
    
    @property
    def supported_particle_distributions(self):
        return self.__supported_particle_distributions
    
    @property
    def supported_velocity_distributions(self):
        return self.__supported_velocity_distributions

    @property
    def obs_type(self):
        return "discrete"
    
    @property
    def rho(self): 
        if self._init_memory_obs_dependent:
            if isinstance(self.psi, torch.Tensor):
                return nn.Softmax(dim=1)(self.psi)
            else:
                return utils.softmax(self.psi, axis=1)
        else:
            if isinstance(self.psi, torch.Tensor):
                return nn.Softmax(dim=0)(self.psi)
            else:
                return utils.softmax(self.psi)
    
    @property
    def trajectories_tofit(self):
        if not self.__loaded_trajectories_inference:
            raise ValueError("No trajectories to fit have been loaded.")

        return self.inferencer.ObsAct_trajectories
        
    @property
    def policy_params(self):
        param_dict = {}

        for key in self.GPModel.param_names:
            param_dict[key] = getattr(self.GPModel, key)

        return param_dict
        
    @property
    def policy_type(self):
        return self.GPModel.name
    
    @property
    def trained(self):
        return self.__trained

    ##############################################################################################################################