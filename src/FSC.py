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
                 policy_model = "softmax",
                 policy_params = {"theta": None, "zeta": None},
                 psi = None, seed = None,
                 ObsSpace = None, ActSpace = None, MemSpace = None,
                 init_memory_obs_dependent = False):
        """
        Initialize a Finite State Controller (FSC) for either generation or inference.

        Parameters
        ----------
        mode : str
            Operating mode. Must be ``'generation'`` (sample trajectories) or
            ``'inference'`` (fit parameters to observed trajectories).
        M : int
            Number of internal memory states.
        A : int
            Number of actions.
        Y : int
            Number of discrete observations.
        policy_model : str, optional
            Name of the generalized policy parametrization. Currently only
            ``'softmax'`` is supported.
        policy_params : dict, optional
            Dictionary of initial policy parameter arrays. Keys must match the
            parameter names expected by the chosen policy model (``'theta'`` for
            memory transitions of shape ``(Y, A, M, M)`` and ``'zeta'`` for the
            action policy of shape ``(A, M)``). Pass ``None`` for a parameter to
            have it randomly initialized.
        psi : np.ndarray or None, optional
            Logit vector for the initial memory distribution ``rho``. Shape must
            be ``(M,)`` when ``init_memory_obs_dependent=False``, or ``(Y, M)``
            when ``init_memory_obs_dependent=True``. Randomly initialized if
            ``None``.
        seed : int or None, optional
            Random seed used when randomly initializing ``psi`` and any ``None``
            policy parameters.
        ObsSpace : array-like or None, optional
            Labels for each observation (length ``Y``). Defaults to
            ``np.arange(Y)``.
        ActSpace : array-like or None, optional
            Labels for each action (length ``A``). Defaults to
            ``np.arange(A)``.
        MemSpace : array-like or None, optional
            Labels for each memory state (length ``M``). Defaults to
            ``np.arange(M)``.
        init_memory_obs_dependent : bool, optional
            If ``True``, the initial memory distribution ``rho`` is conditioned
            on the first observation (``psi`` has shape ``(Y, M)``). Default is
            ``False``.

        Raises
        ------
        ValueError
            If ``mode`` is not ``'generation'`` or ``'inference'``.
        ValueError
            If ``Y`` is not provided or not a positive integer.
        ValueError
            If any provided space array has the wrong length.
        """
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
        """
        Return the initial memory distribution ``rho``.

        ``rho`` is derived from ``psi`` via a softmax transformation. When
        ``init_memory_obs_dependent=True`` the result has shape ``(Y, M)``;
        otherwise it has shape ``(M,)``.

        Returns
        -------
        np.ndarray or torch.Tensor
            Probability distribution over initial memory states.
        """
        return self.rho


    def get_TMat(self):
        """
        Return the joint transition matrix ``T(a, m' | m, y)``.

        The matrix encodes the probability of jointly choosing action ``a`` and
        transitioning to memory state ``m'``, given the current memory ``m`` and
        observation ``y``. The returned shape is ``(Y, M, M, A)``, where axes
        are ``(obs, prev_mem, next_mem, action)``.

        Returns
        -------
        np.ndarray
            In ``'generation'`` mode: NumPy array of shape ``(Y, M, M, A)``.
        torch.Tensor
            In ``'inference'`` mode: PyTorch tensor of shape ``(Y, M, M, A)``.
        """
        if self.mode == 'generation':
            return self.GPModel.get_TMat_numpy().transpose(0, 2, 3, 1)
        elif self.mode == 'inference':
            return self.GPModel.get_TMat_torch().permute(0, 2, 3, 1)
            
    def get_action_policy(self, obs = None):
        """
        Return the marginal action policy ``pi(a | m)``.

        In the ``'softmax'`` parametrization the action policy is
        observation-independent, so the ``obs`` argument is accepted for
        interface consistency but is not used.

        Parameters
        ----------
        obs : optional
            Not used. Providing a value raises a warning.

        Returns
        -------
        np.ndarray
            In ``'generation'`` mode: array of shape ``(M, A)``.
        torch.Tensor
            In ``'inference'`` mode: tensor of shape ``(M, A)``.
        """
        if obs is not None:
            raise Warning("Explicit observation values are not used in the policy model for discrete observations.")
        if self.mode == 'generation':
            return self.generator.get_action_policy()
        elif self.mode == 'inference':
            return self.inferencer.get_action_policy()
            
    def get_memory_transitions(self, obs = None):
        """
        Return the memory transition matrix ``g(m' | a, m, y)``.

        Gives the probability of moving to memory ``m'`` given the current
        memory ``m``, action ``a``, and observation ``y``. The ``obs`` argument
        is accepted for interface consistency but is not used in the current
        ``'softmax'`` parametrization.

        Parameters
        ----------
        obs : optional
            Not used. Providing a value raises a warning.

        Returns
        -------
        np.ndarray
            In ``'generation'`` mode: array of shape ``(Y, A, M, M)``.
        torch.Tensor
            In ``'inference'`` mode: tensor of shape ``(Y, A, M, M)``.
        """
        if obs is not None:
            raise Warning("Explicit observation values are not used in the policy model for discrete observations.")
        if self.mode == 'generation':
            return self.generator.get_memory_transition()
        elif self.mode == 'inference':
            return self.inferencer.get_memory_transition()
                        
    def save(self, directory, filename = None,
             custom_postname = ""):
        """
        Serialize the FSC object to a pickle file.

        If ``filename`` is not provided, an automatic name is built from the
        policy model name, the dimensions ``M``, ``A``, ``Y``, the best
        training loss (when trained), and an optional custom suffix.

        Parameters
        ----------
        directory : str
            Target directory. Created recursively if it does not exist.
        filename : str or None, optional
            Output filename (without or with ``.pkl`` extension). Auto-
            generated when ``None``.
        custom_postname : str, optional
            Additional suffix appended to the auto-generated filename before
            the ``.pkl`` extension.
        """
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
        """
        Switch the FSC between ``'generation'`` and ``'inference'`` modes.

        When switching to ``'generation'``, all PyTorch parameters are detached
        and converted to NumPy arrays, and a fresh ``GenerationDiscreteObs``
        instance is created. Previously loaded observations (stored in
        ``_fitted_observations_numpy``) are forwarded to the new generator.
        When switching to ``'inference'``, a fresh ``InferenceDiscreteObs``
        instance is created.

        Parameters
        ----------
        mode : str
            Target mode. Must be ``'generation'`` or ``'inference'``.

        Raises
        ------
        ValueError
            If ``mode`` is not ``'generation'`` or ``'inference'``.
        """
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
        """
        Convert the internally stored observation trajectories back to NumPy
        arrays in the original (external) observation space.

        The converted sequences are stored in ``_fitted_observations_numpy`` and
        can subsequently be passed to a ``GenerationDiscreteObs`` instance after
        a mode switch from ``'inference'`` to ``'generation'``.

        Raises
        ------
        ValueError
            If no inference trajectories have been loaded yet.
        """
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
        """
        Compute the mean negative log-likelihood over a set of trajectories.

        Works in both ``'generation'`` mode (NumPy forward pass) and
        ``'inference'`` mode (PyTorch forward pass without gradient tracking).

        Parameters
        ----------
        trajectories : list of dict
            Each dictionary must contain the keys ``'observations'`` and
            ``'actions'`` with array-like values of equal length.

        Returns
        -------
        float
            Mean negative log-likelihood across all trajectories.
        """
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
        """
        Evaluate the negative log-likelihood for a single loaded trajectory.

        Only available in ``'inference'`` mode after trajectories have been
        loaded via ``load_trajectories_tofit``.

        Parameters
        ----------
        trajectory_idx : int
            Index into the list of loaded trajectories.

        Returns
        -------
        float
            Negative log-likelihood of the specified trajectory.

        Raises
        ------
        ValueError
            If the current mode is not ``'inference'``.
        ValueError
            If no trajectories have been loaded.
        """
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
        """
        Reset the ``trained`` flag to ``False``.

        Useful when re-running inference on an already-trained FSC without
        creating a new instance.
        """
        self.__trained = False
    
    def set_trained_status(self, val):
        """
        Manually override the ``trained`` flag.

        Parameters
        ----------
        val : bool
            New value for the ``trained`` flag.
        """
        self.__trained = val

    def initialize_for_inference(self):
        """
        Prepare the FSC for a new inference run.

        Switches the mode to ``'inference'`` and resets the flag that tracks
        whether trajectories have been loaded, so that a fresh set of
        trajectories can be provided.
        """
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
        """
        Configure and validate the inference hyper-parameters before calling ``fit``.

        The optimization pipeline supports three optional stages, which can be
        combined freely:

        1. **MAPSO** — Modified Adaptive Particle Swarm Optimization over the
           full parameter space (global or local-best kNN topology).
        2. **Gradient descent** — Adam or SGD optimizer with a learning-rate
           scheduler, applied after (or instead of) MAPSO.
        3. **Convex–concave optimization (ccopt)** — an alternative analytical
           solver for the initial memory distribution ``rho`` only.

        The validated parameters are stored in ``self.inference_params`` and
        consumed by ``fit``.

        Parameters
        ----------
        use_gradient : bool, optional
            Enable the gradient-descent stage. Default ``True``.
        use_MAPSO : bool, optional
            Enable the MAPSO stage. Default ``True``.
        use_ccopt : bool, optional
            Enable the convex–concave optimization stage. Default ``False``.
        trainable_parameters : str or list of str, optional
            Which parameters to optimize. Use ``'all'`` to train everything,
            ``'elementwise_mask'`` to supply a Boolean mask per parameter, or a
            list of parameter name strings (e.g. ``['theta', 'psi']``).
            Default ``'all'``.
        trainable_mask : dict or None, optional
            Required when ``trainable_parameters='elementwise_mask'``. Keys
            are parameter names; values are Boolean NumPy arrays (or tensors)
            of the same shape as the corresponding parameter.
        NEpochs_gradient : int or None
            Number of gradient-descent epochs. Required when
            ``use_gradient=True``.
        NBatch_gradient : int or None
            Mini-batch size for gradient descent (number of trajectories per
            gradient step). Required when ``use_gradient=True``.
        lr_gradient : float or None
            Initial learning rate. Required when ``use_gradient=True``.
        optimizer_gradient : str, optional
            Optimizer name. Supported values: ``'ADAM'``, ``'SGD'``.
            Default ``'ADAM'``.
        scheduler_gradient : dict, optional
            Learning-rate scheduler specification. Must contain the key
            ``'type'`` (``'exponential'`` or ``'fixed'``). For
            ``'exponential'``, also provide ``'decay_rate'`` (float).
            Default ``{"type": "exponential", "decay_rate": 0.8}``.
        train_split_gradient : float, optional
            Fraction of trajectories used for training; the remainder forms a
            validation set. Must be in ``(0, 1]``. Default ``1`` (no split).
        n_particles_MAPSO : int or None
            Number of PSO particles. Required when ``use_MAPSO=True``.
        NEpochs_MAPSO : int or None
            Number of PSO iterations. Required when ``use_MAPSO=True``.
        init_particles_MAPSO : dict, optional
            Initial particle distribution specification. Must contain
            ``'distribution'`` (``'uniform'``, ``'normal'``,
            ``'multivariate_normal'``, or ``'uniform_with_biases'``) and the
            corresponding parameters (e.g. ``'xmin'``/``'xmax'`` for
            ``'uniform'``). Default uniform in ``[-5, 5]``.
        init_velocities_MAPSO : dict, optional
            Initial velocity distribution specification. Supports
            ``'uniform'`` (with ``'vmin'``/``'vmax'``) and ``'normal'``
            (with ``'mean'``/``'std'``). Default zero velocities.
        c1_init_MAPSO : float, optional
            Initial cognitive (personal-best) acceleration coefficient.
            Default ``2.0``.
        c2_init_MAPSO : float, optional
            Initial social (global-best) acceleration coefficient.
            Default ``2.0``.
        w_init_MAPSO : float, optional
            Initial inertia weight. Default ``0.9``.
        sigma_min_MAPSO : float, optional
            Minimum mutation standard deviation used in the convergence
            strategy. Default ``0.1``.
        sigma_max_MAPSO : float, optional
            Maximum mutation standard deviation. Default ``1``.
        dynamic_topology_MAPSO : bool, optional
            If ``True``, use a local-best kNN topology with a time-varying
            neighbourhood size instead of the global-best topology.
            Default ``False``.
        num_neighbors_init_MAPSO : int or None
            Initial neighbourhood size. Required when
            ``dynamic_topology_MAPSO=True``.
        num_neighbors_final_MAPSO : int or None
            Final neighbourhood size. Required when
            ``dynamic_topology_MAPSO=True``.
        num_neighbors_mid_MAPSO : int or None
            Neighbourhood size at the midpoint of the run. Required when
            ``dynamic_topology_MAPSO=True``.
        maxiter_ccopt : int, optional
            Maximum number of ccopt iterations. Default ``1000``.
        rho0_ccopt : np.ndarray or None, optional
            Starting point for ccopt. Randomly initialized when ``None``.
        th_ccopt : float, optional
            Convergence threshold for ccopt. Default ``1e-6``.
        c_gauge_ccopt : float, optional
            Gauge constant added to the ccopt objective. Default ``0``.
        print_params : bool, optional
            If ``True``, pretty-print the validated parameter dictionary after
            validation. Default ``True``.

        Raises
        ------
        ValueError
            If both ``use_gradient`` and ``use_MAPSO`` are ``False``.
        ValueError
            If ``trainable_parameters`` contains unknown parameter names.
        AssertionError
            If required parameters for the enabled optimization stages are
            not provided.
        """

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
        """
        Run the full inference pipeline to fit FSC parameters to observed trajectories.

        Calls ``initialize_for_inference``, loads the trajectories, and then
        runs the optimization stages configured by ``set_inference_params``
        (MAPSO and/or gradient descent). When ``overwrite=True`` the optimized
        parameters are written back into the FSC and the ``trained`` flag is set.

        Parameters
        ----------
        trajectories : list of dict
            Training trajectories. Each dictionary must contain the keys
            ``'observations'`` and ``'actions'`` with equal-length sequences.
        overwrite : bool, optional
            If ``True`` (default), copy the best inferred parameters back to
            the FSC's policy model and ``psi`` after training.
        verbose_MAPSO : bool, optional
            If ``True``, print per-particle MAPSO diagnostics. Default ``True``.
        verbose_epochs_MAPSO : bool, optional
            If ``True``, print the best loss at each MAPSO iteration.
            Default ``True``.

        Returns
        -------
        dict
            Dictionary with keys ``'train'`` (and optionally ``'val'``) mapping
            to lists of per-epoch negative log-likelihood values.

        Raises
        ------
        AssertionError
            If the FSC has already been trained (call ``reset_trained_status``
            first to re-train).
        ValueError
            If ``set_inference_params`` has not been called yet.
        """
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
        """
        Load externally provided policy parameters and ``psi`` into the FSC.

        Arrays are automatically converted to the type required by the current
        mode (NumPy for ``'generation'``, PyTorch tensors for ``'inference'``).

        Parameters
        ----------
        policy_params : dict
            Dictionary mapping parameter names (e.g. ``'theta'``, ``'zeta'``)
            to NumPy arrays or PyTorch tensors of the expected shapes.
        psi : np.ndarray or torch.Tensor
            Logit vector for the initial memory distribution. Shape must be
            ``(M,)`` or ``(Y, M)`` depending on
            ``init_memory_obs_dependent``.
        """

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
        """
        Load an observation sequence (or list of sequences) for trajectory
        generation.

        Delegates to ``GenerationDiscreteObs.load_observations``. Only
        available in ``'generation'`` mode.

        Parameters
        ----------
        observations : np.ndarray or list of np.ndarray
            A single 1-D observation array or a list of such arrays. Every
            observation value must belong to ``ObsSpace``.

        Raises
        ------
        ValueError
            If the current mode is not ``'generation'``.
        """
        if self.mode != 'generation':
            raise ValueError("Mode must be 'generation' to load observations.")

        self.generator.load_observations(observations)

    def load_trajectories_tofit(self, trajectories):
        """
        Validate and load a set of observed trajectories for inference.

        Each trajectory is checked for the required ``'observations'`` and
        ``'actions'`` keys and for equal sequence lengths, then forwarded to
        the internal ``InferenceDiscreteObs`` object.

        Parameters
        ----------
        trajectories : list of dict
            List of trajectory dictionaries, each with keys ``'observations'``
            and ``'actions'`` containing equal-length array-like sequences.

        Raises
        ------
        ValueError
            If the current mode is not ``'inference'``.
        AssertionError
            If any trajectory is missing required keys or has mismatched
            sequence lengths.
        """

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
        """
        Print all supported policy models, optimizers, schedulers, and particle
        / velocity distributions.
        """
        print("=============================")
        print("Supported policy models: ", self.supported_policy_models)
        print("Supported optimizers for gradient descent: ", self.supported_optimizers)
        print("Supported schedulers for gradient descent: ", self.supported_schedulers)
        print("Supported particle distributions for MAPSO: ", self.supported_particle_distributions)
        print("Supported velocity distributions for MAPSO: ", self.supported_velocity_distributions)
        print("=============================")

    def show_summary(self):
        """
        Print a human-readable summary of the FSC structure and current
        parameter values.

        Displays the number of memory states, actions, and observations;
        the policy model name; the current mode; the trained status; and the
        values of ``psi``, ``rho``, and all policy parameters.
        """
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
        """
        Generate a single trajectory in ``'generation'`` mode.

        Observations can be supplied in three ways:

        * **Environment model**: pass a ``BaseEnvironmentModel`` instance.
          The FSC interacts with the environment step-by-step, so ``NSteps``
          must be provided.
        * **Explicit observations**: pass an array (or list) of observations
          directly via ``observations``.
        * **Pre-loaded observations**: call ``load_observations_for_generation``
          first and select a sequence by index via ``idx_observation``.

        Parameters
        ----------
        environment_model : BaseEnvironmentModel or None, optional
            If provided, observations are generated on-the-fly from the
            environment model. Cannot be combined with ``observations`` or
            ``idx_observation``.
        NSteps : int or None, optional
            Number of steps. Required when using ``environment_model``;
            otherwise inferred from the observation array length.
        observations : array-like or None, optional
            Explicit observation sequence. Cannot be combined with
            ``environment_model``.
        idx_observation : int or None, optional
            Index of a pre-loaded observation sequence. Cannot be combined
            with ``environment_model``.
        initial_state : dict or None, optional
            Initial state dictionary passed to the environment model when
            ``environment_model`` is provided.
        verbose : bool, optional
            Not used directly here; forwarded to the generator. Default
            ``False``.

        Returns
        -------
        dict
            Dictionary with keys ``'actions'``, ``'memories'``, and
            ``'observations'``, each containing a NumPy array of length
            ``NSteps``.

        Raises
        ------
        ValueError
            If the current mode is not ``'generation'``.
        ValueError
            If incompatible argument combinations are provided.
        """
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
        """
        Generate multiple trajectories in ``'generation'`` mode.

        The observation source follows the same three-way logic as
        ``generate_single_trajectory``. When using pre-loaded observations
        without specifying ``idx_observation``, one trajectory is produced per
        loaded sequence. When specifying ``idx_observation``, ``NTraj``
        independent trajectories are generated from the same sequence.

        Parameters
        ----------
        environment_model : BaseEnvironmentModel or None, optional
            Environment model for on-the-fly observation generation. Requires
            ``NSteps`` and ``NTraj``.
        obs_from_act : bool, optional
            If ``True``, observations are generated as a function of the
            actions taken (uses ``generate_trajectories_from_environment_obs_from_act``
            internally). Only relevant when ``environment_model`` is provided.
            Default ``False``.
        NSteps : int or None, optional
            Trajectory length. Required when using ``environment_model``.
        observations : list of np.ndarray or np.ndarray or None, optional
            Single or multiple observation sequences.
        idx_observation : int or None, optional
            Index of a pre-loaded observation sequence to use for all
            trajectories.
        NTraj : int or None, optional
            Number of trajectories to generate. Required when
            ``environment_model`` is provided or when ``idx_observation`` is
            given.
        initial_states : list of dict or None, optional
            Per-trajectory initial states passed to the environment model.
        verbose : bool, optional
            If ``True``, print generation progress. Default ``False``.

        Returns
        -------
        list of dict
            List of trajectory dictionaries, each with keys ``'actions'``,
            ``'memories'``, and ``'observations'``.

        Raises
        ------
        ValueError
            If the current mode is not ``'generation'``.
        ValueError
            If incompatible argument combinations are provided.
        """
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
        """
        Plot the training (and optional validation) negative log-likelihood
        curves recorded during ``fit``.

        A vertical dashed line marks the best epoch; when both MAPSO and
        gradient descent were used, a second line marks the MAPSO–gradient
        boundary.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes object to draw on. A new figure is created when ``None``.
        figsize : tuple, optional
            Figure size when a new figure is created. Default ``(5, 3)``.
        return_ax : bool, optional
            If ``True``, return ``(fig, ax)``. Default ``False``.

        Returns
        -------
        matplotlib.axes.Axes or (matplotlib.figure.Figure, matplotlib.axes.Axes)
            The axes (and figure) used for the plot, only when
            ``return_ax=True``.

        Raises
        ------
        AssertionError
            If the FSC has not been trained yet.
        """
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
        """
        Plot the FSC as a bipartite network of memory nodes and action nodes.

        In this layout each memory state and each (memory, action) pair appear
        as nodes; edges represent memory transitions weighted by ``g(m'|a,m,y)``
        and action choices weighted by ``pi(a|m)``. Observation-specific
        sub-edges are shown as smaller nodes attached to each memory node.

        Parameters
        ----------
        observation_node_colors : list
            Colors for the observation indicator nodes. Length must equal ``Y``.
        memory_ordering : list of int or None, optional
            Permutation of memory state indices controlling the left-to-right
            layout order. Uses default ordering when ``None``.
        **kwargs
            Additional keyword arguments forwarded to
            ``utils.draw_bipartite_FSC_network``. Any default value (e.g.
            ``spacing``, ``mem_node_size``, ``th_mem_transitions``) can be
            overridden here.

        Raises
        ------
        ValueError
            If the current mode is not ``'generation'``.
        AssertionError
            If the length of ``observation_node_colors`` does not equal ``Y``.
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
        """
        Validate that ``psi`` has the shape expected by the current
        ``init_memory_obs_dependent`` setting.

        Parameters
        ----------
        psi : np.ndarray or None
            Logit vector to validate. Skipped when ``None``.

        Raises
        ------
        ValueError
            If ``psi`` has an incorrect shape.
        """
        if self._init_memory_obs_dependent:
            if psi is not None:
                if psi.shape != (self.Y, self.M):
                    raise ValueError("psi must have shape (Y, M) where M is the number of states and Y is the number of observations.")
        else:
            if psi is not None:
                if psi.shape != (self.M,):
                    raise ValueError("psi must have shape (M,) where M is the number of states.")
        
    def __check_and_init_observations(self, Y):
        """
        Validate that the number of observations ``Y`` is a positive integer.

        Parameters
        ----------
        Y : int
            Number of discrete observations.

        Raises
        ------
        ValueError
            If ``Y`` is ``None``, not positive, or not an integer.
        """
        
        if Y is None:
            raise ValueError("Please provide the number of possible observations Y.")
        if Y <= 0:
            raise ValueError("Y must be a positive integer.")
        if Y != int(Y):
            raise ValueError("Y must be an integer.")            

    def __init_generalized_policy_model(self, policy_model,
                                        policy_params, seed):
        """
        Instantiate the generalized policy model (``self.GPModel``).

        Parameters
        ----------
        policy_model : str
            Name of the parametrization (e.g. ``'softmax'``).
        policy_params : dict
            Dictionary of initial parameter values.
        seed : int or None
            Random seed forwarded to the policy model constructor.

        Raises
        ------
        ValueError
            If ``policy_model`` is not in ``supported_policy_models``.
        """
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
        """
        Populate the private lists of supported policy models, optimizers,
        schedulers, and particle / velocity distributions.
        """
        self.__supported_policy_models_discrete = ["softmax"]

        self.__supported_optimizers = ["ADAM", "SGD"]
        self.__supported_schedulers = ["exponential", "fixed"]

        self.__supported_particle_distributions = ["uniform", "normal", "multivariate_normal", "uniform_with_biases"]
        self.__supported_velocity_distributions = ["uniform", "normal"]


    def __initialize_structure(self, M, A, Y):
        """
        Store the core FSC dimensions as instance attributes.

        Parameters
        ----------
        M : int
            Number of memory states.
        A : int
            Number of actions.
        Y : int
            Number of observations.
        """
        self.M = M
        self.A = A
        self.Y = Y

    def __initialize_psi(self, seed):
        """
        Randomly initialize the logit vector ``psi`` for the initial memory
        distribution.

        Shape is ``(M,)`` when ``init_memory_obs_dependent=False``, or
        ``(Y, M)`` when ``init_memory_obs_dependent=True``.

        Parameters
        ----------
        seed : int or None
            Random seed. Not applied when ``None``.
        """
        if seed is not None:
            np.random.seed(seed)
        if self._init_memory_obs_dependent:
            self.psi = np.random.randn(self.Y, self.M)
        else:
            self.psi = np.random.randn(self.M)

    def __check_ready_for_inference(self):
        """
        Assert that the FSC is properly configured and ready to run inference.

        Raises
        ------
        ValueError
            If the mode is not ``'inference'``, no trajectories have been
            loaded, or the ``InferenceDiscreteObs`` object is missing.
        """
        if self.mode != 'inference':
            raise ValueError("Mode must be 'inference' to perform inference.")
        if not self.__loaded_trajectories_inference:
            raise ValueError("No trajectories to fit have been loaded.")
        if not hasattr(self, 'inferencer'):
            raise ValueError("Inferencer has not been initialized.")
        
    def __initialize_spaces(self, ObsSpace, ActSpace, MemSpace):
        """
        Initialize the observation, action, and memory label arrays.

        When a custom space is provided, its length is validated against the
        corresponding dimension (``Y``, ``A``, or ``M``). Otherwise the
        internal integer index range is used.

        Parameters
        ----------
        ObsSpace : array-like or None
            Custom observation labels (length ``Y``).
        ActSpace : array-like or None
            Custom action labels (length ``A``).
        MemSpace : array-like or None
            Custom memory labels (length ``M``).

        Raises
        ------
        ValueError
            If any provided space has a length that does not match the
            corresponding dimension.
        """
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
        """List of supported policy parametrization names."""
        return self.__supported_policy_models_discrete
        
    @property
    def supported_optimizers(self):
        """List of supported gradient-descent optimizer names."""
        return self.__supported_optimizers
    
    @property
    def supported_schedulers(self):
        """List of supported learning-rate scheduler names."""
        return self.__supported_schedulers
    
    @property
    def supported_particle_distributions(self):
        """List of supported initial particle distribution names for MAPSO."""
        return self.__supported_particle_distributions
    
    @property
    def supported_velocity_distributions(self):
        """List of supported initial velocity distribution names for MAPSO."""
        return self.__supported_velocity_distributions

    @property
    def obs_type(self):
        """Observation type. Always ``'discrete'`` for this class."""
        return "discrete"
    
    @property
    def rho(self):
        """
        Initial memory distribution derived from ``psi`` via softmax.

        Shape is ``(M,)`` when ``init_memory_obs_dependent=False``, or
        ``(Y, M)`` when ``init_memory_obs_dependent=True``. Returns a
        ``torch.Tensor`` when ``psi`` is a tensor, otherwise a NumPy array.
        """
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
        """
        The list of loaded inference trajectories as ``(observations, actions)``
        tensor pairs.

        Raises
        ------
        ValueError
            If no trajectories have been loaded yet.
        """
        if not self.__loaded_trajectories_inference:
            raise ValueError("No trajectories to fit have been loaded.")

        return self.inferencer.ObsAct_trajectories
        
    @property
    def policy_params(self):
        """
        Current policy parameters as a dictionary mapping parameter names to
        their values (NumPy arrays in generation mode, tensors in inference
        mode).
        """
        param_dict = {}

        for key in self.GPModel.param_names:
            param_dict[key] = getattr(self.GPModel, key)

        return param_dict
        
    @property
    def policy_type(self):
        """Human-readable name of the active generalized policy model."""
        return self.GPModel.name
    
    @property
    def trained(self):
        """``True`` if the FSC has been successfully trained via ``fit``."""
        return self.__trained

    ##############################################################################################################################