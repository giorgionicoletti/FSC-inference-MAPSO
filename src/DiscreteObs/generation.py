import numpy as np
import numba as nb
import utils

import matplotlib.pyplot as plt

class GenerationDiscreteObs:

    def __init__(self, FSC):
        """
        Initialize the generation backend for a discrete-observation FSC.

        Parameters:
        --- FSC: FSC
            Parent FSC instance providing policy parameters, spaces, and
            initialization settings.
        """
        self.FSC = FSC

        self.InternalMemSpace = np.arange(self.FSC.M)
        self.InternalActSpace = np.arange(self.FSC.A)
        self.InternalObsSpace = np.arange(self.FSC.Y)
        self.InternalActMemSpace = utils.combine_spaces(self.InternalActSpace, self.InternalMemSpace)
    
    @property
    def TMat(self):
        """Return the current joint transition matrix as a numpy array."""
        return self.FSC.GPModel.get_TMat_numpy()
    
    def get_TMat(self):
        """
        Return the current joint transition matrix.

        Returns:
        --- np.ndarray
            Transition tensor with shape ``(Y, A, M, M)``.
        """
        return self.FSC.GPModel.get_TMat_numpy()
    
    def get_memory_transition(self):
        """
        Return the memory transition component of the policy.

        Returns:
        --- np.ndarray
            Memory transition tensor ``g(m' | a, m, y)`` with shape
            ``(Y, A, M, M)``.
        """
        return self.FSC.GPModel.get_memory_transition_numpy()
    
    def get_action_policy(self):
        """
        Return the marginal action policy.

        Returns:
        --- np.ndarray
            Action policy matrix ``pi(a | m)`` with shape ``(M, A)``.
        """
        return self.FSC.GPModel.get_action_policy_numpy()

    def load_observations(self, observations):
        """
        Loads a sequence of observations to be used to generate a trajectory.

        Parameters:
        --- observations: list of np.arrays
            List of observation sequences. Each element of the list is an np.array of possibly different lengths,
            containing the observations for a single trajectory.
        """

        self.observations = []

        if type(observations[0]) is np.ndarray or type(observations[0]) is list:
            for obs_seq in observations:
                for obs in obs_seq:
                    assert obs in self.FSC.ObsSpace, "All observations must be in the observation space."
                self.observations.append(self.__map_obs_to_internal_space(np.array(obs_seq)))
        else:
            for obs in observations:
                assert obs in self.FSC.ObsSpace, "All observations must be in the observation space."

            self.observations.append(self.__map_obs_to_internal_space(np.array(observations)))

        self.observations_lengths = np.array([len(obs) for obs in observations])
        self.min_obs_length = np.min(self.observations_lengths)
        self.max_obs_length = np.max(self.observations_lengths)


    def generate_single_trajectory(self, NSteps, observations = None, idx_observation = None):
        """
        Generates a single trajectory of NSteps length given a sequence of observations. If no observations are provided,
        the method uses the loaded observations and generates a trajectory for the indexed observation sequence.

        The number of steps NSteps must be smaller or equal than the number of observations in the provided observation
        sequence. Note that the trajectory is not stored, but returned as a dictionary.

        Parameters:
        --- NSteps: int
            Number of steps for the trajectory.
        --- observations: np.array (default = None)
            Array of observations. If None, the method uses the loaded observations.
        --- idx_observation: int (default = None)
            Index of the observation sequence to use if no observations are provided.

        Returns:
        --- trajectory: dict
            Dictionary containing the actions, memories, and observations for the generated trajectory.
        """
        if observations is None:
            assert hasattr(self, "observations"), "No observations have been loaded. Load observations with the load_observations method."
            assert idx_observation is not None, "If no observations are provided, the idx_observation parameter must not be None."
            assert NSteps <= self.observations[idx_observation].size, "NSteps must be smaller or equal than the number of observations."

            observations_cut = self.observations[idx_observation][:NSteps]
        else:
            # check that observation is a single array or a single list of observations
            if type(observations) is list:
                assert len(observations) == 1, "If observations is a list, it must contain a single observation sequence."
                if type(observations[0]) is not np.ndarray:
                    obs_seq = np.array(observations[0])
            elif type(observations) is np.ndarray:
                assert observations.ndim == 1, "If observations is a numpy array, it must be a 1D array."
                obs_seq = observations
            else:
                raise ValueError("Observations must be a numpy array or a list of numpy arrays.")
            if NSteps is None:
                NSteps = len(obs_seq)
            assert NSteps <= observations.size, "NSteps must be smaller or equal than the number of observations."

            for obs in obs_seq:
                assert obs in self.FSC.ObsSpace, "All observations must be in the observation space."

            observations_cut = self.__map_obs_to_internal_space(obs_seq[:NSteps])

        if self.FSC._init_memory_obs_dependent:
            rho = self.FSC.rho[observations_cut[0]]
        else:
            rho = self.FSC.rho

        int_actions, int_memories = GenerationDiscreteObs._nb_generate_trajectory(NSteps, self.InternalMemSpace, self.InternalActMemSpace,
                                                                                  self.TMat, rho, observations_cut)
        actions = np.array([self.FSC.ActSpace[act] for act in int_actions])
        memories = np.array([self.FSC.MemSpace[mem] for mem in int_memories])
        obs = np.array([self.FSC.ObsSpace[obs] for obs in observations_cut])
        
        trajectory = {"actions": actions, "memories": memories, "observations": obs}

        return trajectory
    
    def generate_trajectories(self, NSteps, observations = None, idx_observation = None, NTraj = None,
                                                verbose = False):
        """
        Generates NTraj trajectories of NSteps length given a sequence of observations. If no observations are provided,
        the method uses the loaded observations. It is also possible to generate NTraj trajectories for the same observation
        sequence by providing the idx_observation parameter and setting the NTraj parameter.

        In any case, the number of steps NSteps must be smaller or equal than the number of observations in the provided
        observation sequence.

        Note that the trajectories are not stored in the object, but returned as a list of dictionaries.

        Parameters:
        --- NSteps: int
            Number of steps for the trajectory.
        --- observations: list of np.arrays (default = None)
            List of observation sequences. If None, the method uses the loaded observations.
        --- idx_observation: int (default = None)
            Index of the observation sequence to use if no observations are provided.
        --- NTraj: int (default = None)
            Number of trajectories to generate. If None, the method generates one trajectory per observation sequence.
        --- verbose: bool (default = False)
            If True, prints information about the generation process.

        Returns:
        --- trajectories: list of dicts
            List of dictionaries containing the actions, memories, and observations for each generated trajectory.
        """
        observations_cut = []

        if observations is None:
            if idx_observation is None:
                if verbose:
                    print("No observations provided. Using the loaded observations and generating one trajectory per observation sequence.")
                assert hasattr(self, "observations"), "No observations have been loaded. Load observations with the load_observations method."
                assert NSteps <= self.min_obs_length, "NSteps must be smaller than the shortest observation length."
                NTraj = len(self.observations)
                
                for n in range(NTraj):
                    observations_cut.append(self.observations[n][:NSteps])

            else:
                if verbose:
                    print("No observations provided. Using the indexed observation sequence and generating NTraj trajectories for the same observation sequence.")
                assert hasattr(self, "observations"), "No observations have been loaded. Load observations with the load_observations method."
                assert NSteps <= self.observations[idx_observation].size, "NSteps must be smaller or equal than the number of observations."
                assert NTraj is not None, "If no observations are provided, the NTraj parameter must be provided."

                for n in range(NTraj):
                    observations_cut.append(self.observations[idx_observation][:NSteps])

        elif type(observations[0]) is np.ndarray:
            if verbose:
                print("Multiple observation sequences provided. Generating one trajectory per observation sequence.")
            obs_lengths = np.array([len(obs) for obs in observations])
            if NSteps is None:
                NSteps = np.min(obs_lengths)
            assert np.all(obs_lengths >= NSteps), "All observation sequences must have at least NSteps observations."

            for n in range(len(observations)):
                for obs in observations[n]:
                    assert obs in self.FSC.ObsSpace, "All observations must be in the observation space."
                observations_cut.append(self.__map_obs_to_internal_space(observations[n][:NSteps]))
            NTraj = len(observations)

        else:
            if verbose:
                print("Single observation sequence provided. Generating NTraj trajectories for the same observation sequence.")
            if NSteps is None:
                NSteps = len(observations)
            assert NSteps <= observations.size, "NSteps must be smaller or equal than the number of observations."
            assert NTraj is not None, "If observations is a single array, the NTraj parameter must be provided."

            for n in range(NTraj):
                for obs in observations:
                    assert obs in self.FSC.ObsSpace, "All observations must be in the observation space."
                observations_cut.append(self.__map_obs_to_internal_space(observations[:NSteps]))

        if self.FSC._init_memory_obs_dependent:
            rho = utils.softmax(self.FSC.psi, axis = 1)

        trajectories_rho = []
        for idx_trj in range(len(observations_cut)):
            if self.FSC._init_memory_obs_dependent:
                trajectories_rho.append(rho[observations_cut[idx_trj][0]])
            else:
                trajectories_rho.append(self.FSC.rho)

        int_actions, int_memories = GenerationDiscreteObs._nb_generate_trajectories_parallel(NTraj, NSteps, self.InternalMemSpace, self.InternalActMemSpace,
                                                                                             self.TMat, trajectories_rho, observations_cut)
        trajectories = []

        for n in range(NTraj):
            actions = np.array([self.FSC.ActSpace[act] for act in int_actions[n]])
            memories = np.array([self.FSC.MemSpace[mem] for mem in int_memories[n]])
            obs = np.array([self.FSC.ObsSpace[obs] for obs in observations_cut[n]])
            trajectory = {"actions": actions, "memories": memories, "observations": obs}
            trajectories.append(trajectory)
        
        return trajectories

    
    def evaluate_nloglikelihood(self, trajectory):
        """
        Evaluates the negative log-likelihood of a given trajectory.

        Parameters:
        --- trajectory: dict
            Dictionary containing the actions, memories, and observations for the trajectory.

        Returns:
        --- nLL: float
            Negative log-likelihood of the trajectory.
        """

        actions = trajectory["actions"]
        observations = trajectory["observations"]

        actions = self.__map_act_to_internal_space(actions)
        observations = self.__map_obs_to_internal_space(observations)

        if self.FSC._init_memory_obs_dependent:
            rho = self.FSC.rho[observations[0]]
        else:
            rho = self.FSC.rho

        nLL = GenerationDiscreteObs._nb_evaluate_nloglikelihood(observations, actions, self.TMat, rho)

        return nLL
    
    def __map_obs_to_internal_space(self, obs):
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
    
    def __map_act_to_internal_space(self, act):
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

    def plot_trajectory(self, trj, Time = None):
        """
        Plots the actions, memories, and observations of a given trajectory.
        """

        if Time is None:
            Time = np.arange(len(trj["observations"]))

        fig, ax = plt.subplots(3,1, figsize=(10,5))
        plt.subplots_adjust(hspace=0.5)

        ax[0].plot(np.arange(len(trj["observations"])), trj["observations"], 'o', c= 'k')
        ax[0].plot(np.arange(len(trj["observations"])), trj["observations"], c = 'k')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Observations')

        ax[1].plot(np.arange(len(trj["memories"])), trj["memories"], 'o', c= 'k')
        ax[1].plot(np.arange(len(trj["memories"])), trj["memories"], c = 'k')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Memories')

        ax[2].plot(np.arange(len(trj["actions"])), trj["actions"], 'o', c= 'k')
        ax[2].plot(np.arange(len(trj["actions"])), trj["actions"], c = 'k')
        ax[2].set_xlabel('Time')
        ax[2].set_ylabel('Actions')

        return fig, ax    

    @staticmethod
    @nb.njit
    def _nb_evaluate_nloglikelihood(observations, actions, TMat, rho):
            """
            Static method providing a numba-compiled implementation of the negative log-likelihood evaluation
            for a given trajectory. The method is then wrapped in the evaluate_nloglikelihood method.

            Parameters:
            --- observations: np.array
                Array of observations.
            --- actions: np.array
                Array of actions.
            --- TMat: np.array of shape (Y, M, M, A)
                Transition probability matrix.
            --- rho: np.array of shape (M)
                Initial memory occupation.
            --- ActSpace: np.array
                Array of actions.
            --- MemSpace: np.array
                Array of memory states.

            Returns:
            --- nLL: float
                Negative log-likelihood of the trajectory.
            """
            nLL = 0.
    
            for t, obs in enumerate(observations):
                a = actions[t]
    
                transition_probs = TMat[obs, a].T
    
                if t == 0:
                    m = transition_probs @ rho
                else:
                    m = transition_probs @ m
    
                mv = np.sum(m)
                nLL = nLL - np.log(mv)
                m /= mv
    
            return nLL - np.log(np.sum(m))

    @staticmethod
    @nb.njit
    def _nb_generate_trajectory(NSteps, MSpace, AMSpace, TMat, rho, observations):
        """
        Static method providing a numba-compiled implementation of the trajectory generation
        for a given observation sequence. The method is then wrapped in the generate_single_trajectory method.

        Parameters:
        --- NSteps: int
            Number of steps for the trajectory.
        --- MSpace: np.array
            Array of memory states.
        --- AMSpace: np.array
            Array of memory-action pairs.
        --- TMat: np.array of shape (Y, M, M, A)
            Transition probability matrix.
        --- rho: np.array of shape (M)
            Initial memory occupation.
        --- observations: np.array
            Array of observations.

        Returns:
        --- actions: np.array
            Array of actions.
        --- memories: np.array
            Array of memory states.
        """
        actions = np.zeros(NSteps, dtype = np.int32)
        memories = np.zeros(NSteps, dtype = np.int32)

        initial_memory = utils.numba_random_choice(MSpace, rho)

        memories[0] = initial_memory

        for t in range(0, NSteps):
            transition_probs = TMat[observations[t], :, memories[t]].flatten()
            new_AM = utils.numba_random_choice(AMSpace, transition_probs)
            if t < NSteps - 1:
                memories[t + 1] = new_AM[1]
            actions[t] = new_AM[0]

        return actions, memories
    
    @staticmethod
    @nb.njit(parallel = True)
    def _nb_generate_trajectories_parallel(NTraj, NSteps, MSpace, AMSpace, TMat, trajectories_rho, observations):
            """
            Static method providing a numba-compiled implementation of the trajectory generation
            for a given observation sequence. The method is then wrapped in the generate_trajectories method,
            and it generates NTraj trajectories in parallel.

            Parameters:
            --- NTraj: int
                Number of trajectories to generate.
            --- NSteps: int
                Number of steps for the trajectory.
            --- MSpace: np.array
                Array of memory states.
            --- AMSpace: np.array
                Array of memory-action pairs.
            --- TMat: np.array of shape (Y, M, M, A)
                Transition probability matrix.
            --- rho: np.array of shape (M)
                Initial memory occupation.
            --- observations: np.array
                Array of observations.

            Returns:
            --- actions: np.array
                Array of actions.
            --- memories: np.array
                Array of memory states.
            """
            actions = np.zeros((NTraj, NSteps), dtype = np.int32)
            memories = np.zeros((NTraj, NSteps), dtype = np.int32)
    
            for n in nb.prange(NTraj):
                initial_memory = utils.numba_random_choice(MSpace, trajectories_rho[n])
    
                memories[n, 0] = initial_memory
    
                for t in range(0, NSteps):
                    transition_probs = TMat[observations[n][t], :, memories[n][t]].flatten()
                    new_AM = utils.numba_random_choice(AMSpace, transition_probs)
                    if t < NSteps - 1:
                        memories[n, t+1] = new_AM[1]
                    actions[n, t] = new_AM[0]
    
            return actions, memories
    
    def generate_trajectories_from_environment_obs_from_act(self, EModel,
                                                            NSteps, NTraj, initial_states=None):
        """
        Generate trajectories where observations are produced by an
        environment model conditioned on the sampled action sequence.

        This routine simulates FSC memory/action evolution jointly with
        environment state transitions. It is only compatible with
        observation-independent initial memory distributions.

        Parameters:
        --- EModel: BaseEnvironmentModel
            Environment model implementing numba-compatible transition and
            observation functions.
        --- NSteps: int
            Number of trajectory time steps.
        --- NTraj: int
            Number of trajectories to generate.
        --- initial_states: list or np.ndarray or None (default = None)
            Initial state per trajectory. If ``None``, states are sampled via
            ``EModel.generate_initial_state``.

        Returns:
        --- trajectories: list of dict
            Generated trajectories with keys ``actions``, ``memories``,
            ``observations``, and ``states``.
        """
        if self.FSC._init_memory_obs_dependent:
            raise ValueError("This method is not compatible with observation-dependent initial memory.")
        
        if initial_states is None:
            initial_states = np.zeros((NTraj,) + EModel.state_shape, dtype = np.int32)
            for n in range(NTraj):
                gen_state = EModel.generate_initial_state(overwrite=False)
                initial_states[n] = np.array(EModel.create_state_tuple(gen_state))
        else:
            if isinstance(initial_states, list):
                if len(initial_states) != NTraj:
                    raise ValueError("Initial states must be a list of length NTraj.")
                for n in range(NTraj):
                    if isinstance(initial_states[n], dict):
                        initial_states[n] = np.array(EModel.create_state_tuple(initial_states[n]))
                    elif isinstance(initial_states[n], tuple):
                        initial_states[n] = np.array(initial_states[n])
                    elif not isinstance(initial_states[n], np.ndarray):
                        raise ValueError("Each initial state must be a dictionary, tuple, or numpy array.")
            elif isinstance(initial_states, np.ndarray):
                if initial_states.shape[0] != NTraj:
                    raise ValueError("Initial states must be a numpy array of shape (NTraj, state_shape).")
            else:
                raise ValueError("Initial states must be a list or numpy array.")
            
            initial_states = np.array(initial_states)

        rho = np.zeros((NTraj, self.FSC.M))

        for n in range(NTraj):
            rho[n] = self.FSC.rho

        fun_gen = GenerationDiscreteObs._nb_generate_trajectory_environment_obs_from_act
        fun_gen_par = GenerationDiscreteObs._nb_generate_trajectories_environment_parallel_obs_from_act

        generated_trj = fun_gen_par(fun_gen,
                                    piMat = self.FSC.get_action_policy(),
                                    gMat = self.FSC.get_memory_transitions(),
                                    nb_generate_state=EModel._nb_generate_state_transition,
                                    nb_generate_observation=EModel._nb_generate_observation,
                                    env_params=EModel.params,
                                    initial_states=initial_states, 
                                    state_shape=EModel.state_shape,
                                    NTraj=NTraj, NSteps=NSteps,
                                    MSpace=self.InternalMemSpace, ASpace=self.InternalActSpace, rho=rho)

        trajectories = []
        for n in range(NTraj):
            actions = np.array([self.FSC.ActSpace[act] for act in generated_trj[0][n]])
            memories = np.array([self.FSC.MemSpace[mem] for mem in generated_trj[1][n]])
            states = []

            for t in range(NSteps):
                states.append(EModel.create_state_dict(generated_trj[2][n][t]))
            observations = np.array([self.FSC.ObsSpace[int(obs)] for obs in generated_trj[3][n]])

            trajectory = {"actions": actions, "memories": memories, "observations": observations, "states": states}
            trajectories.append(trajectory)

        return trajectories
    
    @staticmethod
    @nb.njit(parallel=True)
    def _nb_generate_trajectories_environment_parallel_obs_from_act(fun_single_trj, piMat, gMat,
                                                       nb_generate_state, nb_generate_observation,
                                                       env_params,
                                                       initial_states,
                                                       state_shape,
                                                       NTraj, NSteps, MSpace, ASpace, rho):
        """
        Numba-parallel wrapper for environment-based trajectory generation.

        Parameters:
        --- fun_single_trj: callable
            Numba-compiled function generating one trajectory.
        --- piMat: np.ndarray
            Action policy matrix.
        --- gMat: np.ndarray
            Memory transition tensor.
        --- nb_generate_state: callable
            Numba-compiled environment state transition function.
        --- nb_generate_observation: callable
            Numba-compiled environment observation function.
        --- env_params: tuple or dict-like
            Environment parameters consumed by the numba callbacks.
        --- initial_states: np.ndarray
            Array of initial states, one per trajectory.
        --- state_shape: tuple
            Shape of one environment state.
        --- NTraj: int
            Number of trajectories.
        --- NSteps: int
            Number of time steps.
        --- MSpace: np.ndarray
            Memory index space.
        --- ASpace: np.ndarray
            Action index space.
        --- rho: np.ndarray
            Initial memory distributions, one row per trajectory.

        Returns:
        --- tuple
            ``(actions, memories, states, observations)`` arrays.
        """
        
        actions = np.zeros((NTraj, NSteps), dtype = np.int32)
        memories = np.zeros((NTraj, NSteps), dtype = np.int32)
        observations = np.zeros((NTraj, NSteps), dtype = np.int32)
        states = np.zeros((NTraj, NSteps) + state_shape)

        for n in nb.prange(NTraj):
            res = fun_single_trj(piMat, gMat,
                                nb_generate_state, nb_generate_observation,
                                env_params,
                                initial_states[n],
                                state_shape,
                                NSteps, MSpace, ASpace, rho[n])
            
            actions[n] = res[0]
            memories[n] = res[1]
            states[n] = res[2]
            observations[n] = res[3]

        return actions, memories, states, observations

    @staticmethod
    @nb.njit
    def _nb_generate_trajectory_environment_obs_from_act(piMat, gMat,
                                                         nb_generate_state, nb_generate_observation,
                                                         env_params,
                                                         initial_state,
                                                         state_shape,
                                                         NSteps, MSpace, ASpace, rho):
        """
        Numba kernel generating one trajectory with environment-driven
        observations.

        Parameters:
        --- piMat: np.ndarray
            Action policy matrix.
        --- gMat: np.ndarray
            Memory transition tensor.
        --- nb_generate_state: callable
            Numba-compiled environment transition function.
        --- nb_generate_observation: callable
            Numba-compiled observation function.
        --- env_params: tuple or dict-like
            Environment parameters.
        --- initial_state: np.ndarray
            Initial environment state.
        --- state_shape: tuple
            Shape of one state.
        --- NSteps: int
            Number of trajectory steps.
        --- MSpace: np.ndarray
            Memory index space.
        --- ASpace: np.ndarray
            Action index space.
        --- rho: np.ndarray
            Initial memory distribution.

        Returns:
        --- tuple
            ``(actions, memories, states, observations)`` for one trajectory.
        """
        actions = np.zeros(NSteps, dtype = np.int32)
        memories = np.zeros(NSteps, dtype = np.int32)
        observations = np.zeros(NSteps, dtype = np.int32)
        states = np.zeros((NSteps,) + state_shape, dtype=initial_state.dtype)

        initial_memory = utils.numba_random_choice(MSpace, rho)
        memories[0] = initial_memory
        states[0] = initial_state

        for t in range(0, NSteps):
            actions[t] = utils.numba_random_choice(ASpace, piMat[memories[t]])

            new_state = nb_generate_state(states[t], actions[t], env_params)
            observations[t] = nb_generate_observation(new_state, actions[t], states[t], env_params)

            new_memory = utils.numba_random_choice(MSpace, gMat[observations[t], actions[t], memories[t]])

            if t < NSteps - 1:
                memories[t + 1] = new_memory

                for idx_state in range(len(state_shape)):
                    states[t + 1][idx_state] = new_state[idx_state]

        return actions, memories, states, observations

    def generate_trajectories_from_stateseq_obs_from_act(self, EModel,
                                                         state_seq, NTraj):
        """
        Generate trajectories using a fixed sequence of environment states.

        Unlike ``generate_trajectories_from_environment_obs_from_act``, this
        method does not sample state transitions: it receives a full state
        sequence and only samples actions/memory transitions, while
        observations are generated from each provided state transition.

        Parameters:
        --- EModel: BaseEnvironmentModel
            Environment model providing the observation function.
        --- state_seq: list or np.ndarray
            Sequence of states of length ``NSteps + 1``.
        --- NTraj: int
            Number of trajectories to sample against the same state sequence.

        Returns:
        --- trajectories: list of dict
            Generated trajectories with keys ``actions``, ``memories``,
            ``observations``, and ``states``.
        """
        if self.FSC._init_memory_obs_dependent:
            raise ValueError("This method is not compatible with observation-dependent initial memory.")
        
        NSteps = len(state_seq) - 1
        if isinstance(state_seq, list):
            for t in range(NSteps + 1):
                if isinstance(state_seq[t], dict):
                    state_seq[t] = np.array(EModel.create_state_tuple(state_seq[t]))
                elif isinstance(state_seq[t], tuple):
                    state_seq[t] = np.array(state_seq[t])
                elif not isinstance(state_seq[t], np.ndarray):
                    raise ValueError("Each state must be a dictionary, tuple, or numpy array.")
        elif not isinstance(state_seq, np.ndarray):
            raise ValueError("Initial states must be a list or numpy array.")
        
        rho = np.zeros((NTraj, self.FSC.M))

        for n in range(NTraj):
            rho[n] = self.FSC.rho

        state_seq = np.array(state_seq)

        fun_gen = GenerationDiscreteObs._nb_generate_trajectory_stateseq_obs_from_act
        fun_gen_par = GenerationDiscreteObs._nb_generate_trajectories_stateseq_parallel_obs_from_act

        generated_trj = fun_gen_par(fun_gen,
                                    piMat = self.FSC.get_action_policy(),
                                    gMat = self.FSC.get_memory_transitions(),
                                    state_seq = state_seq,
                                    nb_generate_observation=EModel._nb_generate_observation,
                                    env_params=EModel.params,
                                    state_shape=EModel.state_shape,
                                    NTraj=NTraj, NSteps=NSteps,
                                    MSpace=self.InternalMemSpace, ASpace=self.InternalActSpace, rho=rho)

        trajectories = []
        for n in range(NTraj):
            actions = np.array([self.FSC.ActSpace[act] for act in generated_trj[0][n]])
            memories = np.array([self.FSC.MemSpace[mem] for mem in generated_trj[1][n]])
            states = []

            for t in range(NSteps):
                states.append(EModel.create_state_dict(generated_trj[2][n][t]))
            observations = np.array([self.FSC.ObsSpace[int(obs)] for obs in generated_trj[3][n]])

            trajectory = {"actions": actions, "memories": memories, "observations": observations, "states": states}
            trajectories.append(trajectory)

        return trajectories

    @staticmethod
    @nb.njit(parallel=True)
    def _nb_generate_trajectories_stateseq_parallel_obs_from_act(fun_single_trj, piMat, gMat,
                                                       state_seq, nb_generate_observation,
                                                       env_params,
                                                       state_shape,
                                                       NTraj, NSteps, MSpace, ASpace, rho):
        """
        Numba-parallel wrapper for fixed-state-sequence trajectory generation.

        Parameters:
        --- fun_single_trj: callable
            Numba-compiled function generating one trajectory.
        --- piMat: np.ndarray
            Action policy matrix.
        --- gMat: np.ndarray
            Memory transition tensor.
        --- state_seq: np.ndarray
            Shared state sequence of length ``NSteps + 1``.
        --- nb_generate_observation: callable
            Numba-compiled environment observation function.
        --- env_params: tuple or dict-like
            Environment parameters.
        --- state_shape: tuple
            Shape of one environment state.
        --- NTraj: int
            Number of trajectories.
        --- NSteps: int
            Number of steps.
        --- MSpace: np.ndarray
            Memory index space.
        --- ASpace: np.ndarray
            Action index space.
        --- rho: np.ndarray
            Initial memory distributions, one per trajectory.

        Returns:
        --- tuple
            ``(actions, memories, states, observations)`` arrays.
        """
        
        actions = np.zeros((NTraj, NSteps), dtype = np.int32)
        memories = np.zeros((NTraj, NSteps), dtype = np.int32)
        observations = np.zeros((NTraj, NSteps), dtype = np.int32)
        states = np.zeros((NTraj, NSteps) + state_shape)

        for n in nb.prange(NTraj):
            res = fun_single_trj(piMat, gMat,
                                state_seq, nb_generate_observation,
                                env_params,
                                state_shape,
                                NSteps, MSpace, ASpace, rho[n])
            
            actions[n] = res[0]
            memories[n] = res[1]
            states[n] = res[2]
            observations[n] = res[3]

        return actions, memories, states, observations

    @staticmethod
    @nb.njit
    def _nb_generate_trajectory_stateseq_obs_from_act(piMat, gMat,
                                                    state_seq, nb_generate_observation,
                                                    env_params,
                                                    state_shape,
                                                    NSteps, MSpace, ASpace, rho):
        """
        Numba kernel generating one trajectory over a provided state sequence.

        Parameters:
        --- piMat: np.ndarray
            Action policy matrix.
        --- gMat: np.ndarray
            Memory transition tensor.
        --- state_seq: np.ndarray
            State sequence of length ``NSteps + 1``.
        --- nb_generate_observation: callable
            Numba-compiled observation function.
        --- env_params: tuple or dict-like
            Environment parameters.
        --- state_shape: tuple
            Shape of one environment state.
        --- NSteps: int
            Number of steps.
        --- MSpace: np.ndarray
            Memory index space.
        --- ASpace: np.ndarray
            Action index space.
        --- rho: np.ndarray
            Initial memory distribution.

        Returns:
        --- tuple
            ``(actions, memories, states, observations)`` for one trajectory.
        """
        actions = np.zeros(NSteps, dtype = np.int32)
        memories = np.zeros(NSteps, dtype = np.int32)
        observations = np.zeros(NSteps, dtype = np.int32)
        states = np.zeros((NSteps,) + state_shape, dtype=state_seq[0].dtype)

        initial_memory = utils.numba_random_choice(MSpace, rho)

        memories[0] = initial_memory
        states[0] = state_seq[0]

        for t in range(0, NSteps):
            actions[t] = utils.numba_random_choice(ASpace, piMat[memories[t]])

            new_state = state_seq[t+1]
            observations[t] = nb_generate_observation(new_state, actions[t], states[t], env_params)

            new_memory = utils.numba_random_choice(MSpace, gMat[observations[t], actions[t], memories[t]])

            if t < NSteps - 1:
                memories[t + 1] = new_memory

                for idx_state in range(len(state_shape)):
                    states[t + 1][idx_state] = new_state[idx_state]

        return actions, memories, states, observations