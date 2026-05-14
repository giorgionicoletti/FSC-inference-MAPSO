import numpy as np
import numba as nb
from .base_environment_model import BaseEnvironmentModel

"""
This file defines an environmental model that can be used to generate state-observation pairs from a given
state and action. The model can be used with the generator instance of the FSC class to generate trajectories
from the finite state controller. The model should be defined for a specific type of observations (discrete or
continuous) for compatibility. Compatibility with numpy and numba is necessary. Numba compatibility must be 
provided in custom functions.

The EnvironmentModel class should provide the following methods:
    --- generate_initial_state ---
        generates the initial state of the environment. The state is a dictionary with keys defined in
        expected_state_names. 
    ----------------------

    --- generate_initial_observation ---
        generates the initial observation of the environment. The observation is generated using the
        initial state and the environment parameters defined in expected_param_names.
    ----------------------

    --- generate_observation_sequence ---
        this method is optional, but should be provided for the environment models for which observations are
        independent of the actions taken and states trantisions. The observation sequence is a numpy
        array generated using the environment parameters defined in expected_param_names.
    ----------------------

    --- state_shape ---
        returns the shape of the state. The shape is a tuple with the number of dimensions and the size of each
        dimension. The shape is used to define the shape of the state in the FSC.
    ----------------------

    --- obs_type ---
        returns the type of the observation. The type is a string with the value "discrete" or "continuous".
        The type is used to check compatibility with the FSC.
    ----------------------

    --- _nb_generate_state_transition ---
        generates the state transition of the environment. The state transition is generated using the old state,
        action and environment parameters defined in expected_param_names.
        This function should be decorated with @nb.njit and should take arguments old_state, action, env_params.
    -----------------------

    --- _nb_generate_observation ---
        generates the observation of the environment. The observation is generated using the new state, action,
        old state and environment parameters defined in expected_param_names.
        This function should be decorated with @nb.njit and should take arguments new_state, action, old_state,
        env_params.
    -----------------------
"""


class EnvironmentModel(BaseEnvironmentModel):
    
    def __init__(self,
                 environment_params: dict):
        """
        Environment model for two-armed Bernoulli bandits with symmetric winning probabilities. The state
        of the environment defines the winning probabilities of the two arms. If the first arm is better than
        the second arm, the state is 0, otherwise it is 1. In state zero, the agent has a winning probability
        of theta for the first arm and 1 - theta for the second arm. In state one, the winning probabilities
        are swapped. The agent can choose between two arms at each step, and the observations are generated
        based on the chosen arm and the current state.

        The state can switch between the two arms with a probability epsilon, indepdendent of the action taken.

        Parameters
        ----------
        environment_params : dict
            Dictionary with the following keys:
                - theta: float
                    The winning probability of the better arm in a given state.
                - epsilon: float
                    The probability of switching the state to the other arm, independent of the action taken.

        Raises
        ------
        AssertionError
            If the number of environment parameters does not match the expected number.
        ValueError
            If a required environment parameter is missing.
        TypeError
            If an environment parameters has an incorrect type.

        Returns
        -------
        None
            The environment model is initialized with the given parameters.
        """

        name = "Two-armed Bernoulli Bandits with symmetric winning probabilities"

        expected_param_names = ("theta", "epsilon")

        expected_param_types = (float, float)

        expected_actions = ("play arm 1", "play arm 2")

        expected_state_names = ("better arm", )
        expected_state_types = (int, )

        observation_type = "discrete"

        super().__init__(
            name = name,
            expected_param_names = expected_param_names,
            expected_param_types = expected_param_types,
            expected_actions = expected_actions,
            expected_state_names = expected_state_names,
            expected_state_types = expected_state_types,
            observation_type = observation_type,
        )

        self._BaseEnvironmentModel__check_environment_model_consistency(environment_params)
        self._BaseEnvironmentModel__initialize_parameters(environment_params)
        self._BaseEnvironmentModel__initialize_state_attributes()

    def generate_initial_state(self, overwrite = True):
        """
        Generates an initial state for the environment model. The state is a dictionary with keys defined in
        expected_state_names.

        Parameters
        ----------
        overwrite : bool, optional
            If True, the state is loaded into the environment model. If False, the state is not loaded. The
            default is True.

        Returns
        -------
        state : dict
            The initial state of the environment model. The state is a dictionary with keys defined in
            expected_state_names.
        """

        state = np.random.randint(0, 2)

        state = self.create_state_dict((state,))

        if overwrite:
            self.load_state(state)
        
        return state

    def generate_initial_observation(self, initial_state):
        """
        Generates the initial observation of the environment model. The observation is generated
        using the initial state and the environment parameters defined in expected_param_names.

        Parameters
        ----------
        initial_state : dict
            The initial state of the environment model. The state is a dictionary with keys defined in
            expected_state_names.

        Returns
        -------
        observation : np.ndarray
            The initial observation of the environment model. 
        """

        raise ValueError("This environment model does not support generating an initial observation independently on the action.")

        arm = self.create_state_tuple(initial_state)

        return self._nb_generate_observation(None, None, arm, self.params)

    def generate_transition(self, action):
        """
        Helper method to generate the transition of the environment model from a given action and the current state.
        The transition is generated using the _nb_generate_state_transition method. The observation is generated
        using the _nb_generate_observation method. The state is updated with the new state.

        Parameters
        ----------
        action : int or str
            The action taken by the agent. If needed, the action is mapped to the internal space of the
            environment model using the _map_act_to_internal_space method defined in the base class.

        Returns
        -------
        observation : np.ndarray
            The observation generated by the environment model.
        """

        action = self._map_act_to_internal_space(action)
        new_state = self._nb_generate_state_transition(self.state, action, self.params)
        observation = self._nb_generate_observation(new_state, action, self.state, self.params)

        self.load_state(self.create_state_dict(new_state))

        return observation

    def generate_observation(self, new_state, action, old_state):
        """
        Helper method to generate the observation of the environment model from a given initial state, action, and
        new state. The observation is generated using the _nb_generate_observation method. The state is not updated.

        Parameters
        ----------
        new_state : dict
            The new state of the environment model. The state is a dictionary with keys defined in
            expected_state_names.
        action : int
            The action taken by the agent. If needed, the action is mapped to the internal space of the
            environment model using the _map_act_to_internal_space method defined in the base class.
        old_state : dict
            The old state of the environment model. The state is a dictionary with keys defined in
            expected_state_names.

        Returns
        -------
        observation : np.ndarray
            The observation generated by the environment model.
        """
        
        action = self._map_act_to_internal_space(action)
        return self._nb_generate_observation(new_state, action, old_state, self.params)
    
    def generate_observation_sequence(self, NSteps):
        """
        Function to generate an observation sequence. This function can only be implemented for environments whose
        observations are independent of the actions taken and states transitions. The observation sequence is a
        numpy array with shape (NSteps,) for continuous observations, and (Y, NSteps) for discrete observations.

        Parameters
        ----------
        NSteps : int
            The number of steps in the observation sequence.

        Raises
        -------
        TypeError
            If the environment model does not support generating an observation sequence independently on the states
            and actions.
        ValueError
            If NSteps is less than 1 or not valid.

        Returns
        -------
        observation_sequence : np.ndarray
            The observation sequence generated by the environment model. The shape of the observation sequence
            is (NSteps,) for continuous observations, and (Y, NSteps) for discrete observations.
        """

        if NSteps < 1:
            raise ValueError("NSteps must be greater than 0.")
        
        raise TypeError("This environment model does not support generating an observation sequence independently on the states and actions.")
    
    @property
    def state_shape(self):
        """
        Returns the shape of the state. The shape must be a tuple and it is used to define the shape of the state
        in the FSC.

        Returns
        -------
        tuple
            The shape of the state. 

        """

        return (1, )
    
    
    @staticmethod
    @nb.njit
    def _nb_generate_state_transition(old_state, action, env_params):
        """
        Numba-accelerated static method to generate the state transition of the environment model. The state is generated from the previous
        state and the action taken.

        Parameters
        ----------
        old_state : tuple
            The old state of the environment model.
        action : int
            The action taken by the agent. The action is an integer that maps to the internal space of the
            environment model.
        env_params : tuple
            The environment parameters defined in expected_param_names. The parameters must be a tuple.

        Returns
        -------
        new_state : tuple
            The new state of the environment model. 
        """

        theta, epsilon = env_params
        
        old_state = old_state[0]

        if np.random.rand() < epsilon:
            new_state = 1 - old_state
        else:
            new_state = old_state

        return [new_state]
    
    @staticmethod
    @nb.njit
    def _nb_generate_observation(new_state, action, old_state, env_params):
        """
        Numba-accelerated static method to generate the observation of the environment model. The observation is generated from the new state,
        action taken, old state and the environment parameters.

        Parameters
        ----------
        new_state : tuple
            The new state of the environment model.
        action : int
            The action taken by the agent. The action is an integer that maps to the internal space of the
            environment model.
        old_state : tuple
            The old state of the environment model.
        env_params : tuple
            The environment parameters defined in expected_param_names. The parameters must be a tuple.

        Returns
        -------
        observation : float
            The observation generated by the environment model.
        """
        
        theta, epsilon = env_params
        
        new_arm = old_state[0]

        if action == new_arm:
            p_obs_loose = 1 - theta
        else:
            p_obs_loose = theta

        if np.random.rand() < p_obs_loose:
            observation = 0
        else:
            observation = 1

        return observation

        