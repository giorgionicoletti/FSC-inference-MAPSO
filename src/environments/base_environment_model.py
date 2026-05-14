import numpy as np


class BaseEnvironmentModel:
    def __init__(self,
                 name,
                 expected_param_names, expected_param_types,
                 expected_actions,
                 expected_state_names, expected_state_types,
                 observation_type):
        """
        Base class for shared functionality among environment models.

        Parameters
        ----------
        name : str
            Name of the environment model.
        expected_param_names : tuple
            Tuple of expected parameter names.
        expected_param_types : tuple
            Tuple of expected parameter types.
        expected_actions : tuple
            Tuple of expected actions.
        expected_state_names : tuple
            Tuple of expected state names.
        expected_state_types : tuple
            Tuple of expected state types.
        observation_type : str
            Type of observation (e.g., "continuous", "discrete").
        """

        self.__name = name
        self.__expected_param_names = expected_param_names
        self.__expected_param_types = expected_param_types

        if expected_actions == ("none", ):
            self.__actdefined = False
        else:
            self.__actdefined = True
            self.__ActSpace = expected_actions

        if expected_state_names == ("none", ) and expected_state_types == (None, ):
            self.__statedefined = False
        else:
            self.__statedefined = True
            self.__expected_state_names = expected_state_names
            self.__expected_state_types = expected_state_types

        self.__observation_type = observation_type

    def __check_environment_model_consistency(self, environment_params):
        """
        Internal method to check the consistency of the environment model parameters.

        Parameters
        ----------
        environment_params : dict
            Dictionary of environment parameters.

        Raises
        ------
        AssertionError
            If the number of parameters does not match the expected number.
        ValueError
            If a required parameter is missing.
        TypeError
            If a parameter has an incorrect type.
        """

        assert isinstance(environment_params, dict), "environment_params must be a dictionary"
        
        assert len(self.__expected_param_names) == len(self.__expected_param_types), \
            f"Expected {len(self.__expected_param_names)} parameter names, but got {len(self.__expected_param_types)}."
        
        for param_name, param_type in zip(self.__expected_param_names, self.__expected_param_types):
            if param_name not in environment_params:
                raise ValueError(f"Parameter '{param_name}' must be specified in environment_params.")
            
            if not isinstance(param_type, (tuple, list)):
                param_type = (param_type,)
            
            valid = False
            for pt in param_type:
                if pt == callable:
                    if not callable(environment_params[param_name]):
                        raise TypeError(f"Parameter '{param_name}' must be a callable function.")
                    else:
                        valid = True
                        break
                elif isinstance(environment_params[param_name], pt):
                    valid = True
                    break
            if valid is False:
                allowed_types = ", ".join([t.__name__ for t in param_type])
                raise TypeError(f"Parameter '{param_name}' must be of type(s): {allowed_types}.")
                
            
    def __initialize_parameters(self, environment_params):
        """
        Internal method to initialize the environment model parameters.

        Parameters
        ----------
        environment_params : dict
            Dictionary of environment parameters.

        Returns
        -------
        None
        """
        
        for key, val in environment_params.items():
            if val is not None:
                self.__setattr__(key, val)
            else:
                raise ValueError(f"Parameter {key} cannot be None.")
            
        self.__param_names = environment_params.keys()
    
    def __initialize_state_attributes(self):
        """
        Internal method to initialize the state attributes of the environment model.
        Returns None if the environment model does not depend on a state.

        Returns
        -------
        None
        """
        if self.__statedefined:
            for state_name in self.__expected_state_names:
                setattr(self, state_name, None)

    def check_state_validity(self, state):
        """
        Check the validity of the state. A valid state must contain all expected state names and types.
        Returns True if the state is valid, False otherwise, and None if the environment model does not
        depend on a state.

        Parameters
        ----------
        state : dict
            Dictionary containing the state of the environment model.

        Returns
        -------
        bool
            True if the state is valid, False otherwise.
        """
        if self.__statedefined:
            if set(state.keys()) != set(self.__expected_state_names):
                return False
            for state_name, expected_type in zip(self.__expected_state_names, self.__expected_state_types):
                if not isinstance(state[state_name], expected_type):
                    return False
            return True


    def load_state(self, state):
        """
        Load the state of the environment model.

        Parameters
        ----------
        state : dict
            Dictionary containing the state of the environment model.

        Raises
        ------
        ValueError
            If the environment model does not depend on a state.

        Returns
        -------
        None
        """
        if not self.__statedefined:
            raise ValueError(f"Cannot load states in an environment model that does not explicitly depend on a state.")
        
        self.check_state_validity(state)
        
        for state_name in self.__expected_state_names:
            setattr(self, state_name, state[state_name])

    def update_state(self, state):
        """
        Update the state of the environment model.

        Parameters
        ----------
        state : dict
            Dictionary containing the state of the environment model.

        Raises
        ------
        ValueError
            If the environment model does not depend on a state.

        Returns
        -------
        None
        """
        if not self.__statedefined:
            raise ValueError(f"Cannot update states in an environment model that does not explicitly depend on a state.")

        self.check_state_validity(state)
        
        for state_name in self.__expected_state_names:
            setattr(self, state_name, state[state_name])

    def create_state_dict(self, ordered_vals):
        """
        Create a dictionary representing the state of the environment model.

        Parameters
        ----------
        ordered_vals : list
            List of values representing the state.

        Raises
        ------
        ValueError
            If the environment model does not depend on a state.

        Returns
        -------
        dict
            Dictionary containing the state values.
        """
        if not self.__statedefined:
            raise ValueError(f"Cannot access states in an environment model that does not explicitly depend on a state.")
        
        return {name: val for name, val in zip(self.__expected_state_names, ordered_vals)}
    
    def create_state_tuple(self, state_dict):
        """
        Create a tuple representing the state of the environment model.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing the state values.

        Raises
        ------
        ValueError
            If the environment model does not depend on a state.

        Returns
        -------
        tuple
            Tuple containing the state values.
        """
        if not self.__statedefined:
            raise ValueError(f"Cannot access states in an environment model that does not explicitly depend on a state.")
        
        return tuple([state_dict[name] for name in self.__expected_state_names])


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

        if not self.__actdefined:
            raise ValueError(f"Cannot access actions in an environment model that does not explicitly depend on an action.")

        return np.where(self.ActSpace == act)[0][0]
    
    @property
    def state(self):
        """
        Property to return the current state of the environment model.

        Raises
        ------
        ValueError
            If the environment model does not depend on a state.

        Returns
        -------
        dict
            Dictionary containing the current state values.
        """
        if not self.__statedefined:
            raise ValueError(f"Cannot access states in an environment model that does not explicitly depend on a state.")
        
        s = tuple([self.__getattribute__(key) for key in self.__expected_state_names])
        return self.create_state_dict(s)
            

    @property
    def state_names(self):
        """
        Property to return the expected state names of the environment model.

        Returns
        -------
        tuple
            Tuple of expected state names.
        """
        if not self.__statedefined:
            raise ValueError(f"Cannot access states in an environment model that does not explicitly depend on a state.")
        return self.__expected_state_names
    
    @property
    def params(self):
        """
        Property to return the environment model parameters. Return type must be a tuple.

        Returns
        -------
        tuple
            Tuple of policy parameters.
        """

        return tuple([self.__getattribute__(key) for key in self.__param_names])
            
    @property
    def name(self):
        """
        Property to return the name of the environment model.

        Returns
        -------
        str
            Name of the environment model.
        """
        return self.__name
    
    @property
    def param_names(self):
        """
        Property to return the names of the environment model parameters.

        Returns
        -------
        list
            List of parameter names.
        """
        return self.__param_names
        
    @property
    def ActSpace(self):
        """
        Property to return the expected actions of the environment model.

        Returns
        -------
        tuple
            Tuple of expected actions.
        """
        if not self.__actdefined:
            raise ValueError(f"Cannot access actions in an environment model that does not explicitly depend on a state.")
        
        return self.__ActSpace
    
    @property
    def A(self):
        """
        Property to return the number of expected actions.

        Returns
        -------
        int
            Number of expected actions.
        """
        if not self.__actdefined:
            raise ValueError(f"Cannot access actions in an environment model that does not explicitly depend on a state.")
        
        return len(self.__ActSpace)
    
    @property
    def state_names(self):
        """
        Property to return the expected state names of the environment model.

        Returns
        -------
        tuple
            Tuple of expected state names.
        """
        if not self.__statedefined:
            raise ValueError(f"Cannot access states in an environment model that does not explicitly depend on a state.")
        
        return self.__expected_state_names
    
    @property
    def observation_type(self):
        """
        Property to return the observation type of the environment model.

        Returns
        -------
        str
            Observation type (e.g., "continuous", "discrete").
        """
        return self.__observation_type
    
    @property
    def depends_on_state(self):
        """
        Property to check if the environment model depends on a state.

        Returns
        -------
        bool
            True if the environment model depends on a state, False otherwise.
        """
        return self.__statedefined
    
    @property
    def depends_on_action(self):
        """
        Property to check if the environment model depends on an action.

        Returns
        -------
        bool
            True if the environment model depends on an action, False otherwise.
        """
        return self.__actdefined