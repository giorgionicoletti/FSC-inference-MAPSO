import numpy as np
import torch

class BasePolicyModel:
    def __init__(self, 
                 name, obs_type,
                 param_names, expected_params_shape,
                 M, A, F = None, Y = None):
        """
        Base class for shared functionality among policy models.

        Parameters
        ----------
        name : str
            Name of the policy model.
        obs_type : str
            Type of observations, either "discrete" or "continuous".
        param_names : list
            List of parameter names.
        expected_params_shape : list
            List of expected shapes for the parameters.
        M : int
            Number of memories.
        A : int
            Number of actions.
        F : int, optional
            Number of features (for continuous observations).
        Y : int, optional
            Number of observations (for discrete observations).

        Raises
        ------
        ValueError
            If obs_type is not "discrete" or "continuous".
        AssertionError
            If Y is not provided for discrete observations.
        AssertionError
            If F is not provided for continuous observations.

        Returns
        -------
        None
        """

        self.__M = M
        self.__A = A
        if obs_type == "discrete":
            assert Y is not None, "Y must be provided for discrete observations."
            self.__Y = Y
        elif obs_type == "continuous":
            assert F is not None, "F must be provided for continuous observations."
            self.__F = F
        else:
            raise ValueError("obs_type must be either 'discrete' or 'continuous'.")

        self.__param_names = param_names
        self.__expected_params_shape = expected_params_shape
        self.__name = name

    def __check_policy_model_consistency(self, policy_params):
        """
        Internal method to check the consistency of the policy model parameters.

        Parameters
        ----------
        policy_params : dict
            Dictionary of policy parameters.

        Raises
        ------
        ValueError
            If the policy parameters shape is not correct.
        TypeError
            If the policy parameters are not a numpy array or a torch tensor.

        Returns
        -------
        None
        """

        for key in self.__param_names:
            par_flag = False
            if isinstance(policy_params[key], np.ndarray):
                if policy_params[key].shape != self.__expected_params_shape[self.__param_names.index(key)]:
                    par_flag = True
            elif isinstance(policy_params[key], torch.Tensor):
                if policy_params[key].shape != torch.Size(self.__expected_params_shape[self.__param_names.index(key)]):
                    par_flag = True
            elif policy_params[key] is None:
                pass
            else:
                raise TypeError(f"{key} must be either a numpy array or a torch tensor.")
            
            if par_flag:
                raise ValueError(f"{key} must have shape {self.__expected_params_shape[self.__param_names.index(key)]}.")
        
    def __initialize_parameters(self, seed, policy_params):
        """
        Internal method to initialize the policy parameters. The parameters are initialized randomly if the seed is
        provided and are stored as attributes of the class.

        Parameters
        ----------
        seed : int
            Seed for random number generation.
        policy_params : dict
            Dictionary of policy parameters.

        Returns
        -------
        None
        """

        if seed is not None:
            np.random.seed(seed)

        for key, val in policy_params.items():
            if val is not None:
                self.__setattr__(key, val)
            else:
                self.__setattr__(key, np.random.rand(*self.__expected_params_shape[self.__param_names.index(key)]))
                
    def _load_params(self, policy_params):
        """
        Method to load the policy parameters. The parameters are checked for consistency and are initialized
        using the __initialize_parameters method.

        Parameters
        ----------
        policy_params : dict
            Dictionary of policy parameters.

        Returns
        -------
        None
        """

        self.__check_policy_model_consistency(policy_params)
        self.__initialize_parameters(None, policy_params)

    @property
    def M(self):
        """
        Property to return the number of memories.

        Returns
        -------
        int
            Number of memories.
        """

        return self.__M
    
    @property
    def A(self):
        """
        Property to return the number of actions.

        Returns
        -------
        int
            Number of actions.
        """

        return self.__A
    
    @property
    def F(self):
        """
        Property to return the number of features.

        Returns
        -------
        int
            Number of features.
        """

        if hasattr(self, "__F"):
            return self.__F
        else:
            raise AttributeError("F is not defined for this policy model.")
    
    @property
    def Y(self):
        """
        Property to return the number of observations.

        Returns
        -------
        int
            Number of observations.
        """

        if hasattr(self, "__Y"):
            return self.__Y
        else:
            raise AttributeError("Y is not defined for this policy model.")


    @property
    def params(self):
        """
        Property to return the policy parameters. Return type must be a tuple.

        Returns
        -------
        tuple
            Tuple of policy parameters.
        """

        return tuple([self.__getattribute__(key) for key in self.__param_names])
    
    @property
    def param_names(self):
        """
        Property to return the names of the policy parameters.

        Returns
        -------
        list
            List of parameter names.
        """

        return self.__param_names
    
    @property
    def name(self):
        """
        Property to return the name of the policy model.

        Returns
        -------
        str
            Name of the policy model.
        """

        return self.__name
    
    @property
    def expected_params_shape(self):
        """
        Property to return the expected shape of the policy parameters.

        Returns
        -------
        dict
            Dictionary of expected shapes for the parameters.
        """

        param_dict = {}
        for key, val in zip(self.__param_names, self.__expected_params_shape):
            param_dict[key] = val
        return param_dict
    
    @property
    def num_params(self):
        """
        Property to return the number of parameters.

        Returns
        -------
        int
            Number of parameters.
        """

        return len(self.__param_names)
    
    @property
    def dim_params(self):
        """
        Property to return the dimension of the parameters.

        Returns
        -------
        int
            Dimension of the parameters.
        """

        return sum([np.prod(val) for val in self.__expected_params_shape])