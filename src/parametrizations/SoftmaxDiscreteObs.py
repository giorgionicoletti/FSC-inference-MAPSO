import numpy as np
import numba as nb
import torch
import utils
from .base_policy_model import BasePolicyModel


"""
This file defines a generalized policy parametrization that can be used in the FSC class. For consistency,
the generalized policy (y, m) -> (a, m') should be defined as an array or a tensor of shape (Y, A, M, M')
for discrete observations, and (A, M, M') for continuous observations. Compatibility with both numpy and
torch is necessary. Flat-index functions are needed for compatibility with APSO (adaptive particle swarm
optimization). Numba compatibility must be provided in custom functions.

The GeneralizedPolicyModel class should provide the following methods:
    --- get_TMat_numpy ---
        returns the transition matrix TMat as a numpy array of shape (A, M, M) for continuous
        observations, and (Y, A, M, M) for discrete observations.
    ----------------------

    --- get_action_policy_numpy ---
        returns the action policy pi as a numpy array of shape (M, A) for continuous observations,
        and (Y, M, A) for discrete observations.
    -------------------------------

    --- get_memory_transition_numpy ---
        returns the memory transition matrix g as a numpy array of shape (A, M, M) for continuous
        observations, and (Y, A, M, M) for discrete observations.
    -----------------------------------

    --- get_TMat_torch ---
        returns the transition matrix TMat as a torch tensor of shape (A, M, M) for continuous
        observations, and (Y, A, M, M) for discrete observations.
    ----------------------

    --- get_action_policy_torch ---
        returns the action policy pi as a torch tensor of shape (M, A) for continuous observations,
        and (Y, M, A) for discrete observations.
    -------------------------------

    --- get_memory_transition_torch ---
        returns the memory transition matrix g as a torch tensor of shape (A, M, M) for continuous
        observations, and (Y, A, M, M) for discrete observations.
    -----------------------------------

    --- _nb_get_TMat ---
        returns the transition matrix TMat as a numpy array of shape (A, M, M) for continuous
        observations. This function should be decorated with @nb.njit. Not used for discrete observations.
    --------------------

    --- _nb_get_TMat_flat ---
        returns the transition matrix TMat as a numpy array of shape (A, M, M) for continuous
        observations, and (Y, A, M, M) for discrete observations. This function should be decorated
        with @nb.njit.
    -------------------------
"""


class GeneralizedPolicyModel(BasePolicyModel):
    def __init__(self, obs_type,
                 M, A, Y,
                 seed = None,
                 policy_params = {"theta": None, "zeta": None}):
        """
        Generalized policy model for discrete observations. The generalized policy is parametrized by the product of two 
        Boltzmann / softmax distributions. The first distribution is the transition probability from the current memory to
        the next memory, conditioned on the observation, the previous memory and the action. The second distribution is the
        action policy, which is the probability of taking an action conditioned on the memory. The default name of the
        parameters of the policy are theta for the memory transition and zeta for the action policy.
        
        The parameters of the policy must be of shape (Y, A, M, M) for the memory transition and (A, M) for the action
        policy. The generalized policy is defined as

        *** g(m' | a, m, y) = exp(theta[y, a, m, m']) / sum_{m''} exp(theta[y, a, m, m'']) ***
        *** pi(a | m) = exp(zeta[a, m]) / sum_{a'} exp(zeta[a', m]) ***

        *** T(a, m' | m, y) = g(m' | a, m, y) * pi(a | m) ***

        
        Parameters
        ----------
        obs_type : str
            Type of observations. Must be "discrete" for consistency.
        M : int
            Number of memories.
        A : int
            Number of actions.
        Y : int
            Number of observations.
        seed : int
            Seed for random number generation. Default is None.
        policy_params : dict
            Dictionary of policy parameters.
            Default is {"theta": None, "zeta": None}.
            If the parameters are None, they are randomly initialized using the seed, if provided.

        Raises
        ------
        AssertionError
            If the observation type is not "discrete".
        AssertionError
            If the number of parameters is not equal to the expected number of parameters.
        ValueError
            If the policy parameters shapes are not (Y, A, M, M) and (A, M) respectively.
        TypeError
            If the policy parameters are not a numpy array or a torch tensor.

        Returns
        -------
        None
        """

        assert obs_type == "discrete", "SoftmaxDiscreteObs only supports discrete observations."

        name = "Observation-independent actions for discrete observations"
        expected_params_shape = ((Y, A, M, M), 
                                 (A, M))
        
        assert len(expected_params_shape) == len(policy_params), \
            f"Expected {len(expected_params_shape)} parameters, but got {len(policy_params)}."
    
        super().__init__(name, obs_type,
                         param_names = list(policy_params.keys()),
                         expected_params_shape = expected_params_shape,
                         M = M, A = A,
                         Y = Y)

        self._BasePolicyModel__check_policy_model_consistency(policy_params)
        self._BasePolicyModel__initialize_parameters(seed, policy_params)


    ##################################
    # Numpy functions for generation #
    ##################################

    def get_TMat_numpy(self):
        """
        Method to return the transition matrix TMat as a numpy array. TMat must have shape (Y, A, M, M), where the last
        two dimensions are the previous memory and the new memory, respectively.

        Returns
        -------
        numpy.ndarray
            Transition matrix TMat.
        """

        pi = utils.softmax(self.params[1], axis = 0)
        g = utils.softmax(self.params[0], axis = 3)
        
        TM = g * pi[None, ..., None]
        return TM

    def get_action_policy_numpy(self):
        """
        Method to return the action policy pi as a numpy array. pi must have shape (A, M), and is the probability of
        taking an action given the previous memory and the observation.

        Returns
        -------
        numpy.ndarray
            Action policy pi.
        """

        pi = utils.softmax(self.params[1], axis = 0)
        return pi.transpose(1, 0)

    def get_memory_transition_numpy(self):
        """
        Method to return the memory transition matrix g as a numpy array. g must have shape (Y, A, M, M), where the last
        two dimensions are the previous memory and the new memory, respectively. g is the probability of transitioning
        to a new memory given the previous memory, the action, and the observation.

        Returns
        -------
        numpy.ndarray
            Memory transition matrix g.
        """

        g = utils.softmax(self.params[0], axis = 3)
        return g
    
    ##################################
    ##################################
    ##################################


    #################################
    # Torch functions for inference #
    #################################

    def get_TMat_torch(self):
        """
        Method to return the transition matrix TMat as a torch tensor. TMat must have shape (Y, A, M, M), where the last
        two dimensions are the previous memory and the new memory, respectively.

        Returns
        -------
        torch.Tensor
            Transition matrix TMat.
        """

        pi = torch.softmax(self.params[1], dim = 0)
        g = torch.softmax(self.params[0], dim = 3)
        
        TM = g * pi[None, ..., None]
        return TM
    
    def get_action_policy_torch(self):
        """
        Method to return the action policy pi as a torch tensor. pi must have shape (A, M), and is the probability of
        taking an action given the previous memory and the observation.

        Returns
        -------
        torch.Tensor
            Action policy pi.
        """

        pi = torch.softmax(self.params[1], dim = 0)
        return pi.permute(1, 0)

    def get_memory_transition_torch(self):
        """
        Method to return the memory transition matrix g as a torch tensor. g must have shape (Y, A, M, M), where the last
        two dimensions are the previous memory and the new memory, respectively. g is the probability of transitioning
        to a new memory given the previous memory, the action, and the observation.

        Returns
        -------
        torch.Tensor
            Memory transition matrix g.
        """

        g = torch.softmax(self.params[0], dim = 3)
        return g

    #################################
    #################################
    #################################


    ###########################################
    # Numba functions for generation and APSO #
    ###########################################

    @staticmethod
    @nb.njit
    def _nb_get_TMat_flat(params, M, A, Y):
        """
        Numba-accelerated static method to return the transition matrix TMat as a numpy array. The parameters are assumed to be flattened
        from their original shape to ensure compatibility with APSO (adaptive particle swarm optimization). TMat must have shape
        (Y, A, M, M), where the last two dimensions are the previous memory and the new memory, respectively.

        Parameters
        ----------
        params : numpy.ndarray
            Flattened policy parameters.
        M : int
            Number of memories.
        A : int
            Number of actions.
        Y : int
            Number of observations.

        Returns
        -------
        numpy.ndarray
            Transition matrix TMat.
        """

        theta = params[ : Y*A*M*M]
        zeta = params[Y*A*M*M : ]

        g_shape = (Y, A, M, M)
        pi_shape = (A, M)
        
        # Create output array and pre-allocate temp arrays to avoid repeated allocations
        TM = np.zeros(g_shape, dtype=np.float64)
        pi_temp = np.zeros(A, dtype=np.float64)
        g_temp = np.zeros(M, dtype=np.float64)
        
        # Work with views but compute into temporary variables
        g_view = theta.reshape(g_shape)
        pi_view = zeta.reshape(pi_shape)

        # Optimize both pi and g: merge loops and combine operations
        for m in range(M):
            # Process pi for this memory state m - combined max, exp, and sum
            max_pi_m = pi_view[0, m]
            for a in range(1, A):
                if pi_view[a, m] > max_pi_m:
                    max_pi_m = pi_view[a, m]
            
            pi_sum = 0.0
            for a in range(A):
                pi_temp[a] = np.exp(pi_view[a, m] - max_pi_m)
                pi_sum += pi_temp[a]
            
            # Normalize pi and process g in the same loop
            for a in range(A):
                pi_temp[a] /= pi_sum
                
                # Process g for this (m, a) combination across all observations
                for y in range(Y):
                    # Combined max finding and exp computation for g
                    max_g_slice = g_view[y, a, m, 0]
                    for m_next in range(1, M):
                        if g_view[y, a, m, m_next] > max_g_slice:
                            max_g_slice = g_view[y, a, m, m_next]
                    
                    g_sum = 0.0
                    for m_next in range(M):
                        g_temp[m_next] = np.exp(g_view[y, a, m, m_next] - max_g_slice)
                        g_sum += g_temp[m_next]
                    
                    # Normalize g and compute final TM in one step
                    for m_next in range(M):
                        TM[y, a, m, m_next] = (g_temp[m_next] / g_sum) * pi_temp[a]

        return TM

    ###########################################
    ###########################################
    ###########################################

