import numpy as np
import numba as nb

strat_names = ("S1 exploration", "S2 exploitation", "S3 convergence", "S4 jumping out")

@nb.njit
def S1_exploration(f):
    """
    This function returns the value of the exploration strategy given the value
    of the fitness parameter f.

    The exploration strategy increases the cognitive parameter c1 and decreases
    the social parameter c2. It is more likely at intermediate values of f. A higher
    value of the strategy implies that the strategy is more likely to be chosen.

    Parameters
    ----------
    f : float
        The fitness parameter f.

    Returns
    -------
    float
        The value of the exploration strategy. 
    """
    if f <= 0.4 or f > 0.8:
        return 0.
    elif 0.4 < f <= 0.6:
        return 5 * f - 2
    elif 0.6 < f <= 0.7:
        return 1.
    else:
        return -10 * f + 8

@nb.njit
def S2_exploitation(f):
    """
    This function returns the value of the exploitation strategy given the value
    of the fitness parameter f.

    The exploitation strategy increases the cognitive parameter c1 and decreases
    the social parameter c2 by a fraction smaller than the exploration strategy.
    It is more likely to be selected at lower-intermediate values of f. A higher
    value of the strategy implies that the strategy is more likely to be chosen.

    Parameters
    ----------
    f : float
        The fitness parameter f.

    Returns
    -------
    float
        The value of the exploitation strategy.
    """
    if f <= 0.2 or f > 0.6:
        return 0.
    elif 0.2 < f <= 0.3:
        return 10 * f - 2
    elif 0.3 < f <= 0.4:
        return 1.
    else:
        return -5 * f + 3

@nb.njit
def S3_convergence(f):
    """
    This function returns the value of the convergence strategy given the value
    of the fitness parameter f.

    The convergence strategy increases both the cognitive and social parameters
    by a small and equal fraction. It is more likely to be selected at lower values
    of f, when the algorithm is close to convergence. A higher value of the strategy
    implies that the strategy is more likely to be chosen.

    Parameters
    ----------
    f : float
        The fitness parameter f.

    Returns
    -------
    float
        The value of the convergence strategy.
    """
    if f <= 0.1 or f > 0.3:
        return 1.
    else:
        return -5 * f + 1.5

@nb.njit
def S4_jumping_out(f):
    """
    This function returns the value of the jumping out strategy given the value
    of the fitness parameter f.

    The jumping out strategy decreases the cognitive parameter c1 and increases
    the social parameter c2. It is more likely to be selected at higher values of
    f, when the algorithm is stuck in a local minimum. A higher value of the strategy
    implies that the strategy is more likely to be chosen.

    Parameters
    ----------
    f : float
        The fitness parameter f.

    Returns
    -------
    float
        The value of the jumping out strategy.
    """
    if f <= 0.7:
        return 0.
    elif 0.7 < f <= 0.9:
        return 5 * f - 3.5
    else:
        return 1.

@nb.njit(parallel = True)
def particle_swarm_optimization_discrete(nb_get_TMat_flat, trainable_mask,
                                         n_dimensions, n_particles, n_iterations,
                                         M, A, Y, observations, actions,
                                         init_particles, init_velocities,
                                         c1 = 2.0, c2 = 2.0, w = 0.9,
                                         sigma_min = 0.1, sigma_max = 1.,
                                         verbose = True, verbose_epochs = True,
                                         init_memory_obs_dependent = False):

    particles = init_particles
    velocities = init_velocities
    pbest = particles.copy()

    gbest_array = np.zeros((n_iterations, n_dimensions), dtype=np.float64)
    gbest_values_array = np.zeros(n_iterations, dtype=np.float64)

    pbest_values = np.zeros(n_particles)
    for j in nb.prange(n_particles):
        pbest_values[j], _ = fun_to_min_discrete(nb_get_TMat_flat, particles[j], M, A, Y, observations, actions, init_memory_obs_dependent)

    idx_min = pbest_values.argmin()
    gbest = pbest[idx_min]
    gbest_value = pbest_values[idx_min]

    if verbose_epochs:
        print("Initial best value: ", gbest_value)

    gbest_array[0] = gbest
    gbest_values_array[0] = gbest_value

    idxs_trainable = np.where(trainable_mask)[0]
    current_strategy = 0

    # if there are infinite values in init_particles, it means they have been initialized like this and
    # they won't be trained
    masks_infs = np.zeros((n_particles, n_dimensions), dtype=np.bool_)
    for i in range(n_particles):
        masks_infs[i] = ~np.isinf(particles[i])


    for idx_iter in range(n_iterations):
        for j in nb.prange(n_particles):
            new_velocity = velocities[j].copy()
            new_position = particles[j].copy()

            for idx_dim in idxs_trainable:
                new_velocity[idx_dim] *= w
                new_velocity[idx_dim] += c1*np.random.rand()*(pbest[j, idx_dim] - particles[j, idx_dim])
                new_velocity[idx_dim] += c2*np.random.rand()*(gbest[idx_dim] - particles[j, idx_dim])

                new_position[idx_dim] += new_velocity[idx_dim]

            fitness_value, flag_trj = fun_to_min_discrete(nb_get_TMat_flat, new_position, M, A, Y, observations, actions, init_memory_obs_dependent)

            if not flag_trj:
                velocities[j] = new_velocity.copy()
                particles[j] = new_position.copy()

                # Update the pbest
                if fitness_value < pbest_values[j]:
                    pbest_values[j] = fitness_value
                    pbest[j] = particles[j]
            else:
                velocities[j] *= w

        idx_best = pbest_values.argmin()
        if pbest_values[idx_best] < gbest_value:
            gbest = pbest[idx_best]
            gbest_value = pbest_values[idx_best]

        # get the average distance between the particles for normalization
        D = np.zeros(n_particles, dtype = np.float64)

        for i in nb.prange(n_particles):
            for j in range(n_particles):
                diff = particles[i] - particles[j]
                mask = masks_infs[i] & masks_infs[j]
                D[i] += np.sqrt(np.sum((diff[mask])**2))

        D /= n_particles - 1
        Dmin = D.min()
        Dmax = D.max()
        
        fval = (D[idx_best] - Dmin) / (Dmax - Dmin + 1e-10)

        strategies_vals = np.zeros(4, dtype=np.float64)
        strategies_vals[0] = S1_exploration(fval)
        strategies_vals[1] = S2_exploitation(fval)
        strategies_vals[2] = S3_convergence(fval)
        strategies_vals[3] = S4_jumping_out(fval)

        non_zero_indices = [i for i in range(4) if strategies_vals[i] != 0]
        count_zero = 4 - len(non_zero_indices)

        if count_zero == 3:
            for i in range(4):
                if strategies_vals[i] != 0:
                    current_strategy = i
                    break
        elif count_zero == 2:
            for i in range(1, 5):
                next_strategy = (current_strategy + i) % 4
                if strategies_vals[next_strategy] != 0:
                    current_strategy = next_strategy
                    break

        delta = 0.05 + (np.random.random() * 0.05)

        if current_strategy == 0:
            c1 += delta
            c2 -= delta
        elif current_strategy == 1:
            c1 += 0.5 * delta
            c2 -= 0.5 * delta
        elif current_strategy == 2:
            c1 += 0.5 * delta
            c2 += 0.5 * delta
        elif current_strategy == 3:
            c1 -= delta
            c2 += delta

        c1 = max(1.5, min(2.5, c1))
        c2 = max(1.5, min(2.5, c2))

        if c1 + c2 > 4.0:
            c1 = 4.0 * (c1 / (c1 + c2))
            c2 = 4.0 * (c2 / (c1 + c2))

        w = 1 / (1 + 1.5 * np.exp(-2.6 * fval))

        if current_strategy == 2:
            sigma = sigma_max - (sigma_max - sigma_min) * idx_iter / n_iterations

            random_dim = np.random.randint(0, len(idxs_trainable))
            mutated_gbest = gbest.copy()
            mutated_gbest[idxs_trainable[random_dim]] += np.random.randn() * sigma
            mutated_fitness_value, flag_trj = fun_to_min_discrete(nb_get_TMat_flat, mutated_gbest, M, A, Y, observations, actions, init_memory_obs_dependent)

            if not flag_trj and mutated_fitness_value < gbest_value:
                if verbose:
                    print("\t Mutated gbest: ", gbest_value, "-> ", mutated_fitness_value)
                gbest = mutated_gbest
                gbest_value = mutated_fitness_value

        gbest_array[idx_iter] = gbest
        gbest_values_array[idx_iter] = gbest_value

        if verbose_epochs:
            print(f"Iteration {idx_iter+1}/{n_iterations}, best value:", gbest_value)
        if verbose:
            print("\t f value:", np.round(fval, 3), "- strategy:", strat_names[current_strategy], "- c1:", np.round(c1, 3), "- c2:", np.round(c2, 3), "w: ", np.round(w, 3))

    return gbest_array, gbest_values_array

@nb.njit(parallel=True)
def particle_swarm_optimization_discrete_kNN(nb_get_TMat_flat, trainable_mask,
                                             n_dimensions, n_particles, n_iterations,
                                             M, A, Y, observations, actions,
                                             init_particles, init_velocities,
                                             num_neighbors_init, num_neighbors_final,
                                             num_neighbors_mid,
                                             c1 = 2.0, c2 = 2.0, w = 0.9,
                                             sigma_min = 0.1, sigma_max = 1.,
                                             verbose = True, verbose_epochs = True,
                                             init_memory_obs_dependent = False):
    particles = init_particles
    velocities = init_velocities
    pbest = particles.copy()

    # print("Initialization", particles)

    gbest_array = np.zeros((n_iterations, n_dimensions), dtype=np.float64)
    gbest_values_array = np.zeros(n_iterations, dtype=np.float64)

    pbest_values = np.zeros(n_particles)
    for j in nb.prange(n_particles):
        pbest_values[j], _ = fun_to_min_discrete(nb_get_TMat_flat, particles[j], M, A, Y, observations, actions, init_memory_obs_dependent)

    idx_min = pbest_values.argmin()
    gbest = pbest[idx_min]
    gbest_value = pbest_values[idx_min]

    lbest = pbest.copy()
    lbest_values = pbest_values.copy()

    if verbose_epochs:
        print("Initial best value:", gbest_value)

    gbest_array[0] = gbest
    gbest_values_array[0] = gbest_value

    idxs_trainable = np.where(trainable_mask)[0]

    current_strategy = 0

    # if there are infinite values in init_particles, it means they have been initialized like this and
    # they won't be trained
    masks_infs = np.zeros((n_particles, n_dimensions), dtype=np.bool_)
    for i in nb.prange(n_particles):
        masks_infs[i] = ~np.isinf(particles[i])

    for idx_iter in range(n_iterations):

        if idx_iter < n_iterations // 2:
            num_neighbors = int(num_neighbors_init + (num_neighbors_mid - num_neighbors_init) * (idx_iter / n_iterations) ** 4)
        else:
            num_neighbors = int(num_neighbors_mid + (num_neighbors_final - num_neighbors_mid) * ((idx_iter - n_iterations // 2) / (n_iterations // 2)) ** 4)

        if verbose:
            print(f"\t Number of spatial neighbors: {num_neighbors}")

        # Compute pairwise distances and reuse for both local best and adaptive strategies
        distances = np.zeros((n_particles, n_particles), dtype=np.float64)
        for i in nb.prange(n_particles):
            for j in range(i + 1, n_particles):
                mask = masks_infs[i] & masks_infs[j]
                diff = particles[i] - particles[j]
                distances[i, j] = np.sqrt(np.sum(diff[mask] ** 2))
                distances[j, i] = distances[i, j]
        
        # Find local best (lbest) for each particle
        for i in nb.prange(n_particles):
            # Select the closest num_neighbors
            neighbors = np.argsort(distances[i])[:num_neighbors]
            best_neighbor_idx = neighbors[np.argmin(pbest_values[neighbors])]

            # update lbest only if the best neighbor is better than the current lbest
            if pbest_values[best_neighbor_idx] < lbest_values[i]:
                lbest[i] = pbest[best_neighbor_idx]
                lbest_values[i] = pbest_values[best_neighbor_idx]

            # Mutate the lbest value
            if current_strategy == 2:  # Apply mutation only for the convergence strategy
                sigma = sigma_max - (sigma_max - sigma_min) * idx_iter / n_iterations
                random_dim = np.random.randint(0, len(idxs_trainable))
                mutated_lbest = lbest[i].copy()
                mutated_lbest[idxs_trainable[random_dim]] += np.random.randn() * sigma

                # Evaluate the fitness of the mutated lbest
                mutated_fitness_value, flag_trj = fun_to_min_discrete(
                    nb_get_TMat_flat, mutated_lbest, M, A, Y, observations, actions, init_memory_obs_dependent
                )

                # Accept the mutation only if it improves the fitness and is better than the current lbest
                current_lbest_fitness, _ = fun_to_min_discrete(
                    nb_get_TMat_flat, lbest[i], M, A, Y, observations, actions, init_memory_obs_dependent
                )
                if not flag_trj and mutated_fitness_value < current_lbest_fitness:
                    lbest[i] = mutated_lbest
                    lbest_values[i] = mutated_fitness_value
                    
        # Update particles
        for j in nb.prange(n_particles):
            new_velocity = velocities[j].copy()
            new_position = particles[j].copy()

            for idx_dim in idxs_trainable:
                new_velocity[idx_dim] *= w
                new_velocity[idx_dim] += c1 * np.random.rand() * (pbest[j, idx_dim] - particles[j, idx_dim])
                new_velocity[idx_dim] += c2 * np.random.rand() * (lbest[j, idx_dim] - particles[j, idx_dim])

                new_position[idx_dim] += new_velocity[idx_dim]

            fitness_value, flag_trj = fun_to_min_discrete(nb_get_TMat_flat, new_position, M, A, Y, observations, actions, init_memory_obs_dependent)

            if not flag_trj:
                velocities[j] = new_velocity.copy()
                particles[j] = new_position.copy()

                # Update the pbest
                if fitness_value < pbest_values[j]:
                    pbest_values[j] = fitness_value
                    pbest[j] = particles[j]
            else:
                velocities[j] *= w

        if current_strategy == 2:
            sigma = sigma_max - (sigma_max - sigma_min) * idx_iter / n_iterations

            for j in nb.prange(n_particles):
                random_dim = np.random.randint(0, len(idxs_trainable))
                mutated_lbest = lbest[j].copy()
                mutated_lbest[idxs_trainable[random_dim]] += np.random.randn() * sigma
                mutated_fitness_value, flag_trj = fun_to_min_discrete(nb_get_TMat_flat, mutated_lbest, M, A, Y, observations, actions, init_memory_obs_dependent)

                if not flag_trj and mutated_fitness_value < lbest_values[j]:
                    if verbose:
                        print("\t Mutated lbest: ", lbest_values[j], "-> ", mutated_fitness_value)
                    lbest[j] = mutated_lbest
                    lbest_values[j] = mutated_fitness_value

        idx_best = lbest_values.argmin()
        if lbest_values[idx_best] < gbest_value:
            gbest = lbest[idx_best]
            gbest_value = lbest_values[idx_best]

        gbest_array[idx_iter] = gbest
        gbest_values_array[idx_iter] = gbest_value

        # Adaptive strategies, ignoring infinite values
        D = distances.sum(axis=1) / (n_particles - 1)
        Dmin = D.min()
        Dmax = D.max()
        
        fval = (D[idx_best] - Dmin) / (Dmax - Dmin + 1e-10)

        strategies_vals = np.zeros(4, dtype=np.float64)
        strategies_vals[0] = S1_exploration(fval)
        strategies_vals[1] = S2_exploitation(fval)
        strategies_vals[2] = S3_convergence(fval)
        strategies_vals[3] = S4_jumping_out(fval)

        non_zero_indices = [i for i in range(4) if strategies_vals[i] != 0]
        count_zero = 4 - len(non_zero_indices)

        if count_zero == 3:
            for i in range(4):
                if strategies_vals[i] != 0:
                    current_strategy = i
                    break
        elif count_zero == 2:
            for i in range(1, 5):
                next_strategy = (current_strategy + i) % 4
                if strategies_vals[next_strategy] != 0:
                    current_strategy = next_strategy
                    break

        delta = 0.05 + (np.random.random() * 0.05)

        if current_strategy == 0:
            c1 += delta
            c2 -= delta
        elif current_strategy == 1:
            c1 += 0.5 * delta
            c2 -= 0.5 * delta
        elif current_strategy == 2:
            c1 += 0.5 * delta
            c2 += 0.5 * delta
        elif current_strategy == 3:
            c1 -= delta
            c2 += delta

        c1 = max(1.5, min(2.5, c1))
        c2 = max(1.5, min(2.5, c2))

        if c1 + c2 > 4.0:
            c1 = 4.0 * (c1 / (c1 + c2))
            c2 = 4.0 * (c2 / (c1 + c2))

        w = 1 / (1 + 1.5 * np.exp(-2.6 * fval))

        if verbose_epochs:
            print(f"Iteration {idx_iter+1}/{n_iterations}, best value:", gbest_value)
        if verbose:
            print("\t f value:", np.round(fval, 3), "- strategy:", strat_names[current_strategy], "- c1:", np.round(c1, 3), "- c2:", np.round(c2, 3), "w: ", np.round(w, 3))
            print("\t Average local best fitness:", np.round(np.mean(lbest_values), 3), "+-", np.round(np.std(lbest_values), 3))

    return gbest_array, gbest_values_array




@nb.njit
def nb_evaluate_nloglikelihood_flat_discrete(TMat, observations, actions, rho):
    nLL = 0.
    m = np.zeros_like(rho)

    for t in range(actions.size):
        a = actions[t]
        obs = observations[t]

        transition_probs = TMat[obs, a].T

        if np.sum(transition_probs) == 0:
            nLL = np.nan
            break
            
        if t == 0:
            new_m = transition_probs @ rho
        else:
            new_m = transition_probs @ m

        if np.sum(new_m) == 0:
            nLL = np.nan
            break

        mv = np.sum(new_m)
        new_nLL = nLL - np.log(mv)
        new_m /= mv

        if np.isnan(nLL):
            break

        m = new_m
        nLL = new_nLL

    return nLL - np.log(np.sum(m))


@nb.njit
def fun_to_min_discrete(nb_get_TMat_flat, particle_pos, M, A, Y, observations, actions, init_memory_obs_dependent,
                        toprint = False):
    if init_memory_obs_dependent:
        params = particle_pos[:-Y*M]
        params_psi = particle_pos[-Y*M:]
    else:
        params = particle_pos[:-M]
        
        rho = np.exp(particle_pos[-M:])
        rho /= np.sum(rho)
    TMat = nb_get_TMat_flat(params, M, A, Y)

    NTraj = len(actions)

    nLL = 0.
    counter = 0
    flag_uncompatible_traj = False

    for i in range(NTraj):
        if init_memory_obs_dependent:
            init_obs = observations[i][0]
            rho = np.exp(params_psi[init_obs * M:(init_obs + 1) * M])
            rho /= np.sum(rho)

        new_nLL = nb_evaluate_nloglikelihood_flat_discrete(TMat, observations[i], actions[i], rho)
        if not np.isnan(new_nLL) and new_nLL != np.inf:
            nLL += new_nLL
            counter += 1
        else:
            flag_uncompatible_traj = True
            break

    if counter != 0:
        nLL = nLL / counter

    return nLL, flag_uncompatible_traj
