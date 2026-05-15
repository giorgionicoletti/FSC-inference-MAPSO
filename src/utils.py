import numpy as np
import numba as nb

import networkx as nx
import matplotlib.pyplot as plt

import torch

import networkx as nx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def softmax(x, axis = 0):
    """
    Computes the softmax of an array x along a given axis.

    Parameters:
    --- x: np.array
        Array to be softmaxed.
    --- axis: int or tuple of ints
        Axis or axes along which the softmax is computed.

    Returns:
    --- np.array
        Softmaxed array, of the same shape as x.
    """
    max_x = np.max(x, axis = axis, keepdims = True)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x, axis = axis, keepdims = True)
    return exp_x / sum_exp_x

@nb.njit
def numba_random_choice(vals, probs):
    """
    Chooses a value from vals with probabilities given by probs.

    Parameters:
    --- vals: np.array
        Array of values to choose from.
    --- probs: np.array
        Array of probabilities for each value in vals.

    Returns:
    --- object
        Value chosen from vals.
    """
    r = np.random.rand()
    cum_probs = np.cumsum(probs)
    for idx in range(len(probs)):
        if r < cum_probs[idx]:
            return vals[idx]

def combine_spaces(space1, space2):
    """
    Combines two spaces into a single space. Useful to index the combined
    space with a single index.

    Parameters:
    --- space1: np.array
        First space to be combined.
    --- space2: np.array
        Second space to be combined.

    Returns:
    --- np.array
        Combined space, with shape (space1.size * space2.size, 2).
    """
    return np.array(np.meshgrid(space1, space2)).T.reshape(-1, 2)

def draw_FSC_complex_network(transition_weights, action_probabilities=None, memory_ids=None, arrangement='horizontal', 
                                          spacing=6, fig_size=None, min_weightsum_threshold=0.01,
                                          mem_node_size=2000, action_node_size=1200, obs_node_size=500,
                                          action_node_yoffset=1.5, action_node_xoffset=1.5,
                                          obs_node_xoffset=0.7, obs_node_yoffset=0.7,
                                          max_line_width=1, min_line_width=0.,
                                          max_action_width=3, min_action_width=0,
                                          memory_names=None, action_names=None, observation_names=None,
                                          suppress_zero_action_transitions=False, action_prob_threshold=1e-10,
                                          fade_no_incoming=False, no_incoming_alpha=0.3,
                                          arrowhead_distance = 15, arrowhead_distance_actions = 15,
                                          action_edge_color='blue', memory_node_color='lightblue', 
                                          action_node_color='lightgreen', observation_node_colors=None,
                                          plot_legend = False, reverse_obs_below=True, hide_unused_actions=False, 
                                          AllowedObsFromAct=None, obs_rotation=0.0, ax=None):
    """
    Draw multiple complex node structures with weighted transitions between observation and memory nodes.
    
    Parameters:
    -----------
    transition_weights : numpy.ndarray
        Shape (Y, A, M, M) where:
        - Y: observation index
        - A: action index  
        - M: starting memory node index
        - M: ending memory node index
    action_probabilities : numpy.ndarray or None
        Shape (M, A) - probability of taking action A from memory state M
        Controls thickness of edges from memory nodes to action nodes
    memory_ids : list or None
        List of memory node IDs that determines the spatial order of complex nodes.
        This completely rearranges both the visual layout AND the underlying data.
        For example, memory_ids=[1,0,2] will swap the first two memory nodes including
        all their transition probabilities and action probabilities.
        Must contain exactly the values [0, 1, ..., M-1] in any order.
        If None, uses [0, 1, 2, ..., M-1]
    arrangement : str
        'horizontal' or 'vertical' arrangement
    spacing : float
        Distance between adjacent complex nodes
    fig_size : tuple or None
        Figure size (width, height). If None, automatically calculated
    min_weightsum_threshold : float
        Minimum weight sum for a memory node to avoid being faded (if fade_no_incoming=True)
    max_line_width : float
        Maximum line width for strongest transition connections
    min_line_width : float
        Minimum line width for weakest transition connections
    max_action_width : float
        Maximum line width for action connections (probability = 1.0)
    min_action_width : float
        Minimum line width for action connections (probability = 0.0)
    memory_names : list or None
        Custom names for memory nodes. If None, uses default 'M{id}' format
    action_names : list or None
        Custom names for action nodes. If None, uses default 'A0', 'A1', etc.
    observation_names : list or None
        Custom names for observation nodes. If None, uses default 'O0', 'O1', etc.
    suppress_zero_action_transitions : bool
        If True, suppress transition edges from observation nodes when the corresponding
        memory-to-action probability is below action_prob_threshold. Default: False
    action_prob_threshold : float
        Threshold below which action probabilities are considered zero for suppression.
        Only used when suppress_zero_action_transitions=True. Default: 1e-10
    fade_no_incoming : bool
        If True, apply reduced alpha to complex nodes that have no incoming connections
        from any other memory node. Default: False
    no_incoming_alpha : float
        Alpha (transparency) value for complex nodes with no incoming connections.
        Only used when fade_no_incoming=True. Range: 0.0 (fully transparent) to 1.0 (opaque).
        Default: 0.3
    arrowhead_distance : float
        Distance (in points) between arrowheads and target nodes. Larger values create
        more space between the arrowhead tip and the node edge. This affects all edges
        with arrows (action edges and transition edges). Default: 15
    action_edge_color : str
        Color for action probability edges (memory to action nodes). Default: 'blue'
    memory_node_color : str
        Color for memory nodes (large circles). Default: 'lightblue'
    action_node_color : str
        Color for action nodes (squares). Default: 'lightgreen'
    observation_node_colors : list or None
        List of colors for observation nodes (diamonds). Must have exactly Y elements
        if provided. Default: None (uses ['lightcoral', 'lightsalmon'] for Y=2)
    plot_legend : bool
        Whether to display a legend explaining the different node types and edge colors.
        Default: False
    reverse_obs_below : bool
        Whether to reverse the order of observation nodes for action nodes positioned
        below the memory node. When True (default), observation nodes are mirrored:
        - For Y=2: Top actions: left obs = obs[0], right obs = obs[1]
                   Bottom actions: left obs = obs[1], right obs = obs[0]
        - For Y>2: Top actions: circular arrangement starting from right (0°)
                   Bottom actions: mirrored circular arrangement (angles negated)
        This creates a symmetric layout around the memory node. Default: True
    hide_unused_actions : bool
        Whether to hide action nodes and their associated observation nodes when
        action_probabilities[m, a] = 0. When True, action nodes with zero probability
        will not be drawn, along with their observation nodes and edges. This requires
        action_probabilities to be provided. Default: False
    AllowedObsFromAct : numpy.ndarray or None
        Shape (A, Y) boolean array specifying which observations are allowed from each action.
        If AllowedObsFromAct[a, y] is False, the observation node O_y for action a will not
        be drawn and all transition weights from that observation will be set to zero.
        If None, all observations are allowed from all actions. Default: None
    obs_rotation : float
        Rotation angle (in radians) to apply to observation node positions when using
        circular arrangement (for Y > 2). Positive values rotate counter-clockwise.
        For example, π/4 (≈0.785) rotates by 45 degrees. Default: 0.0 (no rotation)
    ax : matplotlib.axes.Axes or None
        Matplotlib axis to plot on. If None, creates a new figure with plt.figure().
        When provided, the plot will be drawn on this axis instead of creating a new figure.
        Default: None
    """
    Y, A, M, M_end = transition_weights.shape
    assert M == M_end, "Transition matrix must be square in memory dimensions"
    
    # Validate action probabilities if provided
    if action_probabilities is not None:
        assert action_probabilities.shape == (M, A), f"Action probabilities must be shape (M, A) = ({M}, {A}), got {action_probabilities.shape}"
    
    # Validate AllowedObsFromAct if provided
    if AllowedObsFromAct is not None:
        assert isinstance(AllowedObsFromAct, np.ndarray), "AllowedObsFromAct must be a numpy array"
        assert AllowedObsFromAct.dtype == bool, "AllowedObsFromAct must be a boolean array"
        assert AllowedObsFromAct.shape == (A, Y), f"AllowedObsFromAct must be shape (A, Y) = ({A}, {Y}), got {AllowedObsFromAct.shape}"
    
    num_nodes = M
    
    # Set default memory IDs if not provided
    if memory_ids is None:
        memory_ids = list(range(num_nodes))
    elif len(memory_ids) != num_nodes:
        raise ValueError(f"Length of memory_ids ({len(memory_ids)}) must match M ({M})")
    
    # Validate that memory_ids contains unique values in range [0, M-1]
    if set(memory_ids) != set(range(M)):
        raise ValueError(f"memory_ids must contain exactly the values [0, 1, ..., {M-1}], got {memory_ids}")
    
    # Create permutation arrays to rearrange data according to memory_ids
    # memory_ids[i] tells us which original memory should be at position i
    perm_indices = memory_ids  # This is the permutation for rows/columns
    
    # Rearrange transition weights: (Y, A, M, M) -> permute both M dimensions
    transition_weights_permuted = transition_weights[:, :, perm_indices, :][:, :, :, perm_indices]
    
    # Rearrange action probabilities: (M, A) -> permute M dimension
    if action_probabilities is not None:
        action_probabilities_permuted = action_probabilities[perm_indices, :]
    else:
        action_probabilities_permuted = None
    
    # Identify memory nodes with no incoming connections (if fade_no_incoming is enabled)
    nodes_with_no_incoming = set()
    if fade_no_incoming:
        # Iteratively identify nodes that should be faded
        # Start with nodes that have no incoming connections from any other node
        # Then add nodes that only receive connections from already-faded nodes
        
        # Compute weighted incoming connections for each memory node
        transition_weights_no_self = transition_weights_permuted.copy()
        
        # Set diagonal elements to zero to exclude self-connections
        for m in range(M):
            transition_weights_no_self[:, :, m, m] = 0
        
        # Create a matrix of weighted connections between memory nodes
        # Shape: (M_source, M_target) - weighted sum of connections from source to target
        memory_to_memory_weights = np.zeros((M, M))
        
        for y_idx in range(Y):
            for a_idx in range(A):
                for m_source in range(M):
                    for m_target in range(M):
                        if m_source != m_target:  # Exclude self-connections
                            weight = transition_weights_no_self[y_idx, a_idx, m_source, m_target]
                            if action_probabilities_permuted is not None:
                                action_prob = action_probabilities_permuted[m_source, a_idx]
                                weight *= action_prob
                            memory_to_memory_weights[m_source, m_target] += weight
        
        # Iteratively identify nodes to fade
        faded_nodes = set()
        changed = True
        
        while changed:
            changed = False
            for mem_idx in range(M):
                mem_id = memory_ids[mem_idx]
                
                if mem_id not in faded_nodes:
                    # Check if this node should be faded
                    # Sum incoming connections from non-faded nodes
                    incoming_from_active = 0
                    for source_idx in range(M):
                        source_id = memory_ids[source_idx]
                        if source_id not in faded_nodes:
                            incoming_from_active += memory_to_memory_weights[source_idx, mem_idx]
                    
                    # If no significant incoming connections from non-faded nodes, fade this node
                    if incoming_from_active <= min_weightsum_threshold:
                        faded_nodes.add(mem_id)
                        changed = True
        
        nodes_with_no_incoming = faded_nodes
        #print(f"Memory nodes to be faded (no incoming connections from active nodes): {sorted(nodes_with_no_incoming)}")
    
    # Set default names if not provided
    if memory_names is None:
        memory_names = [f'M{mem_id}' for mem_id in memory_ids]
    elif len(memory_names) != num_nodes:
        raise ValueError(f"Length of memory_names ({len(memory_names)}) must match M ({M})")
        
    if action_names is None:
        action_names = [f'A{i}' for i in range(A)]
    elif len(action_names) != A:
        raise ValueError(f"Length of action_names ({len(action_names)}) must match A ({A})")
        
    if observation_names is None:
        observation_names = [f'O{i}' for i in range(Y)]
    elif len(observation_names) != Y:
        raise ValueError(f"Length of observation_names ({len(observation_names)}) must match Y ({Y})")
    
    # Set default observation node colors if not provided
    if observation_node_colors is None:
        if Y == 2:
            observation_node_colors = ['lightcoral', 'lightsalmon']
        else:
            # For other Y values, create a default color palette
            default_colors = ['lightcoral', 'lightsalmon', 'lightpink', 'lightsteelblue', 'lightseagreen']
            observation_node_colors = (default_colors * ((Y // len(default_colors)) + 1))[:Y]
    elif len(observation_node_colors) != Y:
        raise ValueError(f"Length of observation_node_colors ({len(observation_node_colors)}) must match Y ({Y})")
    
    # Calculate figure size if not provided
    if fig_size is None:
        if arrangement == 'horizontal':
            fig_size = (6 * num_nodes, 8)
        else:  # vertical
            fig_size = (10, 4 * num_nodes)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Store all node names and positions
    all_nodes = []
    pos = {}
    
    # Store action positions for each memory node for later use in transition edge creation
    action_positions_dict = {}
    
    # Generate nodes and positions for each complex node
    for m_idx, mem_id in enumerate(memory_ids):
        # Calculate offset for this complex node
        if arrangement == 'horizontal':
            x_offset = (m_idx - (num_nodes - 1) / 2) * spacing
            y_offset = 0
        else:  # vertical
            x_offset = 0
            y_offset = ((num_nodes - 1) / 2 - m_idx) * spacing

        # Define node IDs for this complex node
        center_mem = f"M{mem_id}"
        
        # Calculate action node positions based on number of actions
        if A == 2:
            # Original layout: one above, one below
            action_positions = [
                (x_offset, y_offset + action_node_yoffset),  # top
                (x_offset, y_offset - action_node_yoffset)   # bottom
            ]
        elif A == 3:
            # One above, two below
            action_positions = [
                (x_offset, y_offset + action_node_yoffset),                    # top
                (x_offset - action_node_xoffset/2, y_offset - action_node_yoffset),  # bottom left
                (x_offset + action_node_xoffset/2, y_offset - action_node_yoffset)   # bottom right
            ]
        elif A == 4:
            # Two above, two below
            action_positions = [
                (x_offset - action_node_xoffset/2, y_offset + action_node_yoffset),  # top left
                (x_offset + action_node_xoffset/2, y_offset + action_node_yoffset),  # top right
                (x_offset - action_node_xoffset/2, y_offset - action_node_yoffset),  # bottom left
                (x_offset + action_node_xoffset/2, y_offset - action_node_yoffset)   # bottom right
            ]
        else:
            # General case: distribute actions in rows
            # For A actions, put ceil(A/2) on top, floor(A/2) on bottom
            top_actions = (A + 1) // 2
            bottom_actions = A // 2
            
            action_positions = []
            
            # Top row actions
            if top_actions == 1:
                action_positions.append((x_offset, y_offset + action_node_yoffset))
            else:
                for j in range(top_actions):
                    x_pos = x_offset + (j - (top_actions - 1) / 2) * action_node_xoffset
                    action_positions.append((x_pos, y_offset + action_node_yoffset))
            
            # Bottom row actions
            if bottom_actions == 1:
                action_positions.append((x_offset, y_offset - action_node_yoffset))
            else:
                for j in range(bottom_actions):
                    x_pos = x_offset + (j - (bottom_actions - 1) / 2) * action_node_xoffset
                    action_positions.append((x_pos, y_offset - action_node_yoffset))
        
        # Store action positions for this memory node
        action_positions_dict[mem_id] = action_positions
        
        # Create action nodes and their observation nodes
        complex_nodes = [center_mem]
        action_nodes_this_mem = []
        
        for a_idx in range(A):
            # Check if we should hide this action node due to zero probability
            if hide_unused_actions and action_probabilities_permuted is not None:
                action_prob = action_probabilities_permuted[m_idx, a_idx]  # i is the index in memory_ids
                if action_prob == 0:
                    continue  # Skip this action node and its observation nodes
            
            action_node = f"A{mem_id}_{a_idx}"
            action_nodes_this_mem.append(action_node)
            complex_nodes.append(action_node)
            
            # Create observation nodes for this action
            obs_nodes_this_action = []
            for y_idx in range(Y):
                # Only create observation node if this observation is allowed from this action
                if AllowedObsFromAct is None or AllowedObsFromAct[a_idx, y_idx]:
                    obs_node = f"O{mem_id}_A{a_idx}_{y_idx}"
                    obs_nodes_this_action.append(obs_node)
                    complex_nodes.append(obs_node)
                    
                    # Add internal edge from action to observation
                    G.add_edge(action_node, obs_node)
            
            # Add internal edges
            G.add_edge(center_mem, action_node)
        
        all_nodes.extend(complex_nodes)
        
        # Set positions
        pos[center_mem] = (x_offset, y_offset)
        
        for a_idx in range(A):
            # Check if this action node was skipped due to zero probability
            if hide_unused_actions and action_probabilities_permuted is not None:
                action_prob = action_probabilities_permuted[m_idx, a_idx]
                if action_prob == 0:
                    continue  # Skip positioning for this action node
            
            action_node = f"A{mem_id}_{a_idx}"
            pos[action_node] = action_positions[a_idx]
            
            # Position observation nodes relative to action node
            action_x, action_y = action_positions[a_idx]
            
            # Determine if this action is above or below the memory node
            is_action_below = action_y < y_offset
            
            # Get list of allowed observations for this action
            if AllowedObsFromAct is None:
                allowed_obs_indices = list(range(Y))
            else:
                allowed_obs_indices = [y_idx for y_idx in range(Y) if AllowedObsFromAct[a_idx, y_idx]]
            
            # Calculate observation node positions based on number of allowed observations
            Y_allowed = len(allowed_obs_indices)
            
            if Y_allowed == 0:
                # No observations allowed for this action - skip positioning
                continue
            elif Y_allowed == 1:
                # Single observation - place directly at action position with small offset
                obs_positions = [(action_x, action_y + obs_node_yoffset * 0.3)]
            elif Y_allowed == 2:
                # For 2 observations, use left/right arrangement
                if reverse_obs_below and is_action_below:
                    # For actions below, reverse the observation order
                    obs_positions = [
                        (action_x + obs_node_xoffset, action_y),  # first allowed obs -> right position
                        (action_x - obs_node_xoffset, action_y)   # second allowed obs -> left position
                    ]
                else:
                    # For actions above (or when reverse_obs_below is False), use normal order
                    obs_positions = [
                        (action_x - obs_node_xoffset, action_y),  # first allowed obs -> left position
                        (action_x + obs_node_xoffset, action_y)   # second allowed obs -> right position
                    ]
            else:
                # For more than 2 observations, arrange in a pattern around the action node
                obs_positions = []
                
                if Y_allowed == 3:
                    # Special case for 3 observations: left, right, top/bottom
                    base_positions = [
                        (action_x - obs_node_xoffset, action_y),  # left
                        (action_x + obs_node_xoffset, action_y),  # right
                        (action_x, action_y + obs_node_yoffset),   # top
                        (action_x, action_y - obs_node_yoffset)   # bottom
                    ]
                    
                    # Select three positions
                    if reverse_obs_below and is_action_below:
                        # For actions below, mirror horizontally and use bottom instead of top
                        obs_positions = [
                            base_positions[1],  # right position
                            base_positions[0],  # left position
                            base_positions[3]   # bottom position
                        ]
                    else:
                        obs_positions = [
                            base_positions[0],  # left position
                            base_positions[1],  # right position
                            base_positions[2]   # top position
                        ]
                        
                else:
                    # For Y_allowed > 3, use circular arrangement
                    for i in range(Y_allowed):
                        # Calculate angle for this observation (starting from right, going counter-clockwise)
                        # Apply custom rotation offset
                        angle = 2 * np.pi * i / Y_allowed + obs_rotation
                        
                        # Apply reverse_obs_below logic for actions below the memory node
                        if reverse_obs_below and is_action_below:
                            # For actions below, mirror the angles horizontally
                            angle = -angle
                        
                        # Calculate position
                        obs_x = action_x + obs_node_xoffset * np.cos(angle)
                        obs_y = action_y + obs_node_xoffset * np.sin(angle)
                        obs_positions.append((obs_x, obs_y))
            
            # Set positions for allowed observation nodes only
            for i, y_idx in enumerate(allowed_obs_indices):
                if i < len(obs_positions):  # Safety check
                    obs_node = f"O{mem_id}_A{a_idx}_{y_idx}"
                    pos[obs_node] = obs_positions[i]
    
    # Add all nodes to the graph
    G.add_nodes_from(all_nodes)
    
    # Add weighted transition edges between observation and memory nodes
    transition_edges = []
    transition_weights_list = []
    
    # Find maximum weight for normalization
    max_weight = np.max(transition_weights_permuted)
    
    for y_idx in range(Y):  # observation index
        for a_idx in range(A):  # action index
            for m_start_idx in range(M):  # starting memory index (now corresponds to display order)
                for m_end_idx in range(M):  # ending memory index (now corresponds to display order)
                    weight = transition_weights_permuted[y_idx, a_idx, m_start_idx, m_end_idx]
                    
                    # Set weight to zero if this observation is not allowed from this action
                    if AllowedObsFromAct is not None and not AllowedObsFromAct[a_idx, y_idx]:
                        weight = 0
                    
                    # Check if we should suppress this transition due to zero action probability
                    if suppress_zero_action_transitions and action_probabilities_permuted is not None:
                        action_prob = action_probabilities_permuted[m_start_idx, a_idx]
                        if action_prob <= action_prob_threshold:
                            # Skip this transition - don't draw edges from observation nodes
                            # connected to actions with zero probability
                            continue
                    
                    # Check if this action node was hidden due to zero probability
                    if hide_unused_actions and action_probabilities_permuted is not None:
                        action_prob = action_probabilities_permuted[m_start_idx, a_idx]
                        if action_prob == 0:
                            # Skip this transition - don't draw edges from observation nodes
                            # of hidden action nodes
                            continue
                    
                    # Now m_start_idx and m_end_idx directly correspond to positions in memory_ids
                    start_mem_id = memory_ids[m_start_idx]
                    end_mem_id = memory_ids[m_end_idx]
                    
                    # Determine if this action is above or below the memory node
                    action_y = action_positions_dict[start_mem_id][a_idx][1]  # Get y-coordinate of action
                    memory_y = 0 if arrangement == 'horizontal' else ((num_nodes - 1) / 2 - memory_ids.index(start_mem_id)) * spacing  # Memory node y-coordinate
                    is_action_below = action_y < memory_y
                    
                    # Determine which observation node to start from
                    # Use the new naming scheme: O{mem_id}_A{action_idx}_{obs_idx}
                    obs_node = f"O{start_mem_id}_A{a_idx}_{y_idx}"
                    
                    # Target memory node
                    target_mem = f"M{end_mem_id}"
                    
                    # Add edge with weight
                    G.add_edge(obs_node, target_mem, weight=weight)
                    transition_edges.append((obs_node, target_mem))
                    
                    # Calculate line width based on weight
                    normalized_weight = weight / max_weight
                    line_width = min_line_width + (max_line_width - min_line_width) * normalized_weight
                    transition_weights_list.append(line_width)
    
    # Create the plot if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    # Separate nodes by type for different styling
    memory_nodes = [node for node in all_nodes if node.startswith('M')]
    action_nodes = [node for node in all_nodes if node.startswith('A')]
    obs_nodes = [node for node in all_nodes if node.startswith('O')]
    
    # Further separate nodes based on whether they have incoming connections
    if fade_no_incoming:
        memory_nodes_normal = [node for node in memory_nodes if int(node[1:]) not in nodes_with_no_incoming]
        memory_nodes_faded = [node for node in memory_nodes if int(node[1:]) in nodes_with_no_incoming]
        
        action_nodes_normal = [node for node in action_nodes if int(node.split('_')[0][1:]) not in nodes_with_no_incoming]
        action_nodes_faded = [node for node in action_nodes if int(node.split('_')[0][1:]) in nodes_with_no_incoming]
        
        obs_nodes_normal = [node for node in obs_nodes if int(node.split('_')[0][1:]) not in nodes_with_no_incoming]
        obs_nodes_faded = [node for node in obs_nodes if int(node.split('_')[0][1:]) in nodes_with_no_incoming]
    else:
        memory_nodes_normal = memory_nodes
        memory_nodes_faded = []
        action_nodes_normal = action_nodes
        action_nodes_faded = []
        obs_nodes_normal = obs_nodes
        obs_nodes_faded = []
    
    # Draw memory nodes (large circles) - normal alpha
    if memory_nodes_normal:
        nx.draw_networkx_nodes(G, pos, nodelist=memory_nodes_normal, 
                              node_color=memory_node_color, node_size=mem_node_size, 
                              node_shape='o', edgecolors='black', linewidths=2, alpha=1.0,
                              ax=ax)
    
    # Draw memory nodes (large circles) - faded alpha
    if memory_nodes_faded:
        nx.draw_networkx_nodes(G, pos, nodelist=memory_nodes_faded, 
                              node_color=memory_node_color, node_size=mem_node_size, 
                              node_shape='o', edgecolors='black', linewidths=2, alpha=no_incoming_alpha,
                              ax=ax)
    
    # Draw action nodes (squares) - normal alpha
    if action_nodes_normal:
        nx.draw_networkx_nodes(G, pos, nodelist=action_nodes_normal, 
                              node_color=action_node_color, node_size=action_node_size, 
                              node_shape='s', edgecolors='black', linewidths=2, alpha=1.0,
                              ax=ax)
    
    # Draw action nodes (squares) - faded alpha
    if action_nodes_faded:
        nx.draw_networkx_nodes(G, pos, nodelist=action_nodes_faded, 
                              node_color=action_node_color, node_size=action_node_size, 
                              node_shape='s', edgecolors='black', linewidths=2, alpha=no_incoming_alpha,
                              ax=ax)
    
    # Draw observation nodes (diamonds) - separated by observation type for different colors
    for obs_idx in range(Y):
        # Get all observation nodes for this observation type
        obs_nodes_this_type_normal = [node for node in obs_nodes_normal if node.endswith(f'_{obs_idx}')]
        obs_nodes_this_type_faded = [node for node in obs_nodes_faded if node.endswith(f'_{obs_idx}')]
        
        # Draw observation nodes for this type - normal alpha
        if obs_nodes_this_type_normal:
            nx.draw_networkx_nodes(G, pos, nodelist=obs_nodes_this_type_normal, 
                                  node_color=observation_node_colors[obs_idx], node_size=obs_node_size, 
                                  node_shape='D', edgecolors='black', linewidths=2, alpha=1.0,
                                  ax=ax)
        
        # Draw observation nodes for this type - faded alpha
        if obs_nodes_this_type_faded:
            nx.draw_networkx_nodes(G, pos, nodelist=obs_nodes_this_type_faded, 
                                  node_color=observation_node_colors[obs_idx], node_size=obs_node_size, 
                                  node_shape='D', edgecolors='black', linewidths=2, alpha=no_incoming_alpha,
                                  ax=ax)
    
    # Draw internal edges (within complex nodes) - separate action and observation edges
    internal_edges = [(u, v) for u, v in G.edges() if (u, v) not in transition_edges]
    
    # Separate action edges (memory to action) from observation edges (action to observation)
    action_edges = [(u, v) for u, v in internal_edges if u.startswith('M') and v.startswith('A')]
    observation_edges = [(u, v) for u, v in internal_edges if u.startswith('A') and v.startswith('O')]
    
    # Further separate edges by alpha level if fading is enabled
    if fade_no_incoming:
        action_edges_normal = [(u, v) for u, v in action_edges if int(u[1:]) not in nodes_with_no_incoming]
        action_edges_faded = [(u, v) for u, v in action_edges if int(u[1:]) in nodes_with_no_incoming]
        
        observation_edges_normal = [(u, v) for u, v in observation_edges if int(u.split('_')[0][1:]) not in nodes_with_no_incoming]
        observation_edges_faded = [(u, v) for u, v in observation_edges if int(u.split('_')[0][1:]) in nodes_with_no_incoming]
    else:
        action_edges_normal = action_edges
        action_edges_faded = []
        observation_edges_normal = observation_edges
        observation_edges_faded = []
    
    # Draw action edges with thickness based on action probabilities
    if action_probabilities_permuted is not None:
        # Define minimum width threshold for drawing (to avoid zero-width edges with visible arrowheads)
        min_draw_width = 1e-6
        
        # Normal alpha action edges
        if action_edges_normal:
            action_edges_to_draw_normal = []
            action_widths_normal = []
            for mem_node, action_node in action_edges_normal:
                mem_id = int(mem_node[1:])
                mem_idx = memory_ids.index(mem_id)
                # Extract action index from node name A{mem_id}_{action_idx}
                action_idx = int(action_node.split('_')[1])
                prob = action_probabilities_permuted[mem_idx, action_idx]
                width = min_action_width + (max_action_width - min_action_width) * prob
                
                # Only include edges with visible width
                if width > min_draw_width:
                    action_edges_to_draw_normal.append((mem_node, action_node))
                    action_widths_normal.append(width)
            
            if action_edges_to_draw_normal:
                nx.draw_networkx_edges(G, pos, edgelist=action_edges_to_draw_normal, edge_color=action_edge_color, 
                                      arrows=True, arrowstyle='-|>', arrowsize=15, 
                                      width=action_widths_normal, min_target_margin=arrowhead_distance_actions, alpha=0.8,
                                      ax=ax)
        
        # Faded alpha action edges
        if action_edges_faded:
            action_edges_to_draw_faded = []
            action_widths_faded = []
            for mem_node, action_node in action_edges_faded:
                mem_id = int(mem_node[1:])
                mem_idx = memory_ids.index(mem_id)
                # Extract action index from node name A{mem_id}_{action_idx}
                action_idx = int(action_node.split('_')[1])
                prob = action_probabilities_permuted[mem_idx, action_idx]
                width = min_action_width + (max_action_width - min_action_width) * prob
                
                # Only include edges with visible width
                if width > min_draw_width:
                    action_edges_to_draw_faded.append((mem_node, action_node))
                    action_widths_faded.append(width)
            
            if action_edges_to_draw_faded:
                nx.draw_networkx_edges(G, pos, edgelist=action_edges_to_draw_faded, edge_color=action_edge_color, 
                                      arrows=True, arrowstyle='-|>', arrowsize=15, 
                                      width=action_widths_faded, min_target_margin=arrowhead_distance_actions, alpha=no_incoming_alpha,
                                      ax=ax)
    else:
        # Default thickness if no probabilities provided
        if action_edges_normal:
            nx.draw_networkx_edges(G, pos, edgelist=action_edges_normal, edge_color=action_edge_color, 
                                  arrows=True, arrowstyle='-|>', arrowsize=15, width=1,
                                  min_target_margin=arrowhead_distance_actions, alpha=0.8,
                                  ax=ax)
        if action_edges_faded:
            nx.draw_networkx_edges(G, pos, edgelist=action_edges_faded, edge_color=action_edge_color, 
                                  arrows=True, arrowstyle='-|>', arrowsize=15, width=1,
                                  min_target_margin=arrowhead_distance_actions, alpha=no_incoming_alpha,
                                  ax=ax)

    # Draw observation edges (action to observation) - standard thickness, no arrows
    if observation_edges_normal:
        nx.draw_networkx_edges(G, pos, edgelist=observation_edges_normal, edge_color='gray', 
                              arrows=False, width=1, alpha=1.0, ax = ax)
    if observation_edges_faded:
        nx.draw_networkx_edges(G, pos, edgelist=observation_edges_faded, edge_color='gray', 
                              arrows=False, width=1, alpha=no_incoming_alpha, ax = ax)
    
    # Draw transition edges (between complex nodes) - thicker, colored by weight
    if transition_edges:
        # Define minimum width threshold for drawing (to avoid zero-width edges with visible arrowheads)
        min_draw_width = 1e-6
        
        if fade_no_incoming:
            # Separate transition edges by source node alpha level
            transition_edges_normal = []
            transition_weights_normal = []
            transition_edges_faded = []
            transition_weights_faded = []
            
            for i, (obs_node, target_mem) in enumerate(transition_edges):
                # Only include edges with visible width
                if transition_weights_list[i] > min_draw_width:
                    # Extract source memory ID from observation node name O{mem_id}_A{action_idx}_{L/R}
                    source_mem_id = int(obs_node.split('_')[0][1:])
                    if source_mem_id in nodes_with_no_incoming:
                        transition_edges_faded.append((obs_node, target_mem))
                        transition_weights_faded.append(transition_weights_list[i])
                    else:
                        transition_edges_normal.append((obs_node, target_mem))
                        transition_weights_normal.append(transition_weights_list[i])
            
            # Draw normal alpha transition edges
            if transition_edges_normal:
                # Draw transition edges for each observation type (general for any Y)
                for obs_idx in range(Y):
                    transition_edges_obs = []
                    transition_weights_obs = []
                    
                    for i, (obs_node, target_mem) in enumerate(transition_edges_normal):
                        # Extract observation index from node name O{mem_id}_A{action_idx}_{obs_idx}
                        node_obs_idx = int(obs_node.split('_')[-1])
                        
                        if node_obs_idx == obs_idx:
                            transition_edges_obs.append((obs_node, target_mem))
                            transition_weights_obs.append(transition_weights_normal[i])
                    
                    if transition_edges_obs:
                        nx.draw_networkx_edges(
                            G, pos, edgelist=transition_edges_obs,
                            edge_color=observation_node_colors[obs_idx], arrows=True, arrowstyle='-|>',
                            arrowsize=20, width=transition_weights_obs,
                            min_target_margin=arrowhead_distance, alpha=0.7, ax = ax
                        )

            # Draw faded alpha transition edges
            if transition_edges_faded:
                for obs_idx in range(Y):
                    transition_edges_obs = []
                    transition_weights_obs = []
                    
                    for i, (obs_node, target_mem) in enumerate(transition_edges_faded):
                        # Extract observation index from node name O{mem_id}_A{action_idx}_{obs_idx}
                        node_obs_idx = int(obs_node.split('_')[-1])
                        
                        if node_obs_idx == obs_idx:
                            transition_edges_obs.append((obs_node, target_mem))
                            transition_weights_obs.append(transition_weights_faded[i])
                    
                    if transition_edges_obs:
                        nx.draw_networkx_edges(
                            G, pos, edgelist=transition_edges_obs,
                            edge_color=observation_node_colors[obs_idx], arrows=True, arrowstyle='-|>',
                            arrowsize=20, width=transition_weights_obs,
                            min_target_margin=arrowhead_distance, alpha=no_incoming_alpha, ax = ax
                        )
        else:
            # Filter out zero-width edges and draw all transition edges with normal alpha
            visible_edges_by_obs = [[] for _ in range(Y)]
            visible_weights_by_obs = [[] for _ in range(Y)]
            for i, (obs_node, target_mem) in enumerate(transition_edges):
                if transition_weights_list[i] > min_draw_width:
                    # Extract observation index from node name O{mem_id}_A{action_idx}_{obs_idx}
                    obs_idx = int(obs_node.split('_')[-1])
                    visible_edges_by_obs[obs_idx].append((obs_node, target_mem))
                    visible_weights_by_obs[obs_idx].append(transition_weights_list[i])
            # Draw each group with the correct color
            for obs_idx in range(Y):
                if visible_edges_by_obs[obs_idx]:
                    nx.draw_networkx_edges(
                        G, pos, edgelist=visible_edges_by_obs[obs_idx],
                        edge_color=observation_node_colors[obs_idx], arrows=True, arrowstyle='-|>',
                        arrowsize=20, width=visible_weights_by_obs[obs_idx],
                        min_target_margin=arrowhead_distance, alpha=0.7, ax=ax
                    )
    
    # Create labels
    labels = {}
    for i, mem_id in enumerate(memory_ids):
        labels[f"M{mem_id}"] = memory_names[i]
        
        # Add labels for all action nodes
        for a_idx in range(A):
            # Check if this action node was hidden due to zero probability
            if hide_unused_actions and action_probabilities_permuted is not None:
                action_prob = action_probabilities_permuted[i, a_idx]
                if action_prob == 0:
                    continue  # Skip labels for hidden action nodes
            
            action_node = f"A{mem_id}_{a_idx}"
            labels[action_node] = action_names[a_idx] if a_idx < len(action_names) else f"A{a_idx}"
            
            # Add labels for observation nodes
            for y_idx in range(Y):
                # Only add label if this observation is allowed from this action
                if AllowedObsFromAct is None or AllowedObsFromAct[a_idx, y_idx]:
                    obs_node = f"O{mem_id}_A{a_idx}_{y_idx}"
                    labels[obs_node] = observation_names[y_idx] if y_idx < len(observation_names) else f"O{y_idx}"
    
    # Draw labels with different alpha for faded nodes
    if fade_no_incoming and nodes_with_no_incoming:
        # Separate labels for normal and faded nodes
        labels_normal = {}
        labels_faded = {}
        
        for node_key, label_text in labels.items():
            # Extract memory ID from node key
            if node_key.startswith('M'):
                mem_id = int(node_key[1:])
            elif node_key.startswith('A'):
                mem_id = int(node_key.split('_')[0][1:])
            elif node_key.startswith('O'):
                mem_id = int(node_key.split('_')[0][1:])
            
            if mem_id in nodes_with_no_incoming:
                labels_faded[node_key] = label_text
            else:
                labels_normal[node_key] = label_text
        
        # Draw normal labels
        if labels_normal:
            # Dynamically set font size based on node type and node size
            # Use a mapping: memory nodes -> mem_node_size, action nodes -> action_node_size, obs nodes -> obs_node_size
            for node_key, label_text in labels_normal.items():
                if node_key.startswith('M'):
                    node_size = mem_node_size
                elif node_key.startswith('A'):
                    node_size = action_node_size
                elif node_key.startswith('O'):
                    node_size = obs_node_size
                else:
                    node_size = 500  # fallback

                # Heuristic: font size proportional to sqrt(node_size)
                font_size = max(8, min(18, int(0.18 * node_size**0.69)))
                nx.draw_networkx_labels(G, pos, {node_key: label_text}, font_size=font_size, font_weight='bold', alpha=1.0, ax = ax)
        
        # Draw faded labels
        if labels_faded:
            # Dynamically set font size based on node type and node size
            for node_key, label_text in labels_faded.items():
                if node_key.startswith('M'):
                    node_size = mem_node_size
                elif node_key.startswith('A'):
                    node_size = action_node_size
                elif node_key.startswith('O'):
                    node_size = obs_node_size
                else:
                    node_size = 500

                # Heuristic: font size proportional to sqrt(node_size)
                font_size = max(8, min(18, int(0.18 * node_size**0.69)))
                nx.draw_networkx_labels(G, pos, {node_key: label_text}, font_size=font_size, font_weight='bold', alpha=no_incoming_alpha, ax = ax)
    else:
        # Draw all labels with normal alpha
        # Dynamically set font size based on node type and node size
        for node_key, label_text in labels.items():
            if node_key.startswith('M'):
                node_size = mem_node_size
            elif node_key.startswith('A'):
                node_size = action_node_size
            elif node_key.startswith('O'):
                node_size = obs_node_size
            else:
                node_size = 500

            # Heuristic: font size proportional to sqrt(node_size)
            font_size = max(8, min(18, int(0.18 * node_size**0.69)))
            nx.draw_networkx_labels(G, pos, {node_key: label_text}, font_size=font_size, font_weight='bold', alpha=1.0, ax = ax)
        #nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    # Customize plot
    arrangement_str = arrangement.capitalize()
    ax.axis('off')

    if ax is None:
        plt.tight_layout()
    
    if plot_legend:

        # Use the actual colors from the function arguments
        legend_elements = [
            Patch(facecolor=memory_node_color, edgecolor='black', label='Memory Node'),
            Patch(facecolor=action_node_color, edgecolor='black', label='Action Node'),
        ]

        # Add observation node colors to legend
        for i, color in enumerate(observation_node_colors):
            label = f'Observation Node {observation_names[i]}' if observation_names is not None else f'Observation Node {i}'
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))

        legend_elements.extend([
            Line2D([0], [0], color=action_edge_color, lw=2, alpha=0.8, label='Action Probabilities'),
            Line2D([0], [0], color='gray', lw=1, label='Observation Connections'),
        ])

        # For transition edges, use the observation node colors if colored, else fallback to 'red'
        if transition_edges:
            if fade_no_incoming or (observation_node_colors is not None and len(observation_node_colors) > 0):
                for i, color in enumerate(observation_node_colors):
                    label = f'Transition (Obs {observation_names[i]})' if observation_names is not None else f'Transition (Obs {i})'
                    legend_elements.append(Line2D([0], [0], color=color, lw=3, alpha=0.7, label=label))
            else:
                legend_elements.append(Line2D([0], [0], color='red', lw=3, alpha=0.7, label='Memory Transitions'))
        
        if arrangement == 'horizontal':
            legend_loc = 'upper right'
            bbox_anchor = (1.1, 1)
        else:
            legend_loc = 'center right'
            bbox_anchor = (1.2, 0.5)
        
        ax.legend(handles=legend_elements, loc=legend_loc, bbox_to_anchor=bbox_anchor)
    
    return G, ax


def draw_bipartite_FSC_network(transition_weights, action_probabilities=None, memory_ids=None, arrangement='horizontal', 
                               spacing=6, layer_spacing=6, fig_size=None, min_weightsum_threshold=0.01,
                               mem_node_size=2000, action_node_size=1200, obs_node_size=500,
                               action_node_yoffset=1.5, action_node_xoffset=1.5,
                               obs_node_xoffset=0.7, obs_node_yoffset=0.7,
                               max_line_width=1, min_line_width=0.,
                               max_action_width=3, min_action_width=0,
                               memory_names=None, action_names=None, observation_names=None,
                               suppress_zero_action_transitions=False, action_prob_threshold=1e-10,
                               fade_no_incoming=False, no_incoming_alpha=0.3,
                               arrowhead_distance = 15, arrowhead_distance_actions = 15,
                               action_edge_color='blue', memory_node_color='lightblue', 
                               action_node_color='lightgreen', observation_node_colors=None,
                               plot_legend = False, reverse_obs_below=True, hide_unused_actions=False,
                               AllowedObsFromAct=None, obs_rotation=0.0,
                               ax = None):
    """
    Draw multiple complex node structures arranged in two bipartite layers with weighted transitions between observation and memory nodes.
    
    This function is identical to draw_complex_network_with_transitions except that the memory nodes
    are arranged in two linear chains (layers), one on top of the other. Assumes M is even, with
    M//2 nodes per layer.
    
    Parameters:
    -----------
    transition_weights : numpy.ndarray
        Shape (Y, A, M, M) where:
        - Y: observation index
        - A: action index  
        - M: starting memory node index
        - M: ending memory node index
    action_probabilities : numpy.ndarray or None
        Shape (M, A) - probability of taking action A from memory state M
        Controls thickness of edges from memory nodes to action nodes
    memory_ids : list or None
        List of memory node IDs that determines the spatial order of complex nodes.
        This completely rearranges both the visual layout AND the underlying data.
        For example, memory_ids=[1,0,2] will swap the first two memory nodes including
        all their transition probabilities and action probabilities.
        Must contain exactly the values [0, 1, ..., M-1] in any order.
        If None, uses [0, 1, 2, ..., M-1]
    arrangement : str
        'horizontal' or 'vertical' arrangement
    spacing : float
        Distance between adjacent complex nodes within each layer
    layer_spacing : float
        Distance between the two bipartite layers. Default: 6
    fig_size : tuple or None
        Figure size (width, height). If None, automatically calculated
    min_weightsum_threshold : float
        Minimum weight sum for a memory node not to be faded (if fade_no_incoming=True)
    mem_node_size : float
        Maximum size for memory nodes (circles)
    min_line_width : float
        Minimum line width for weakest transition connections
    max_action_width : float
        Maximum line width for action connections (probability = 1.0)
    min_action_width : float
        Minimum line width for action connections (probability = 0.0)
    memory_names : list or None
        Custom names for memory nodes. If None, uses default 'M{id}' format
    action_names : list or None
        Custom names for action nodes. If None, uses default 'A0', 'A1', etc.
    observation_names : list or None
        Custom names for observation nodes. If None, uses default 'O0', 'O1', etc.
    suppress_zero_action_transitions : bool
        If True, suppress transition edges from observation nodes when the corresponding
        memory-to-action probability is below action_prob_threshold. Default: False
    action_prob_threshold : float
        Threshold below which action probabilities are considered zero for suppression.
        Only used when suppress_zero_action_transitions=True. Default: 1e-10
    fade_no_incoming : bool
        If True, apply reduced alpha to complex nodes that have no incoming connections
        from any other memory node. Default: False
    no_incoming_alpha : float
        Alpha (transparency) value for complex nodes with no incoming connections.
        Only used when fade_no_incoming=True. Range: 0.0 (fully transparent) to 1.0 (opaque).
        Default: 0.3
    arrowhead_distance : float
        Distance (in points) between arrowheads and target nodes. Larger values create
        more space between the arrowhead tip and the node edge. This affects all edges
        with arrows (action edges and transition edges). Default: 15
    action_edge_color : str
        Color for action probability edges (memory to action nodes). Default: 'blue'
    memory_node_color : str
        Color for memory nodes (large circles). Default: 'lightblue'
    action_node_color : str
        Color for action nodes (squares). Default: 'lightgreen'
    observation_node_colors : list or None
        List of colors for observation nodes (diamonds). Must have exactly Y elements
        if provided. Default: None (uses ['lightcoral', 'lightsalmon'] for Y=2)
    plot_legend : bool
        Whether to display a legend explaining the different node types and edge colors.
        Default: False
    reverse_obs_below : bool
        Whether to reverse the order of observation nodes for action nodes positioned
        below the memory node. When True (default), observation nodes are mirrored:
        - For Y=2: Top actions: left obs = obs[0], right obs = obs[1]
                   Bottom actions: left obs = obs[1], right obs = obs[0]
        - For Y>2: Top actions: circular arrangement starting from right (0°)
                   Bottom actions: mirrored circular arrangement (angles negated)
        This creates a symmetric layout around the memory node. Default: True
    hide_unused_actions : bool
        Whether to hide action nodes and their associated observation nodes when
        action_probabilities[m, a] = 0. When True, action nodes with zero probability
        will not be drawn, along with their observation nodes and edges. This requires
        action_probabilities to be provided. Default: False
    AllowedObsFromAct : numpy.ndarray or None
        Shape (A, Y) boolean array specifying which observations are allowed from each action.
        If AllowedObsFromAct[a, y] is False, the observation node O_y for action a will not
        be drawn and all transition weights from that observation will be set to zero.
        If None, all observations are allowed from all actions. Default: None
    obs_rotation : float
        Rotation angle (in radians) to apply to observation node positions when using
        circular arrangement (for Y > 2). Positive values rotate counter-clockwise.
        For example, π/4 (≈0.785) rotates by 45 degrees. Default: 0.0 (no rotation)
    ax : matplotlib.axes.Axes or None
        Matplotlib Axes object to draw on. If None, a new figure and axes are created.
        Default: None
    """
    Y, A, M, M_end = transition_weights.shape
    assert M == M_end, "Transition matrix must be square in memory dimensions"
    assert M % 2 == 0, "M must be even for bipartite arrangement"
    
    # Validate action probabilities if provided
    if action_probabilities is not None:
        assert action_probabilities.shape == (M, A), f"Action probabilities must be shape (M, A) = ({M}, {A}), got {action_probabilities.shape}"
    
    # Validate AllowedObsFromAct if provided
    if AllowedObsFromAct is not None:
        assert isinstance(AllowedObsFromAct, np.ndarray), "AllowedObsFromAct must be a numpy array"
        assert AllowedObsFromAct.dtype == bool, "AllowedObsFromAct must be a boolean array"
        assert AllowedObsFromAct.shape == (A, Y), f"AllowedObsFromAct must be shape (A, Y) = ({A}, {Y}), got {AllowedObsFromAct.shape}"
    
    num_nodes = M
    nodes_per_layer = M // 2
    
    # Set default memory IDs if not provided
    if memory_ids is None:
        memory_ids = list(range(num_nodes))
    elif len(memory_ids) != num_nodes:
        raise ValueError(f"Length of memory_ids ({len(memory_ids)}) must match M ({M})")
    
    # Validate that memory_ids contains unique values in range [0, M-1]
    if set(memory_ids) != set(range(M)):
        raise ValueError(f"memory_ids must contain exactly the values [0, 1, ..., {M-1}], got {memory_ids}")
    
    # Create permutation arrays to rearrange data according to memory_ids
    # memory_ids[i] tells us which original memory should be at position i
    perm_indices = memory_ids  # This is the permutation for rows/columns
    
    # Rearrange transition weights: (Y, A, M, M) -> permute both M dimensions
    transition_weights_permuted = transition_weights[:, :, perm_indices, :][:, :, :, perm_indices]
    
    # Rearrange action probabilities: (M, A) -> permute M dimension
    if action_probabilities is not None:
        action_probabilities_permuted = action_probabilities[perm_indices, :]
    else:
        action_probabilities_permuted = None
    
    # Identify memory nodes with no incoming connections (if fade_no_incoming is enabled)
    nodes_with_no_incoming = set()
    if fade_no_incoming:
        # Iteratively identify nodes that should be faded
        # Start with nodes that have no incoming connections from any other node
        # Then add nodes that only receive connections from already-faded nodes
        
        # Compute weighted incoming connections for each memory node
        transition_weights_no_self = transition_weights_permuted.copy()
        
        # Set diagonal elements to zero to exclude self-connections
        for m in range(M):
            transition_weights_no_self[:, :, m, m] = 0
        
        # Create a matrix of weighted connections between memory nodes
        # Shape: (M_source, M_target) - weighted sum of connections from source to target
        memory_to_memory_weights = np.zeros((M, M))
        
        for y_idx in range(Y):
            for a_idx in range(A):
                for m_source in range(M):
                    for m_target in range(M):
                        if m_source != m_target:  # Exclude self-connections
                            weight = transition_weights_no_self[y_idx, a_idx, m_source, m_target]
                            if action_probabilities_permuted is not None:
                                action_prob = action_probabilities_permuted[m_source, a_idx]
                                weight *= action_prob
                            memory_to_memory_weights[m_source, m_target] += weight
        
        # Iteratively identify nodes to fade
        faded_nodes = set()
        changed = True
        
        while changed:
            changed = False
            for mem_idx in range(M):
                mem_id = memory_ids[mem_idx]
                
                if mem_id not in faded_nodes:
                    # Check if this node should be faded
                    # Sum incoming connections from non-faded nodes
                    incoming_from_active = 0
                    for source_idx in range(M):
                        source_id = memory_ids[source_idx]
                        if source_id not in faded_nodes:
                            incoming_from_active += memory_to_memory_weights[source_idx, mem_idx]
                    
                    # If no significant incoming connections from non-faded nodes, fade this node
                    if incoming_from_active <= min_weightsum_threshold:
                        faded_nodes.add(mem_id)
                        changed = True
        
        nodes_with_no_incoming = faded_nodes
        #print(f"Memory nodes to be faded (no incoming connections from active nodes): {sorted(nodes_with_no_incoming)}")
    
    # Set default names if not provided
    if memory_names is None:
        memory_names = [f'M{mem_id}' for mem_id in memory_ids]
    elif len(memory_names) != num_nodes:
        raise ValueError(f"Length of memory_names ({len(memory_names)}) must match M ({M})")
        
    if action_names is None:
        action_names = [f'A{i}' for i in range(A)]
    elif len(action_names) != A:
        raise ValueError(f"Length of action_names ({len(action_names)}) must match A ({A})")
        
    if observation_names is None:
        observation_names = [f'O{i}' for i in range(Y)]
    elif len(observation_names) != Y:
        raise ValueError(f"Length of observation_names ({len(observation_names)}) must match Y ({Y})")
    
    # Set default observation node colors if not provided
    if observation_node_colors is None:
        if Y == 2:
            observation_node_colors = ['lightcoral', 'lightsalmon']
        else:
            # For other Y values, create a default color palette
            default_colors = ['lightcoral', 'lightsalmon', 'lightpink', 'lightsteelblue', 'lightseagreen']
            observation_node_colors = (default_colors * ((Y // len(default_colors)) + 1))[:Y]
    elif len(observation_node_colors) != Y:
        raise ValueError(f"Length of observation_node_colors ({len(observation_node_colors)}) must match Y ({Y})")
    
    # Calculate figure size if not provided
    if fig_size is None:
        if arrangement == 'horizontal':
            fig_size = (6 * nodes_per_layer, 10)  # Adjusted for bipartite layout
        else:  # vertical
            fig_size = (12, 4 * nodes_per_layer)  # Adjusted for bipartite layout
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Store all node names and positions
    all_nodes = []
    pos = {}
    
    # Store action positions for each memory node for later use in transition edge creation
    action_positions_dict = {}
    
    # Generate nodes and positions for each complex node in bipartite arrangement
    for m_idx, mem_id in enumerate(memory_ids):
        # Determine which layer (top or bottom) and position within layer
        layer = m_idx // nodes_per_layer  # 0 for top layer, 1 for bottom layer
        pos_in_layer = m_idx % nodes_per_layer

        # Calculate offset for this complex node in bipartite arrangement
        if arrangement == 'horizontal':
            x_offset = (pos_in_layer - (nodes_per_layer - 1) / 2) * spacing
            y_offset = layer_spacing/2 * (1 - 2*layer)  # Top layer at +layer_spacing/2, bottom layer at -layer_spacing/2
        else:  # vertical
            x_offset = layer_spacing/2 * (1 - 2*layer)  # Left layer at -layer_spacing/2, right layer at +layer_spacing/2
            y_offset = ((nodes_per_layer - 1) / 2 - pos_in_layer) * spacing
        
        # Define node IDs for this complex node
        center_mem = f"M{mem_id}"
        
        # Calculate action node positions based on number of actions
        if A == 2:
            # Original layout: one above, one below
            action_positions = [
                (x_offset, y_offset + action_node_yoffset),  # top
                (x_offset, y_offset - action_node_yoffset)   # bottom
            ]
        elif A == 3:
            # One above, two below
            action_positions = [
                (x_offset, y_offset + action_node_yoffset),                    # top
                (x_offset - action_node_xoffset/2, y_offset - action_node_yoffset),  # bottom left
                (x_offset + action_node_xoffset/2, y_offset - action_node_yoffset)   # bottom right
            ]
        elif A == 4:
            # Two above, two below
            action_positions = [
                (x_offset - action_node_xoffset/2, y_offset + action_node_yoffset),  # top left
                (x_offset + action_node_xoffset/2, y_offset + action_node_yoffset),  # top right
                (x_offset - action_node_xoffset/2, y_offset - action_node_yoffset),  # bottom left
                (x_offset + action_node_xoffset/2, y_offset - action_node_yoffset)   # bottom right
            ]
        else:
            # General case: distribute actions in rows
            # For A actions, put ceil(A/2) on top, floor(A/2) on bottom
            top_actions = (A + 1) // 2
            bottom_actions = A // 2
            
            action_positions = []
            
            # Top row actions
            if top_actions == 1:
                action_positions.append((x_offset, y_offset + action_node_yoffset))
            else:
                for j in range(top_actions):
                    x_pos = x_offset + (j - (top_actions - 1) / 2) * action_node_xoffset
                    action_positions.append((x_pos, y_offset + action_node_yoffset))
            
            # Bottom row actions
            if bottom_actions == 1:
                action_positions.append((x_offset, y_offset - action_node_yoffset))
            else:
                for j in range(bottom_actions):
                    x_pos = x_offset + (j - (bottom_actions - 1) / 2) * action_node_xoffset
                    action_positions.append((x_pos, y_offset - action_node_yoffset))
        
        # Store action positions for this memory node
        action_positions_dict[mem_id] = action_positions
        
        # Create action nodes and their observation nodes
        complex_nodes = [center_mem]
        action_nodes_this_mem = []
        
        for a_idx in range(A):
            # Check if we should hide this action node due to zero probability
            if hide_unused_actions and action_probabilities_permuted is not None:
                action_prob = action_probabilities_permuted[m_idx, a_idx]  # i is the index in memory_ids
                if action_prob == 0:
                    continue  # Skip this action node and its observation nodes
            
            action_node = f"A{mem_id}_{a_idx}"
            action_nodes_this_mem.append(action_node)
            complex_nodes.append(action_node)
            
            # Create observation nodes for this action
            obs_nodes_this_action = []
            for y_idx in range(Y):
                # Only create observation node if this observation is allowed from this action
                if AllowedObsFromAct is None or AllowedObsFromAct[a_idx, y_idx]:
                    obs_node = f"O{mem_id}_A{a_idx}_{y_idx}"
                    obs_nodes_this_action.append(obs_node)
                    complex_nodes.append(obs_node)
                    
                    # Add internal edge from action to observation
                    G.add_edge(action_node, obs_node)
            
            # Add internal edges
            G.add_edge(center_mem, action_node)
        
        all_nodes.extend(complex_nodes)
        
        # Set positions
        pos[center_mem] = (x_offset, y_offset)
        for a_idx in range(A):
            # Check if this action node was skipped due to zero probability
            if hide_unused_actions and action_probabilities_permuted is not None:
                action_prob = action_probabilities_permuted[m_idx, a_idx]
                if action_prob == 0:
                    continue  # Skip positioning for this action node
            
            action_node = f"A{mem_id}_{a_idx}"
            pos[action_node] = action_positions[a_idx]
            
            # Position observation nodes relative to action node
            action_x, action_y = action_positions[a_idx]
            
            # Determine if this action is above or below the memory node
            is_action_below = action_y < y_offset
            
            # Get list of allowed observations for this action
            if AllowedObsFromAct is None:
                allowed_obs_indices = list(range(Y))
            else:
                allowed_obs_indices = [y_idx for y_idx in range(Y) if AllowedObsFromAct[a_idx, y_idx]]
            
            # Calculate observation node positions based on number of allowed observations
            Y_allowed = len(allowed_obs_indices)
            
            if Y_allowed == 0:
                # No observations allowed for this action - skip positioning
                continue
            elif Y_allowed == 1:
                # Single observation - place directly at action position with small offset
                obs_positions = [(action_x, action_y + obs_node_yoffset * 0.3)]
            elif Y_allowed == 2:
                # For 2 observations, use left/right arrangement
                if reverse_obs_below and is_action_below:
                    # For actions below, reverse the observation order
                    obs_positions = [
                        (action_x + obs_node_xoffset, action_y),  # first allowed obs -> right position
                        (action_x - obs_node_xoffset, action_y)   # second allowed obs -> left position
                    ]
                else:
                    # For actions above (or when reverse_obs_below is False), use normal order
                    obs_positions = [
                        (action_x - obs_node_xoffset, action_y),  # first allowed obs -> left position
                        (action_x + obs_node_xoffset, action_y)   # second allowed obs -> right position
                    ]
            else:
                # For more than 2 observations, arrange in a pattern around the action node
                obs_positions = []
                
                if Y_allowed == 3:
                    # Special case for 3 observations: left, right, top/bottom
                    base_positions = [
                        (action_x - obs_node_xoffset, action_y),  # left
                        (action_x + obs_node_xoffset, action_y),  # right
                        (action_x, action_y + obs_node_yoffset),   # top
                        (action_x, action_y - obs_node_yoffset)   # bottom
                    ]
                    
                    # Select three positions
                    if reverse_obs_below and is_action_below:
                        # For actions below, mirror horizontally and use bottom instead of top
                        obs_positions = [
                            base_positions[1],  # right position
                            base_positions[0],  # left position
                            base_positions[3]   # bottom position
                        ]
                    else:
                        obs_positions = [
                            base_positions[0],  # left position
                            base_positions[1],  # right position
                            base_positions[2]   # top position
                        ]
                        
                else:
                    # For Y_allowed > 3, use circular arrangement
                    for i in range(Y_allowed):
                        # Calculate angle for this observation (starting from right, going counter-clockwise)
                        # Apply custom rotation offset
                        angle = 2 * np.pi * i / Y_allowed + obs_rotation
                        
                        # Apply reverse_obs_below logic for actions below the memory node
                        if reverse_obs_below and is_action_below:
                            # For actions below, mirror the angles horizontally
                            angle = -angle
                        
                        # Calculate position
                        obs_x = action_x + obs_node_xoffset * np.cos(angle)
                        obs_y = action_y + obs_node_xoffset * np.sin(angle)
                        obs_positions.append((obs_x, obs_y))
            
            # Set positions for allowed observation nodes only
            for i, y_idx in enumerate(allowed_obs_indices):
                if i < len(obs_positions):  # Safety check
                    obs_node = f"O{mem_id}_A{a_idx}_{y_idx}"
                    pos[obs_node] = obs_positions[i]
    
    # Add all nodes to the graph
    G.add_nodes_from(all_nodes)
    
    # Add weighted transition edges between observation and memory nodes
    transition_edges = []
    transition_weights_list = []
    
    # Find maximum weight for normalization
    max_weight = np.max(transition_weights_permuted)
    
    for y_idx in range(Y):  # observation index
        for a_idx in range(A):  # action index
            for m_start_idx in range(M):  # starting memory index (now corresponds to display order)
                for m_end_idx in range(M):  # ending memory index (now corresponds to display order)
                    weight = transition_weights_permuted[y_idx, a_idx, m_start_idx, m_end_idx]
                    
                    # Set weight to zero if this observation is not allowed from this action
                    if AllowedObsFromAct is not None and not AllowedObsFromAct[a_idx, y_idx]:
                        weight = 0
                    
                    # Check if we should suppress this transition due to zero action probability
                    if suppress_zero_action_transitions and action_probabilities_permuted is not None:
                        action_prob = action_probabilities_permuted[m_start_idx, a_idx]
                        if action_prob <= action_prob_threshold:
                            # Skip this transition - don't draw edges from observation nodes
                            # connected to actions with zero probability
                            continue
                    
                    # Check if this action node was hidden due to zero probability
                    if hide_unused_actions and action_probabilities_permuted is not None:
                        action_prob = action_probabilities_permuted[m_start_idx, a_idx]
                        if action_prob == 0:
                            # Skip this transition - don't draw edges from observation nodes
                            # of hidden action nodes
                            continue
                    
                    # Now m_start_idx and m_end_idx directly correspond to positions in memory_ids
                    start_mem_id = memory_ids[m_start_idx]
                    end_mem_id = memory_ids[m_end_idx]
                    
                    # Determine if this action is above or below the memory node
                    action_y = action_positions_dict[start_mem_id][a_idx][1]  # Get y-coordinate of action
                    # For bipartite arrangement, calculate memory node y-coordinate
                    start_layer = m_start_idx // nodes_per_layer
                    if arrangement == 'horizontal':
                        memory_y = layer_spacing/2 * (1 - 2*start_layer)  # Top layer at +layer_spacing/2, bottom layer at -layer_spacing/2
                    else:
                        memory_y = ((nodes_per_layer - 1) / 2 - (m_start_idx % nodes_per_layer)) * spacing
                    is_action_below = action_y < memory_y
                    
                    # Determine which observation node to start from
                    # Use the new naming scheme: O{mem_id}_A{action_idx}_{obs_idx}
                    obs_node = f"O{start_mem_id}_A{a_idx}_{y_idx}"
                    
                    # Target memory node
                    target_mem = f"M{end_mem_id}"
                    
                    # Add edge with weight
                    G.add_edge(obs_node, target_mem, weight=weight)
                    transition_edges.append((obs_node, target_mem))
                    
                    # Calculate line width based on weight
                    normalized_weight = weight / max_weight
                    line_width = min_line_width + (max_line_width - min_line_width) * normalized_weight
                    transition_weights_list.append(line_width)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    
    # Separate nodes by type for different styling
    memory_nodes = [node for node in all_nodes if node.startswith('M')]
    action_nodes = [node for node in all_nodes if node.startswith('A')]
    obs_nodes = [node for node in all_nodes if node.startswith('O')]
    
    # Further separate nodes based on whether they have incoming connections
    if fade_no_incoming:
        memory_nodes_normal = [node for node in memory_nodes if int(node[1:]) not in nodes_with_no_incoming]
        memory_nodes_faded = [node for node in memory_nodes if int(node[1:]) in nodes_with_no_incoming]
        
        action_nodes_normal = [node for node in action_nodes if int(node.split('_')[0][1:]) not in nodes_with_no_incoming]
        action_nodes_faded = [node for node in action_nodes if int(node.split('_')[0][1:]) in nodes_with_no_incoming]
        
        obs_nodes_normal = [node for node in obs_nodes if int(node.split('_')[0][1:]) not in nodes_with_no_incoming]
        obs_nodes_faded = [node for node in obs_nodes if int(node.split('_')[0][1:]) in nodes_with_no_incoming]
    else:
        memory_nodes_normal = memory_nodes
        memory_nodes_faded = []
        action_nodes_normal = action_nodes
        action_nodes_faded = []
        obs_nodes_normal = obs_nodes
        obs_nodes_faded = []
    
    # Draw memory nodes (large circles) - normal alpha
    if memory_nodes_normal:
        nx.draw_networkx_nodes(G, pos, nodelist=memory_nodes_normal, 
                              node_color=memory_node_color, node_size=mem_node_size, 
                              node_shape='o', edgecolors='black', linewidths=2, alpha=1.0,
                              ax=ax)
    
    # Draw memory nodes (large circles) - faded alpha
    if memory_nodes_faded:
        nx.draw_networkx_nodes(G, pos, nodelist=memory_nodes_faded, 
                              node_color=memory_node_color, node_size=mem_node_size, 
                              node_shape='o', edgecolors='black', linewidths=2, alpha=no_incoming_alpha,
                              ax=ax)
    
    # Draw action nodes (squares) - normal alpha
    if action_nodes_normal:
        nx.draw_networkx_nodes(G, pos, nodelist=action_nodes_normal, 
                              node_color=action_node_color, node_size=action_node_size, 
                              node_shape='s', edgecolors='black', linewidths=2, alpha=1.0,
                              ax=ax)
    
    # Draw action nodes (squares) - faded alpha
    if action_nodes_faded:
        nx.draw_networkx_nodes(G, pos, nodelist=action_nodes_faded, 
                              node_color=action_node_color, node_size=action_node_size, 
                              node_shape='s', edgecolors='black', linewidths=2, alpha=no_incoming_alpha,
                              ax=ax)
    
    # Draw observation nodes (diamonds) - separated by observation type for different colors
    for obs_idx in range(Y):
        # Get all observation nodes for this observation type
        obs_nodes_this_type_normal = [node for node in obs_nodes_normal if node.endswith(f'_{obs_idx}')]
        obs_nodes_this_type_faded = [node for node in obs_nodes_faded if node.endswith(f'_{obs_idx}')]
        
        # Draw observation nodes for this type - normal alpha
        if obs_nodes_this_type_normal:
            nx.draw_networkx_nodes(G, pos, nodelist=obs_nodes_this_type_normal, 
                                  node_color=observation_node_colors[obs_idx], node_size=obs_node_size, 
                                  node_shape='D', edgecolors='black', linewidths=2, alpha=1.0,
                                  ax=ax)
        
        # Draw observation nodes for this type - faded alpha
        if obs_nodes_this_type_faded:
            nx.draw_networkx_nodes(G, pos, nodelist=obs_nodes_this_type_faded, 
                                  node_color=observation_node_colors[obs_idx], node_size=obs_node_size, 
                                  node_shape='D', edgecolors='black', linewidths=2, alpha=no_incoming_alpha,
                                  ax=ax)
    
    # Draw internal edges (within complex nodes) - separate action and observation edges
    internal_edges = [(u, v) for u, v in G.edges() if (u, v) not in transition_edges]
    
    # Separate action edges (memory to action) from observation edges (action to observation)
    action_edges = [(u, v) for u, v in internal_edges if u.startswith('M') and v.startswith('A')]
    observation_edges = [(u, v) for u, v in internal_edges if u.startswith('A') and v.startswith('O')]
    
    # Further separate edges by alpha level if fading is enabled
    if fade_no_incoming:
        action_edges_normal = [(u, v) for u, v in action_edges if int(u[1:]) not in nodes_with_no_incoming]
        action_edges_faded = [(u, v) for u, v in action_edges if int(u[1:]) in nodes_with_no_incoming]
        
        observation_edges_normal = [(u, v) for u, v in observation_edges if int(u.split('_')[0][1:]) not in nodes_with_no_incoming]
        observation_edges_faded = [(u, v) for u, v in observation_edges if int(u.split('_')[0][1:]) in nodes_with_no_incoming]
    else:
        action_edges_normal = action_edges
        action_edges_faded = []
        observation_edges_normal = observation_edges
        observation_edges_faded = []
    
    # Draw action edges with thickness based on action probabilities
    if action_probabilities_permuted is not None:
        # Define minimum width threshold for drawing (to avoid zero-width edges with visible arrowheads)
        min_draw_width = 1e-6
        
        # Normal alpha action edges
        if action_edges_normal:
            action_edges_to_draw_normal = []
            action_widths_normal = []
            for mem_node, action_node in action_edges_normal:
                mem_id = int(mem_node[1:])
                mem_idx = memory_ids.index(mem_id)
                # Extract action index from node name A{mem_id}_{action_idx}
                action_idx = int(action_node.split('_')[1])
                prob = action_probabilities_permuted[mem_idx, action_idx]
                width = min_action_width + (max_action_width - min_action_width) * prob
                
                # Only include edges with visible width
                if width > min_draw_width:
                    action_edges_to_draw_normal.append((mem_node, action_node))
                    action_widths_normal.append(width)
            
            if action_edges_to_draw_normal:
                nx.draw_networkx_edges(G, pos, edgelist=action_edges_to_draw_normal, edge_color=action_edge_color, 
                                      arrows=True, arrowstyle='-|>', arrowsize=15, 
                                      width=action_widths_normal, min_target_margin=arrowhead_distance_actions, alpha=0.8,
                                      ax=ax)
        
        # Faded alpha action edges
        if action_edges_faded:
            action_edges_to_draw_faded = []
            action_widths_faded = []
            for mem_node, action_node in action_edges_faded:
                mem_id = int(mem_node[1:])
                mem_idx = memory_ids.index(mem_id)
                # Extract action index from node name A{mem_id}_{action_idx}
                action_idx = int(action_node.split('_')[1])
                prob = action_probabilities_permuted[mem_idx, action_idx]
                width = min_action_width + (max_action_width - min_action_width) * prob
                
                # Only include edges with visible width
                if width > min_draw_width:
                    action_edges_to_draw_faded.append((mem_node, action_node))
                    action_widths_faded.append(width)
            
            if action_edges_to_draw_faded:
                nx.draw_networkx_edges(G, pos, edgelist=action_edges_to_draw_faded, edge_color=action_edge_color, 
                                      arrows=True, arrowstyle='-|>', arrowsize=15, 
                                      width=action_widths_faded, min_target_margin=arrowhead_distance_actions, alpha=no_incoming_alpha,
                                      ax=ax)
    else:
        # Default thickness if no probabilities provided
        if action_edges_normal:
            nx.draw_networkx_edges(G, pos, edgelist=action_edges_normal, edge_color=action_edge_color, 
                                  arrows=True, arrowstyle='-|>', arrowsize=15, width=1,
                                  min_target_margin=arrowhead_distance_actions, alpha=0.8,
                                  ax=ax)
        if action_edges_faded:
            nx.draw_networkx_edges(G, pos, edgelist=action_edges_faded, edge_color=action_edge_color, 
                                  arrows=True, arrowstyle='-|>', arrowsize=15, width=1,
                                  min_target_margin=arrowhead_distance_actions, alpha=no_incoming_alpha,
                                  ax=ax)

    # Draw observation edges (action to observation) - standard thickness, no arrows
    if observation_edges_normal:
        nx.draw_networkx_edges(G, pos, edgelist=observation_edges_normal, edge_color='gray', 
                              arrows=False, width=1, alpha=1.0,
                              ax=ax)
    if observation_edges_faded:
        nx.draw_networkx_edges(G, pos, edgelist=observation_edges_faded, edge_color='gray', 
                              arrows=False, width=1, alpha=no_incoming_alpha,
                              ax=ax)

    # Draw transition edges (between complex nodes) - thicker, colored by weight
    if transition_edges:
        # Define minimum width threshold for drawing (to avoid zero-width edges with visible arrowheads)
        min_draw_width = 1e-6
        
        if fade_no_incoming:
            # Separate transition edges by source node alpha level
            transition_edges_normal = []
            transition_weights_normal = []
            transition_edges_faded = []
            transition_weights_faded = []
            
            for i, (obs_node, target_mem) in enumerate(transition_edges):
                # Only include edges with visible width
                if transition_weights_list[i] > min_draw_width:
                    # Extract source memory ID from observation node name O{mem_id}_A{action_idx}_{L/R}
                    source_mem_id = int(obs_node.split('_')[0][1:])
                    if source_mem_id in nodes_with_no_incoming:
                        transition_edges_faded.append((obs_node, target_mem))
                        transition_weights_faded.append(transition_weights_list[i])
                    else:
                        transition_edges_normal.append((obs_node, target_mem))
                        transition_weights_normal.append(transition_weights_list[i])
            
            # Draw normal alpha transition edges
            if transition_edges_normal:
                # Draw transition edges for each observation type (general for any Y)
                for obs_idx in range(Y):
                    transition_edges_obs = []
                    transition_weights_obs = []
                    
                    for i, (obs_node, target_mem) in enumerate(transition_edges_normal):
                        # Extract observation index from node name O{mem_id}_A{action_idx}_{obs_idx}
                        node_obs_idx = int(obs_node.split('_')[-1])
                        
                        if node_obs_idx == obs_idx:
                            transition_edges_obs.append((obs_node, target_mem))
                            transition_weights_obs.append(transition_weights_normal[i])
                    
                    if transition_edges_obs:
                        nx.draw_networkx_edges(
                            G, pos, edgelist=transition_edges_obs,
                            edge_color=observation_node_colors[obs_idx], arrows=True, arrowstyle='-|>',
                            arrowsize=20, width=transition_weights_obs,
                            min_target_margin=arrowhead_distance, alpha=0.7, ax = ax
                        )

            # Draw faded alpha transition edges
            if transition_edges_faded:
                for obs_idx in range(Y):
                    transition_edges_obs = []
                    transition_weights_obs = []
                    
                    for i, (obs_node, target_mem) in enumerate(transition_edges_faded):
                        # Extract observation index from node name O{mem_id}_A{action_idx}_{obs_idx}
                        node_obs_idx = int(obs_node.split('_')[-1])
                        
                        if node_obs_idx == obs_idx:
                            transition_edges_obs.append((obs_node, target_mem))
                            transition_weights_obs.append(transition_weights_faded[i])
                    
                    if transition_edges_obs:
                        nx.draw_networkx_edges(
                            G, pos, edgelist=transition_edges_obs,
                            edge_color=observation_node_colors[obs_idx], arrows=True, arrowstyle='-|>',
                            arrowsize=20, width=transition_weights_obs,
                            min_target_margin=arrowhead_distance, alpha=no_incoming_alpha, ax = ax
                        )
        else:
            # Filter out zero-width edges and draw all transition edges with normal alpha
            visible_edges = []
            visible_weights = []
            for i, edge in enumerate(transition_edges):
                if transition_weights_list[i] > min_draw_width:
                    visible_edges.append(edge)
                    visible_weights.append(transition_weights_list[i])
            
            if visible_edges:
                nx.draw_networkx_edges(G, pos, edgelist=visible_edges, 
                                      edge_color='red', arrows=True, arrowstyle='-|>', 
                                      arrowsize=20, width=visible_weights,
                                      min_target_margin=arrowhead_distance, alpha=0.7, ax=ax)
    
    # Create labels
    labels = {}
    for i, mem_id in enumerate(memory_ids):
        labels[f"M{mem_id}"] = memory_names[i]
        
        # Add labels for all action nodes
        for a_idx in range(A):
            # Check if this action node was hidden due to zero probability
            if hide_unused_actions and action_probabilities_permuted is not None:
                action_prob = action_probabilities_permuted[i, a_idx]
                if action_prob == 0:
                    continue  # Skip labels for hidden action nodes
            
            action_node = f"A{mem_id}_{a_idx}"
            labels[action_node] = action_names[a_idx] if a_idx < len(action_names) else f"A{a_idx}"
            
            # Add labels for observation nodes
            for y_idx in range(Y):
                # Only add label if this observation is allowed from this action
                if AllowedObsFromAct is None or AllowedObsFromAct[a_idx, y_idx]:
                    obs_node = f"O{mem_id}_A{a_idx}_{y_idx}"
                    labels[obs_node] = observation_names[y_idx] if y_idx < len(observation_names) else f"O{y_idx}"
    
    # Draw labels with different alpha for faded nodes
    if fade_no_incoming and nodes_with_no_incoming:
        # Separate labels for normal and faded nodes
        labels_normal = {}
        labels_faded = {}
        
        for node_key, label_text in labels.items():
            # Extract memory ID from node key
            if node_key.startswith('M'):
                mem_id = int(node_key[1:])
            elif node_key.startswith('A'):
                mem_id = int(node_key.split('_')[0][1:])
            elif node_key.startswith('O'):
                mem_id = int(node_key.split('_')[0][1:])
            
            if mem_id in nodes_with_no_incoming:
                labels_faded[node_key] = label_text
            else:
                labels_normal[node_key] = label_text
        
        # Draw normal labels
        if labels_normal:
            # Dynamically set font size based on node type and node size
            # Use a mapping: memory nodes -> mem_node_size, action nodes -> action_node_size, obs nodes -> obs_node_size
            for node_key, label_text in labels_normal.items():
                if node_key.startswith('M'):
                    node_size = mem_node_size
                elif node_key.startswith('A'):
                    node_size = action_node_size
                elif node_key.startswith('O'):
                    node_size = obs_node_size
                else:
                    node_size = 500  # fallback

                # Heuristic: font size proportional to sqrt(node_size)
                font_size = max(8, min(18, int(0.18 * node_size**0.69)))
                nx.draw_networkx_labels(G, pos, {node_key: label_text}, font_size=font_size, font_weight='bold', alpha=1.0,
                                        ax=ax)
        
        # Draw faded labels
        if labels_faded:
            # Dynamically set font size based on node type and node size
            for node_key, label_text in labels_faded.items():
                if node_key.startswith('M'):
                    node_size = mem_node_size
                elif node_key.startswith('A'):
                    node_size = action_node_size
                elif node_key.startswith('O'):
                    node_size = obs_node_size
                else:
                    node_size = 500

                # Heuristic: font size proportional to sqrt(node_size)
                font_size = max(8, min(18, int(0.18 * node_size**0.69)))
                nx.draw_networkx_labels(G, pos, {node_key: label_text}, font_size=font_size, font_weight='bold', alpha=no_incoming_alpha,
                                        ax=ax)
    else:
        # Draw all labels with normal alpha
        # Dynamically set font size based on node type and node size
        for node_key, label_text in labels.items():
            if node_key.startswith('M'):
                node_size = mem_node_size
            elif node_key.startswith('A'):
                node_size = action_node_size
            elif node_key.startswith('O'):
                node_size = obs_node_size
            else:
                node_size = 500

            # Heuristic: font size proportional to sqrt(node_size)
            font_size = max(8, min(18, int(0.18 * node_size**0.69)))
            nx.draw_networkx_labels(G, pos, {node_key: label_text}, font_size=font_size, font_weight='bold', alpha=1.0, ax=ax)
        #nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    # Customize plot
    arrangement_str = arrangement.capitalize()
    ax.axis('off')
    if ax is None:
        plt.tight_layout()
    
    if plot_legend:

        # Use the actual colors from the function arguments
        legend_elements = [
            Patch(facecolor=memory_node_color, edgecolor='black', label='Memory Node'),
            Patch(facecolor=action_node_color, edgecolor='black', label='Action Node'),
        ]

        # Add observation node colors to legend
        for i, color in enumerate(observation_node_colors):
            label = f'Observation Node {observation_names[i]}' if observation_names is not None else f'Observation Node {i}'
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))

        legend_elements.extend([
            Line2D([0], [0], color=action_edge_color, lw=2, alpha=0.8, label='Action Probabilities'),
            Line2D([0], [0], color='gray', lw=1, label='Observation Connections'),
        ])

        # For transition edges, use the observation node colors if colored, else fallback to 'red'
        if transition_edges:
            if fade_no_incoming or (observation_node_colors is not None and len(observation_node_colors) > 0):
                for i, color in enumerate(observation_node_colors):
                    label = f'Transition (Obs {observation_names[i]})' if observation_names is not None else f'Transition (Obs {i})'
                    legend_elements.append(Line2D([0], [0], color=color, lw=3, alpha=0.7, label=label))
            else:
                legend_elements.append(Line2D([0], [0], color='red', lw=3, alpha=0.7, label='Memory Transitions'))
        
        if arrangement == 'horizontal':
            legend_loc = 'upper right'
            bbox_anchor = (1.1, 1)
        else:
            legend_loc = 'center right'
            bbox_anchor = (1.2, 0.5)
        
        ax.legend(handles=legend_elements, loc=legend_loc, bbox_to_anchor=bbox_anchor)
    
    return G, ax


def draw_twolayers_FSC_network(transition_weights, action_probabilities=None, memory_ids=None, 
                               second_layer_nodes=None, arrangement='horizontal', 
                               spacing=6, layer_spacing=6, fig_size=None, min_weightsum_threshold=0.01,
                               mem_node_size=2000, action_node_size=1200, obs_node_size=500,
                               action_node_yoffset=1.5, action_node_xoffset=1.5,
                               obs_node_xoffset=0.7, obs_node_yoffset=0.7,
                               max_line_width=1, min_line_width=0.,
                               max_action_width=3, min_action_width=0,
                               memory_names=None, action_names=None, observation_names=None,
                               suppress_zero_action_transitions=False, action_prob_threshold=1e-10,
                               fade_no_incoming=False, no_incoming_alpha=0.3,
                               arrowhead_distance = 15, arrowhead_distance_actions = 15,
                               action_edge_color='blue', memory_node_color='lightblue', 
                               action_node_color='lightgreen', observation_node_colors=None,
                               plot_legend = False, reverse_obs_below=True, hide_unused_actions=False,
                               AllowedObsFromAct=None, obs_rotation=0.0,
                               ax = None):
    """
    Draw multiple complex node structures arranged in two layers with weighted transitions between observation and memory nodes.
    
    This function arranges memory nodes in two layers:
    - First layer: M-2 nodes (arranged in a line)
    - Second layer: exactly 2 nodes (centered below the first layer)
    
    The user can specify which nodes go in the second layer; by default, it uses the last two nodes.
    
    Parameters:
    -----------
    transition_weights : numpy.ndarray
        Shape (Y, A, M, M) where:
        - Y: observation index
        - A: action index  
        - M: starting memory node index
        - M: ending memory node index
    action_probabilities : numpy.ndarray or None
        Shape (M, A) - probability of taking action A from memory state M
        Controls thickness of edges from memory nodes to action nodes
    memory_ids : list or None
        List of memory node IDs that determines the spatial order of complex nodes.
        This completely rearranges both the visual layout AND the underlying data.
        For example, memory_ids=[1,0,2] will swap the first two memory nodes including
        all their transition probabilities and action probabilities.
        Must contain exactly the values [0, 1, ..., M-1] in any order.
        If None, uses [0, 1, 2, ..., M-1]
    second_layer_nodes : list or None
        List of exactly 2 node indices (referring to positions in memory_ids after permutation)
        that should be placed in the second layer. If None, defaults to the last two nodes [M-2, M-1].
        For example, if second_layer_nodes=[0, 3], the nodes at positions 0 and 3 in memory_ids
        will be placed in the centered second layer.
    arrangement : str
        'horizontal' or 'vertical' arrangement
    spacing : float
        Distance between adjacent complex nodes within the first layer
    layer_spacing : float
        Distance between the two layers. Default: 6
    fig_size : tuple or None
        Figure size (width, height). If None, automatically calculated
    min_weightsum_threshold : float
        Minimum weight sum for a memory node to be considered as having incoming connections.
    max_line_width : float
        Maximum line width for strongest transition connections
    min_line_width : float
        Minimum line width for weakest transition connections
    max_action_width : float
        Maximum line width for action connections (probability = 1.0)
    min_action_width : float
        Minimum line width for action connections (probability = 0.0)
    memory_names : list or None
        Custom names for memory nodes. If None, uses default 'M{id}' format
    action_names : list or None
        Custom names for action nodes. If None, uses default 'A0', 'A1', etc.
    observation_names : list or None
        Custom names for observation nodes. If None, uses default 'O0', 'O1', etc.
    suppress_zero_action_transitions : bool
        If True, suppress transition edges from observation nodes when the corresponding
        memory-to-action probability is below action_prob_threshold. Default: False
    action_prob_threshold : float
        Threshold below which action probabilities are considered zero for suppression.
        Only used when suppress_zero_action_transitions=True. Default: 1e-10
    fade_no_incoming : bool
        If True, apply reduced alpha to complex nodes that have no incoming connections
        from any other memory node. Default: False
    no_incoming_alpha : float
        Alpha (transparency) value for complex nodes with no incoming connections.
        Only used when fade_no_incoming=True. Range: 0.0 (fully transparent) to 1.0 (opaque).
        Default: 0.3
    arrowhead_distance : float
        Distance (in points) between arrowheads and target nodes. Larger values create
        more space between the arrowhead tip and the node edge. This affects all edges
        with arrows (action edges and transition edges). Default: 15
    action_edge_color : str
        Color for action probability edges (memory to action nodes). Default: 'blue'
    memory_node_color : str
        Color for memory nodes (large circles). Default: 'lightblue'
    action_node_color : str
        Color for action nodes (squares). Default: 'lightgreen'
    observation_node_colors : list or None
        List of colors for observation nodes (diamonds). Must have exactly Y elements
        if provided. Default: None (uses ['lightcoral', 'lightsalmon'] for Y=2)
    plot_legend : bool
        Whether to display a legend explaining the different node types and edge colors.
        Default: False
    reverse_obs_below : bool
        Whether to reverse the order of observation nodes for action nodes positioned
        below the memory node. When True (default), observation nodes are mirrored:
        - For Y=2: Top actions: left obs = obs[0], right obs = obs[1]
                   Bottom actions: left obs = obs[1], right obs = obs[0]
        - For Y>2: Top actions: circular arrangement starting from right (0°)
                   Bottom actions: mirrored circular arrangement (angles negated)
        This creates a symmetric layout around the memory node. Default: True
    hide_unused_actions : bool
        Whether to hide action nodes and their associated observation nodes when
        action_probabilities[m, a] = 0. When True, action nodes with zero probability
        will not be drawn, along with their observation nodes and edges. This requires
        action_probabilities to be provided. Default: False
    AllowedObsFromAct : numpy.ndarray or None
        Shape (A, Y) boolean array specifying which observations are allowed from each action.
        If AllowedObsFromAct[a, y] is False, the observation node O_y for action a will not
        be drawn and all transition weights from that observation will be set to zero.
        If None, all observations are allowed from all actions. Default: None
    obs_rotation : float
        Rotation angle (in radians) to apply to observation node positions when using
        circular arrangement (for Y > 2). Positive values rotate counter-clockwise.
        For example, π/4 (≈0.785) rotates by 45 degrees. Default: 0.0 (no rotation)
    ax : matplotlib.axes.Axes or None
        Matplotlib Axes object to draw on. If None, a new figure and axes are created.
        Default: None
    """
    Y, A, M, M_end = transition_weights.shape
    assert M == M_end, "Transition matrix must be square in memory dimensions"
    assert M >= 3, "M must be at least 3 for two-layer arrangement (first layer: M-2, second layer: 2)"
    
    # Validate action probabilities if provided
    if action_probabilities is not None:
        assert action_probabilities.shape == (M, A), f"Action probabilities must be shape (M, A) = ({M}, {A}), got {action_probabilities.shape}"
    
    # Validate AllowedObsFromAct if provided
    if AllowedObsFromAct is not None:
        assert isinstance(AllowedObsFromAct, np.ndarray), "AllowedObsFromAct must be a numpy array"
        assert AllowedObsFromAct.dtype == bool, "AllowedObsFromAct must be a boolean array"
        assert AllowedObsFromAct.shape == (A, Y), f"AllowedObsFromAct must be shape (A, Y) = ({A}, {Y}), got {AllowedObsFromAct.shape}"
    
    num_nodes = M
    
    # Set default memory IDs if not provided
    if memory_ids is None:
        memory_ids = list(range(num_nodes))
    elif len(memory_ids) != num_nodes:
        raise ValueError(f"Length of memory_ids ({len(memory_ids)}) must match M ({M})")
    
    # Validate that memory_ids contains unique values in range [0, M-1]
    if set(memory_ids) != set(range(M)):
        raise ValueError(f"memory_ids must contain exactly the values [0, 1, ..., {M-1}], got {memory_ids}")
    
    # Set default second layer nodes if not provided
    if second_layer_nodes is None:
        second_layer_nodes = [M-2, M-1]  # Last two nodes by default
    else:
        if len(second_layer_nodes) != 2:
            raise ValueError(f"second_layer_nodes must contain exactly 2 elements, got {len(second_layer_nodes)}")
        if not all(0 <= node < M for node in second_layer_nodes):
            raise ValueError(f"second_layer_nodes must contain values in range [0, {M-1}], got {second_layer_nodes}")
        if len(set(second_layer_nodes)) != 2:
            raise ValueError(f"second_layer_nodes must contain unique values, got {second_layer_nodes}")
    
    # Create layer assignment: which nodes go in which layer
    first_layer_nodes = [i for i in range(M) if i not in second_layer_nodes]
    nodes_in_first_layer = len(first_layer_nodes)
    
    # Create permutation arrays to rearrange data according to memory_ids
    perm_indices = memory_ids
    
    # Rearrange transition weights: (Y, A, M, M) -> permute both M dimensions
    transition_weights_permuted = transition_weights[:, :, perm_indices, :][:, :, :, perm_indices]
    
    # Rearrange action probabilities: (M, A) -> permute M dimension
    if action_probabilities is not None:
        action_probabilities_permuted = action_probabilities[perm_indices, :]
    else:
        action_probabilities_permuted = None
    
    # Identify memory nodes with no incoming connections (if fade_no_incoming is enabled)
    nodes_with_no_incoming = set()
    if fade_no_incoming:
        # Compute weighted incoming connections for each memory node
        transition_weights_no_self = transition_weights_permuted.copy()
        
        # Set diagonal elements to zero to exclude self-connections
        for m in range(M):
            transition_weights_no_self[:, :, m, m] = 0
        
        # Create a matrix of weighted connections between memory nodes
        memory_to_memory_weights = np.zeros((M, M))
        
        for y_idx in range(Y):
            for a_idx in range(A):
                for m_source in range(M):
                    for m_target in range(M):
                        if m_source != m_target:
                            weight = transition_weights_no_self[y_idx, a_idx, m_source, m_target]
                            if action_probabilities_permuted is not None:
                                action_prob = action_probabilities_permuted[m_source, a_idx]
                                weight *= action_prob
                            memory_to_memory_weights[m_source, m_target] += weight
        
        # Iteratively identify nodes to fade
        faded_nodes = set()
        changed = True
        
        while changed:
            changed = False
            for mem_idx in range(M):
                mem_id = memory_ids[mem_idx]
                
                if mem_id not in faded_nodes:
                    incoming_from_active = 0
                    for source_idx in range(M):
                        source_id = memory_ids[source_idx]
                        if source_id not in faded_nodes:
                            incoming_from_active += memory_to_memory_weights[source_idx, mem_idx]
                    
                    if incoming_from_active <= min_weightsum_threshold:
                        faded_nodes.add(mem_id)
                        changed = True
        
        nodes_with_no_incoming = faded_nodes
    
    # Set default names if not provided
    if memory_names is None:
        memory_names = [f'M{mem_id}' for mem_id in memory_ids]
    elif len(memory_names) != num_nodes:
        raise ValueError(f"Length of memory_names ({len(memory_names)}) must match M ({M})")
        
    if action_names is None:
        action_names = [f'A{i}' for i in range(A)]
    elif len(action_names) != A:
        raise ValueError(f"Length of action_names ({len(action_names)}) must match A ({A})")
        
    if observation_names is None:
        observation_names = [f'O{i}' for i in range(Y)]
    elif len(observation_names) != Y:
        raise ValueError(f"Length of observation_names ({len(observation_names)}) must match Y ({Y})")
    
    # Set default observation node colors if not provided
    if observation_node_colors is None:
        if Y == 2:
            observation_node_colors = ['lightcoral', 'lightsalmon']
        else:
            default_colors = ['lightcoral', 'lightsalmon', 'lightpink', 'lightsteelblue', 'lightseagreen']
            observation_node_colors = (default_colors * ((Y // len(default_colors)) + 1))[:Y]
    elif len(observation_node_colors) != Y:
        raise ValueError(f"Length of observation_node_colors ({len(observation_node_colors)}) must match Y ({Y})")
    
    # Calculate figure size if not provided
    if fig_size is None:
        if arrangement == 'horizontal':
            fig_size = (6 * max(nodes_in_first_layer, 2), 10)
        else:  # vertical
            fig_size = (12, 4 * max(nodes_in_first_layer, 2))
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Store all node names and positions
    all_nodes = []
    pos = {}
    
    # Store action positions for each memory node
    action_positions_dict = {}
    
    # Generate nodes and positions for each complex node in two-layer arrangement
    for m_idx, mem_id in enumerate(memory_ids):
        # Determine which layer and position within layer
        if m_idx in first_layer_nodes:
            layer = 0  # First layer (top)
            pos_in_layer = first_layer_nodes.index(m_idx)
            nodes_in_this_layer = nodes_in_first_layer
        else:  # m_idx in second_layer_nodes
            layer = 1  # Second layer (bottom)
            pos_in_layer = second_layer_nodes.index(m_idx)
            nodes_in_this_layer = 2

        # Calculate offset for this complex node
        if arrangement == 'horizontal':
            # Center nodes in their layer
            x_offset = (pos_in_layer - (nodes_in_this_layer - 1) / 2) * spacing
            y_offset = layer_spacing/2 * (1 - 2*layer)  # Top layer at +layer_spacing/2, bottom at -layer_spacing/2
        else:  # vertical
            x_offset = layer_spacing/2 * (1 - 2*layer)
            y_offset = ((nodes_in_this_layer - 1) / 2 - pos_in_layer) * spacing
        
        # Define node IDs for this complex node
        center_mem = f"M{mem_id}"
        
        # Calculate action node positions based on number of actions
        if A == 2:
            action_positions = [
                (x_offset, y_offset + action_node_yoffset),  # top
                (x_offset, y_offset - action_node_yoffset)   # bottom
            ]
        elif A == 3:
            action_positions = [
                (x_offset, y_offset + action_node_yoffset),
                (x_offset - action_node_xoffset/2, y_offset - action_node_yoffset),
                (x_offset + action_node_xoffset/2, y_offset - action_node_yoffset)
            ]
        elif A == 4:
            action_positions = [
                (x_offset - action_node_xoffset/2, y_offset + action_node_yoffset),
                (x_offset + action_node_xoffset/2, y_offset + action_node_yoffset),
                (x_offset - action_node_xoffset/2, y_offset - action_node_yoffset),
                (x_offset + action_node_xoffset/2, y_offset - action_node_yoffset)
            ]
        else:
            top_actions = (A + 1) // 2
            bottom_actions = A // 2
            action_positions = []
            
            if top_actions == 1:
                action_positions.append((x_offset, y_offset + action_node_yoffset))
            else:
                for j in range(top_actions):
                    x_pos = x_offset + (j - (top_actions - 1) / 2) * action_node_xoffset
                    action_positions.append((x_pos, y_offset + action_node_yoffset))
            
            if bottom_actions == 1:
                action_positions.append((x_offset, y_offset - action_node_yoffset))
            else:
                for j in range(bottom_actions):
                    x_pos = x_offset + (j - (bottom_actions - 1) / 2) * action_node_xoffset
                    action_positions.append((x_pos, y_offset - action_node_yoffset))
        
        # Store action positions for this memory node
        action_positions_dict[mem_id] = action_positions
        
        # Create action nodes and their observation nodes
        complex_nodes = [center_mem]
        
        for a_idx in range(A):
            if hide_unused_actions and action_probabilities_permuted is not None:
                action_prob = action_probabilities_permuted[m_idx, a_idx]
                if action_prob == 0:
                    continue
            
            action_node = f"A{mem_id}_{a_idx}"
            complex_nodes.append(action_node)
            
            for y_idx in range(Y):
                if AllowedObsFromAct is None or AllowedObsFromAct[a_idx, y_idx]:
                    obs_node = f"O{mem_id}_A{a_idx}_{y_idx}"
                    complex_nodes.append(obs_node)
                    G.add_edge(action_node, obs_node)
            
            G.add_edge(center_mem, action_node)
        
        all_nodes.extend(complex_nodes)
        
        # Set positions
        pos[center_mem] = (x_offset, y_offset)
        for a_idx in range(A):
            if hide_unused_actions and action_probabilities_permuted is not None:
                action_prob = action_probabilities_permuted[m_idx, a_idx]
                if action_prob == 0:
                    continue
            
            action_node = f"A{mem_id}_{a_idx}"
            pos[action_node] = action_positions[a_idx]
            
            action_x, action_y = action_positions[a_idx]
            is_action_below = action_y < y_offset
            
            if AllowedObsFromAct is None:
                allowed_obs_indices = list(range(Y))
            else:
                allowed_obs_indices = [y_idx for y_idx in range(Y) if AllowedObsFromAct[a_idx, y_idx]]
            
            Y_allowed = len(allowed_obs_indices)
            
            if Y_allowed == 0:
                continue
            elif Y_allowed == 1:
                obs_positions = [(action_x, action_y + obs_node_yoffset * 0.3)]
            elif Y_allowed == 2:
                if reverse_obs_below and is_action_below:
                    obs_positions = [
                        (action_x + obs_node_xoffset, action_y),
                        (action_x - obs_node_xoffset, action_y)
                    ]
                else:
                    obs_positions = [
                        (action_x - obs_node_xoffset, action_y),
                        (action_x + obs_node_xoffset, action_y)
                    ]
            else:
                obs_positions = []
                
                if Y_allowed == 3:
                    base_positions = [
                        (action_x - obs_node_xoffset, action_y),
                        (action_x + obs_node_xoffset, action_y),
                        (action_x, action_y + obs_node_yoffset),
                        (action_x, action_y - obs_node_yoffset)
                    ]
                    
                    if reverse_obs_below and is_action_below:
                        obs_positions = [
                            base_positions[1],
                            base_positions[0],
                            base_positions[3]
                        ]
                    else:
                        obs_positions = [
                            base_positions[0],
                            base_positions[1],
                            base_positions[2]
                        ]
                else:
                    for i in range(Y_allowed):
                        angle = 2 * np.pi * i / Y_allowed + obs_rotation
                        
                        if reverse_obs_below and is_action_below:
                            angle = -angle
                        
                        obs_x = action_x + obs_node_xoffset * np.cos(angle)
                        obs_y = action_y + obs_node_xoffset * np.sin(angle)
                        obs_positions.append((obs_x, obs_y))
            
            for i, y_idx in enumerate(allowed_obs_indices):
                if i < len(obs_positions):
                    obs_node = f"O{mem_id}_A{a_idx}_{y_idx}"
                    pos[obs_node] = obs_positions[i]
    
    # Add all nodes to the graph
    G.add_nodes_from(all_nodes)
    
    # Add weighted transition edges between observation and memory nodes
    transition_edges = []
    transition_weights_list = []
    
    max_weight = np.max(transition_weights_permuted)
    
    for y_idx in range(Y):
        for a_idx in range(A):
            for m_start_idx in range(M):
                for m_end_idx in range(M):
                    weight = transition_weights_permuted[y_idx, a_idx, m_start_idx, m_end_idx]
                    
                    if AllowedObsFromAct is not None and not AllowedObsFromAct[a_idx, y_idx]:
                        weight = 0
                    
                    if suppress_zero_action_transitions and action_probabilities_permuted is not None:
                        action_prob = action_probabilities_permuted[m_start_idx, a_idx]
                        if action_prob <= action_prob_threshold:
                            continue
                    
                    if hide_unused_actions and action_probabilities_permuted is not None:
                        action_prob = action_probabilities_permuted[m_start_idx, a_idx]
                        if action_prob == 0:
                            continue
                    
                    start_mem_id = memory_ids[m_start_idx]
                    end_mem_id = memory_ids[m_end_idx]
                    
                    obs_node = f"O{start_mem_id}_A{a_idx}_{y_idx}"
                    target_mem = f"M{end_mem_id}"
                    
                    G.add_edge(obs_node, target_mem, weight=weight)
                    transition_edges.append((obs_node, target_mem))
                    
                    normalized_weight = weight / max_weight
                    line_width = min_line_width + (max_line_width - min_line_width) * normalized_weight
                    transition_weights_list.append(line_width)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    
    # Separate nodes by type for different styling
    memory_nodes = [node for node in all_nodes if node.startswith('M')]
    action_nodes = [node for node in all_nodes if node.startswith('A')]
    obs_nodes = [node for node in all_nodes if node.startswith('O')]
    
    if fade_no_incoming:
        memory_nodes_normal = [node for node in memory_nodes if int(node[1:]) not in nodes_with_no_incoming]
        memory_nodes_faded = [node for node in memory_nodes if int(node[1:]) in nodes_with_no_incoming]
        
        action_nodes_normal = [node for node in action_nodes if int(node.split('_')[0][1:]) not in nodes_with_no_incoming]
        action_nodes_faded = [node for node in action_nodes if int(node.split('_')[0][1:]) in nodes_with_no_incoming]
        
        obs_nodes_normal = [node for node in obs_nodes if int(node.split('_')[0][1:]) not in nodes_with_no_incoming]
        obs_nodes_faded = [node for node in obs_nodes if int(node.split('_')[0][1:]) in nodes_with_no_incoming]
    else:
        memory_nodes_normal = memory_nodes
        memory_nodes_faded = []
        action_nodes_normal = action_nodes
        action_nodes_faded = []
        obs_nodes_normal = obs_nodes
        obs_nodes_faded = []
    
    # Draw memory nodes (large circles) - normal alpha
    if memory_nodes_normal:
        nx.draw_networkx_nodes(G, pos, nodelist=memory_nodes_normal, 
                              node_color=memory_node_color, node_size=mem_node_size, 
                              node_shape='o', edgecolors='black', linewidths=2, alpha=1.0,
                              ax=ax)
    
    if memory_nodes_faded:
        nx.draw_networkx_nodes(G, pos, nodelist=memory_nodes_faded, 
                              node_color=memory_node_color, node_size=mem_node_size, 
                              node_shape='o', edgecolors='black', linewidths=2, alpha=no_incoming_alpha,
                              ax=ax)
    
    # Draw action nodes (squares)
    if action_nodes_normal:
        nx.draw_networkx_nodes(G, pos, nodelist=action_nodes_normal, 
                              node_color=action_node_color, node_size=action_node_size, 
                              node_shape='s', edgecolors='black', linewidths=2, alpha=1.0,
                              ax=ax)
    
    if action_nodes_faded:
        nx.draw_networkx_nodes(G, pos, nodelist=action_nodes_faded, 
                              node_color=action_node_color, node_size=action_node_size, 
                              node_shape='s', edgecolors='black', linewidths=2, alpha=no_incoming_alpha,
                              ax=ax)
    
    # Draw observation nodes (diamonds)
    for obs_idx in range(Y):
        obs_nodes_this_type_normal = [node for node in obs_nodes_normal if node.endswith(f'_{obs_idx}')]
        obs_nodes_this_type_faded = [node for node in obs_nodes_faded if node.endswith(f'_{obs_idx}')]
        
        if obs_nodes_this_type_normal:
            nx.draw_networkx_nodes(G, pos, nodelist=obs_nodes_this_type_normal, 
                                  node_color=observation_node_colors[obs_idx], node_size=obs_node_size, 
                                  node_shape='D', edgecolors='black', linewidths=2, alpha=1.0,
                                  ax=ax)
        
        if obs_nodes_this_type_faded:
            nx.draw_networkx_nodes(G, pos, nodelist=obs_nodes_this_type_faded, 
                                  node_color=observation_node_colors[obs_idx], node_size=obs_node_size, 
                                  node_shape='D', edgecolors='black', linewidths=2, alpha=no_incoming_alpha,
                                  ax=ax)
    
    # Draw internal edges
    internal_edges = [(u, v) for u, v in G.edges() if (u, v) not in transition_edges]
    action_edges = [(u, v) for u, v in internal_edges if u.startswith('M') and v.startswith('A')]
    observation_edges = [(u, v) for u, v in internal_edges if u.startswith('A') and v.startswith('O')]
    
    if fade_no_incoming:
        action_edges_normal = [(u, v) for u, v in action_edges if int(u[1:]) not in nodes_with_no_incoming]
        action_edges_faded = [(u, v) for u, v in action_edges if int(u[1:]) in nodes_with_no_incoming]
        
        observation_edges_normal = [(u, v) for u, v in observation_edges if int(u.split('_')[0][1:]) not in nodes_with_no_incoming]
        observation_edges_faded = [(u, v) for u, v in observation_edges if int(u.split('_')[0][1:]) in nodes_with_no_incoming]
    else:
        action_edges_normal = action_edges
        action_edges_faded = []
        observation_edges_normal = observation_edges
        observation_edges_faded = []
    
    # Draw action edges with thickness based on action probabilities
    if action_probabilities_permuted is not None:
        min_draw_width = 1e-6
        
        if action_edges_normal:
            action_edges_to_draw_normal = []
            action_widths_normal = []
            for mem_node, action_node in action_edges_normal:
                mem_id = int(mem_node[1:])
                mem_idx = memory_ids.index(mem_id)
                action_idx = int(action_node.split('_')[1])
                prob = action_probabilities_permuted[mem_idx, action_idx]
                width = min_action_width + (max_action_width - min_action_width) * prob
                
                if width > min_draw_width:
                    action_edges_to_draw_normal.append((mem_node, action_node))
                    action_widths_normal.append(width)
            
            if action_edges_to_draw_normal:
                nx.draw_networkx_edges(G, pos, edgelist=action_edges_to_draw_normal, edge_color=action_edge_color, 
                                      arrows=True, arrowstyle='-|>', arrowsize=15, 
                                      width=action_widths_normal, min_target_margin=arrowhead_distance_actions, alpha=0.8,
                                      ax=ax)
        
        if action_edges_faded:
            action_edges_to_draw_faded = []
            action_widths_faded = []
            for mem_node, action_node in action_edges_faded:
                mem_id = int(mem_node[1:])
                mem_idx = memory_ids.index(mem_id)
                action_idx = int(action_node.split('_')[1])
                prob = action_probabilities_permuted[mem_idx, action_idx]
                width = min_action_width + (max_action_width - min_action_width) * prob
                
                if width > min_draw_width:
                    action_edges_to_draw_faded.append((mem_node, action_node))
                    action_widths_faded.append(width)
            
            if action_edges_to_draw_faded:
                nx.draw_networkx_edges(G, pos, edgelist=action_edges_to_draw_faded, edge_color=action_edge_color, 
                                      arrows=True, arrowstyle='-|>', arrowsize=15, 
                                      width=action_widths_faded, min_target_margin=arrowhead_distance_actions, alpha=no_incoming_alpha,
                                      ax=ax)
    else:
        if action_edges_normal:
            nx.draw_networkx_edges(G, pos, edgelist=action_edges_normal, edge_color=action_edge_color, 
                                  arrows=True, arrowstyle='-|>', arrowsize=15, width=1,
                                  min_target_margin=arrowhead_distance_actions, alpha=0.8,
                                  ax=ax)
        if action_edges_faded:
            nx.draw_networkx_edges(G, pos, edgelist=action_edges_faded, edge_color=action_edge_color, 
                                  arrows=True, arrowstyle='-|>', arrowsize=15, width=1,
                                  min_target_margin=arrowhead_distance_actions, alpha=no_incoming_alpha,
                                  ax=ax)

    # Draw observation edges
    if observation_edges_normal:
        nx.draw_networkx_edges(G, pos, edgelist=observation_edges_normal, edge_color='gray', 
                              arrows=False, width=1, alpha=1.0,
                              ax=ax)
    if observation_edges_faded:
        nx.draw_networkx_edges(G, pos, edgelist=observation_edges_faded, edge_color='gray', 
                              arrows=False, width=1, alpha=no_incoming_alpha,
                              ax=ax)

    # Draw transition edges
    if transition_edges:
        min_draw_width = 1e-6
        
        if fade_no_incoming:
            transition_edges_normal = []
            transition_weights_normal = []
            transition_edges_faded = []
            transition_weights_faded = []
            
            for i, (obs_node, target_mem) in enumerate(transition_edges):
                if transition_weights_list[i] > min_draw_width:
                    source_mem_id = int(obs_node.split('_')[0][1:])
                    if source_mem_id in nodes_with_no_incoming:
                        transition_edges_faded.append((obs_node, target_mem))
                        transition_weights_faded.append(transition_weights_list[i])
                    else:
                        transition_edges_normal.append((obs_node, target_mem))
                        transition_weights_normal.append(transition_weights_list[i])
            
            if transition_edges_normal:
                for obs_idx in range(Y):
                    transition_edges_obs = []
                    transition_weights_obs = []
                    
                    for i, (obs_node, target_mem) in enumerate(transition_edges_normal):
                        node_obs_idx = int(obs_node.split('_')[-1])
                        
                        if node_obs_idx == obs_idx:
                            transition_edges_obs.append((obs_node, target_mem))
                            transition_weights_obs.append(transition_weights_normal[i])
                    
                    if transition_edges_obs:
                        nx.draw_networkx_edges(
                            G, pos, edgelist=transition_edges_obs,
                            edge_color=observation_node_colors[obs_idx], arrows=True, arrowstyle='-|>',
                            arrowsize=20, width=transition_weights_obs,
                            min_target_margin=arrowhead_distance, alpha=0.7, ax=ax
                        )

            if transition_edges_faded:
                for obs_idx in range(Y):
                    transition_edges_obs = []
                    transition_weights_obs = []
                    
                    for i, (obs_node, target_mem) in enumerate(transition_edges_faded):
                        node_obs_idx = int(obs_node.split('_')[-1])
                        
                        if node_obs_idx == obs_idx:
                            transition_edges_obs.append((obs_node, target_mem))
                            transition_weights_obs.append(transition_weights_faded[i])
                    
                    if transition_edges_obs:
                        nx.draw_networkx_edges(
                            G, pos, edgelist=transition_edges_obs,
                            edge_color=observation_node_colors[obs_idx], arrows=True, arrowstyle='-|>',
                            arrowsize=20, width=transition_weights_obs,
                            min_target_margin=arrowhead_distance, alpha=no_incoming_alpha, ax=ax
                        )
        else:
            visible_edges = []
            visible_weights = []
            for i, edge in enumerate(transition_edges):
                if transition_weights_list[i] > min_draw_width:
                    visible_edges.append(edge)
                    visible_weights.append(transition_weights_list[i])
            
            if visible_edges:
                nx.draw_networkx_edges(G, pos, edgelist=visible_edges, 
                                      edge_color='red', arrows=True, arrowstyle='-|>', 
                                      arrowsize=20, width=visible_weights,
                                      min_target_margin=arrowhead_distance, alpha=0.7, ax=ax)
    
    # Create labels
    labels = {}
    for i, mem_id in enumerate(memory_ids):
        labels[f"M{mem_id}"] = memory_names[i]
        
        for a_idx in range(A):
            if hide_unused_actions and action_probabilities_permuted is not None:
                action_prob = action_probabilities_permuted[i, a_idx]
                if action_prob == 0:
                    continue
            
            action_node = f"A{mem_id}_{a_idx}"
            labels[action_node] = action_names[a_idx] if a_idx < len(action_names) else f"A{a_idx}"
            
            for y_idx in range(Y):
                if AllowedObsFromAct is None or AllowedObsFromAct[a_idx, y_idx]:
                    obs_node = f"O{mem_id}_A{a_idx}_{y_idx}"
                    labels[obs_node] = observation_names[y_idx] if y_idx < len(observation_names) else f"O{y_idx}"
    
    # Draw labels
    if fade_no_incoming and nodes_with_no_incoming:
        labels_normal = {}
        labels_faded = {}
        
        for node_key, label_text in labels.items():
            if node_key.startswith('M'):
                mem_id = int(node_key[1:])
            elif node_key.startswith('A'):
                mem_id = int(node_key.split('_')[0][1:])
            elif node_key.startswith('O'):
                mem_id = int(node_key.split('_')[0][1:])
            
            if mem_id in nodes_with_no_incoming:
                labels_faded[node_key] = label_text
            else:
                labels_normal[node_key] = label_text
        
        if labels_normal:
            for node_key, label_text in labels_normal.items():
                if node_key.startswith('M'):
                    node_size = mem_node_size
                elif node_key.startswith('A'):
                    node_size = action_node_size
                elif node_key.startswith('O'):
                    node_size = obs_node_size
                else:
                    node_size = 500

                font_size = max(8, min(18, int(0.18 * node_size**0.69)))
                nx.draw_networkx_labels(G, pos, {node_key: label_text}, font_size=font_size, font_weight='bold', alpha=1.0,
                                        ax=ax)
        
        if labels_faded:
            for node_key, label_text in labels_faded.items():
                if node_key.startswith('M'):
                    node_size = mem_node_size
                elif node_key.startswith('A'):
                    node_size = action_node_size
                elif node_key.startswith('O'):
                    node_size = obs_node_size
                else:
                    node_size = 500

                font_size = max(8, min(18, int(0.18 * node_size**0.69)))
                nx.draw_networkx_labels(G, pos, {node_key: label_text}, font_size=font_size, font_weight='bold', alpha=no_incoming_alpha,
                                        ax=ax)
    else:
        for node_key, label_text in labels.items():
            if node_key.startswith('M'):
                node_size = mem_node_size
            elif node_key.startswith('A'):
                node_size = action_node_size
            elif node_key.startswith('O'):
                node_size = obs_node_size
            else:
                node_size = 500

            font_size = max(8, min(18, int(0.18 * node_size**0.69)))
            nx.draw_networkx_labels(G, pos, {node_key: label_text}, font_size=font_size, font_weight='bold', alpha=1.0, ax=ax)
    
    # Customize plot
    ax.axis('off')
    if ax is None:
        plt.tight_layout()
    
    if plot_legend:
        legend_elements = [
            Patch(facecolor=memory_node_color, edgecolor='black', label='Memory Node'),
            Patch(facecolor=action_node_color, edgecolor='black', label='Action Node'),
        ]

        for i, color in enumerate(observation_node_colors):
            label = f'Observation Node {observation_names[i]}' if observation_names is not None else f'Observation Node {i}'
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))

        legend_elements.extend([
            Line2D([0], [0], color=action_edge_color, lw=2, alpha=0.8, label='Action Probabilities'),
            Line2D([0], [0], color='gray', lw=1, label='Observation Connections'),
        ])

        if transition_edges:
            if fade_no_incoming or (observation_node_colors is not None and len(observation_node_colors) > 0):
                for i, color in enumerate(observation_node_colors):
                    label = f'Transition (Obs {observation_names[i]})' if observation_names is not None else f'Transition (Obs {i})'
                    legend_elements.append(Line2D([0], [0], color=color, lw=3, alpha=0.7, label=label))
            else:
                legend_elements.append(Line2D([0], [0], color='red', lw=3, alpha=0.7, label='Memory Transitions'))
        
        if arrangement == 'horizontal':
            legend_loc = 'upper right'
            bbox_anchor = (1.1, 1)
        else:
            legend_loc = 'center right'
            bbox_anchor = (1.2, 0.5)
        
        ax.legend(handles=legend_elements, loc=legend_loc, bbox_to_anchor=bbox_anchor)
    
    return G, ax


def compute_memory_permutation_by_action_preference(action_probabilities):
    """
    Compute a permutation of memory nodes based on action preferences.
    
    This function sorts memory nodes based on the difference between their action probabilities:
    - Nodes with strong preference for action 2 appear on the left
    - Nodes with mixed preferences appear in the middle
    - Nodes with strong preference for action 1 appear on the right
    
    Parameters:
    -----------
    action_probabilities : numpy.ndarray
        Array of shape (M, A) containing action probabilities for each memory state
        
    Returns:
    --------
    list
        A list containing the memory IDs in the desired order for visualization
    """
    M, A = action_probabilities.shape
    
    # Calculate the action preference for each memory node
    # For the common case of 2 actions, calculate A1-A2 preference
    # Positive values indicate preference for action 1, negative values for action 2
    if A == 2:
        # A2 preference (action index 1) will be negative, so A1-A2 preference is higher
        # when A1 is preferred
        action_preference = action_probabilities[:, 0] - action_probabilities[:, 1]
    else:
        # For more than 2 actions, calculate a weighted preference
        action_preference = np.zeros(M)
        for m in range(M):
            # Preference for action 1 vs others
            action_preference[m] = action_probabilities[m, 0] - np.mean(action_probabilities[m, 1:])
    
    # Create a list of (memory_id, preference) tuples
    memory_preferences = [(m, action_preference[m]) for m in range(M)]
    
    # Sort based on action preference (ascending order: action 2 → mixed → action 1)
    sorted_memories = sorted(memory_preferences, key=lambda x: x[1])
    
    # Extract just the memory IDs from the sorted list
    memory_ids = [m_id for m_id, _ in sorted_memories]
    
    return memory_ids

def find_chain_order_with_connectivity_awareness(adjacency_matrix, node_ids, action_probabilities):
    """
    Find a chain-like ordering of nodes that balances connectivity and action
    preference smoothness.

    The function evaluates multiple candidate orderings (greedy from multiple
    starts, reverse orderings, and degree-based ordering) against an internal
    quality score that combines:
    - reward for strong adjacent connections,
    - penalty for strong skip-connections that bypass neighbors,
    - bonus for smooth transitions in action-probability vectors.

    Parameters:
    --- adjacency_matrix: np.ndarray
        Weighted adjacency matrix for the node subset being ordered.
    --- node_ids: list
        Original memory node identifiers corresponding to rows/columns in
        ``adjacency_matrix``.
    --- action_probabilities: np.ndarray or None
        Full action-probability matrix used to encourage smooth policy
        gradients along the chain. If ``None``, only connectivity is used.

    Returns:
    --- list
        Ordered list of node IDs (same elements as ``node_ids``).
    """
    n = len(node_ids)
    if n <= 1:
        return node_ids
    
    # Make adjacency matrix symmetric for chain finding
    symmetric_adj = adjacency_matrix + adjacency_matrix.T
    
    def evaluate_chain_quality_advanced(order):
        """
        Score a candidate chain order (lower is better).

        The score combines skip-connection penalties, adjacent-edge rewards,
        and action-similarity bonuses.
        """
        if len(order) != n:
            return float('inf')
        
        violations = 0
        chain_strength = 0
        
        for i in range(len(order)-1):
            current_idx = order[i]
            next_idx = order[i+1]
            
            # Reward strong direct connections
            direct_weight = symmetric_adj[current_idx, next_idx]
            chain_strength += direct_weight
            
            # Penalize long-range connections that bypass the chain
            for j in range(len(order)):
                if j != current_idx and j != next_idx:
                    skip_weight = symmetric_adj[current_idx, j]
                    # Penalize more heavily if the skipped connection is strong
                    distance_penalty = abs(i - order.index(j)) - 1
                    violations += skip_weight * distance_penalty
        
        # Bonus for maintaining action preference gradients
        action_consistency = 0
        if action_probabilities is not None:
            for i in range(len(order)-1):
                current_node = node_ids[order[i]]
                next_node = node_ids[order[i+1]]
                # Reward smooth transitions in action preferences
                action_diff = np.linalg.norm(action_probabilities[current_node] - action_probabilities[next_node])
                action_consistency += 1.0 / (1.0 + action_diff)  # Smoother transitions get higher scores
        
        # Combined score (lower is better)
        return violations - chain_strength - action_consistency * 0.5
    
    # Try different approaches
    best_order = list(range(n))
    best_score = evaluate_chain_quality_advanced(best_order)
    
    # Method 1: Greedy construction from different starting points
    for start_idx in range(n):
        current_order = [start_idx]
        remaining = set(range(n)) - {start_idx}
        
        while remaining:
            current_node = current_order[-1]
            
            # Find the best next node based on connection strength and action similarity
            best_next = None
            best_next_score = -float('inf')
            
            for candidate in remaining:
                # Connection strength
                connection_strength = symmetric_adj[current_node, candidate]
                
                # Action preference similarity (if available)
                action_similarity = 0
                if action_probabilities is not None:
                    current_node_id = node_ids[current_node]
                    candidate_node_id = node_ids[candidate]
                    action_diff = np.linalg.norm(action_probabilities[current_node_id] - action_probabilities[candidate_node_id])
                    action_similarity = 1.0 / (1.0 + action_diff)
                
                # Combined score
                candidate_score = connection_strength + action_similarity * 0.3
                
                if candidate_score > best_next_score:
                    best_next_score = candidate_score
                    best_next = candidate
            
            if best_next is not None:
                current_order.append(best_next)
                remaining.remove(best_next)
            else:
                # Add remaining nodes arbitrarily
                current_order.extend(list(remaining))
                break
        
        score = evaluate_chain_quality_advanced(current_order)
        if score < best_score:
            best_score = score
            best_order = current_order
        
        # Also try the reverse
        reverse_order = current_order[::-1]
        reverse_score = evaluate_chain_quality_advanced(reverse_order)
        if reverse_score < best_score:
            best_score = reverse_score
            best_order = reverse_order
    
    # Method 2: Try ordering by connection degree
    degrees = np.sum(symmetric_adj > 0, axis=1)
    degree_order = np.argsort(degrees)
    degree_score = evaluate_chain_quality_advanced(degree_order.tolist())
    if degree_score < best_score:
        best_score = degree_score
        best_order = degree_order.tolist()
    
    # Convert back to actual node IDs
    return [node_ids[i] for i in best_order]

def compute_chain_aware_memory_permutation(action_probabilities, transition_weights, action_threshold=0.6, fade_no_incoming=True, min_weightsum_threshold=0.01):
    """
    Compute a memory permutation that respects action preferences while optimizing for chain-like structures,
    taking into account the fading behavior of nodes with no incoming connections.
    
    Parameters:
    -----------
    action_probabilities : numpy.ndarray
        Array of shape (M, A) containing action probabilities for each memory state
    transition_weights : numpy.ndarray
        Array of shape (Y, A, M, M) containing transition weights
    action_threshold : float
        Threshold above which a memory node is considered to "prefer" an action
    fade_no_incoming : bool
        Whether to consider the fading behavior in the optimization
    min_weightsum_threshold : float
        Minimum weight sum threshold used in the visualization function
        
    Returns:
    --------
    list
        A list containing the memory IDs in the desired order for visualization
    """
    M, A = action_probabilities.shape
    
    def compute_nodes_with_no_incoming(memory_order):
        """Compute which nodes will be faded in the given memory order"""
        # Reorder data according to the proposed memory order
        perm_indices = memory_order
        transition_weights_permuted = transition_weights[:, :, perm_indices, :][:, :, :, perm_indices]
        action_probabilities_permuted = action_probabilities[perm_indices, :]
        
        # Apply the same logic as in draw_complex_network_with_transitions
        transition_weights_no_self = transition_weights_permuted.copy()
        for m in range(M):
            transition_weights_no_self[:, :, m, m] = 0
        
        incoming_sums = np.sum(transition_weights_no_self * (action_probabilities_permuted.T)[None, ..., None], axis=(0, 1, 2))
        
        nodes_with_no_incoming = set()
        for mem_idx in range(M):
            if incoming_sums[mem_idx] <= min_weightsum_threshold:
                nodes_with_no_incoming.add(memory_order[mem_idx])
        
        return nodes_with_no_incoming
    
    def build_effective_adjacency_matrix(node_group, consider_fading=True):
        """Build adjacency matrix considering only non-faded connections"""
        if len(node_group) <= 1:
            return np.zeros((len(node_group), len(node_group)))
        
        adjacency = np.zeros((len(node_group), len(node_group)))
        
        for i, node_i in enumerate(node_group):
            for j, node_j in enumerate(node_group):
                if i != j:
                    total_weight = 0
                    for y in range(transition_weights.shape[0]):
                        for a in range(transition_weights.shape[1]):
                            weight = transition_weights[y, a, node_i, node_j]
                            action_prob = action_probabilities[node_i, a]
                            
                            # Only count this connection if it won't be faded
                            if consider_fading and fade_no_incoming:
                                # Check if this connection would survive the fading logic
                                # This is an approximation - we assume the current partial ordering
                                temp_order = list(range(M))
                                temp_order[node_i], temp_order[node_j] = node_j, temp_order[node_j]
                                faded_nodes = compute_nodes_with_no_incoming(temp_order)
                                
                                if node_i not in faded_nodes:
                                    total_weight += weight * action_prob
                            else:
                                total_weight += weight * action_prob
                    
                    adjacency[i, j] = total_weight
        
        return adjacency
    
    # First, get the basic action preference ordering
    basic_order = compute_memory_permutation_by_action_preference(action_probabilities)
    
    # Classify nodes by their action preference
    action_groups = {}
    mixed_nodes = []
    
    for m in range(M):
        max_prob_idx = np.argmax(action_probabilities[m])
        max_prob = action_probabilities[m, max_prob_idx]
        
        if max_prob >= action_threshold:
            if max_prob_idx not in action_groups:
                action_groups[max_prob_idx] = []
            action_groups[max_prob_idx].append(m)
        else:
            mixed_nodes.append(m)
    
    # For each action group, optimize the internal ordering for chain-like structure
    optimized_groups = {}
    
    for action_idx, nodes in action_groups.items():
        if len(nodes) <= 1:
            optimized_groups[action_idx] = nodes
            continue
        
        # Build effective adjacency matrix
        adjacency = build_effective_adjacency_matrix(nodes, consider_fading=fade_no_incoming)
        
        # Find chain-like ordering for this group
        chain_order = find_chain_order_with_connectivity_awareness(adjacency, nodes, action_probabilities)
        optimized_groups[action_idx] = chain_order
    
    # Reconstruct the final ordering while preserving the overall action preference structure
    final_order = []
    
    # Build the order by going through the basic order and replacing groups with optimized versions
    processed_groups = set()
    
    for node in basic_order:
        # Find which group this node belongs to
        node_group = None
        node_action = None
        
        for action_idx, nodes in optimized_groups.items():
            if node in nodes:
                node_group = nodes
                node_action = action_idx
                break
        
        if node_group is not None and node_action not in processed_groups:
            # Add the whole optimized group
            final_order.extend(node_group)
            processed_groups.add(node_action)
        elif node in mixed_nodes:
            # This is a mixed node, add it individually
            final_order.append(node)
    
    return final_order

def get_optimized_memory_order(action_probabilities, transition_weights, action_threshold=0.6, fade_no_incoming=True, min_weightsum_threshold=0.01, reverse_left_part=False, reverse_right_part=False):
    """
    Corrected version that automatically determines the correct chain direction for each action group.
    
    Parameters:
    -----------
    action_probabilities : numpy.ndarray
        Array of shape (M, A) containing action probabilities for each memory state
    transition_weights : numpy.ndarray
        Array of shape (Y, A, M, M) containing transition weights
    action_threshold : float
        Threshold above which a memory node is considered to "prefer" an action
    fade_no_incoming : bool
        Whether to consider the fading behavior in the optimization
    min_weightsum_threshold : float
        Minimum weight sum threshold used in the visualization function
    reverse_left_part : bool
        If True, reverse the leftmost action group (lowest action preference) at the end
    reverse_right_part : bool
        If True, reverse the rightmost action group (highest action preference) at the end
        
    Returns:
    --------
    list
        A list containing the memory IDs in the desired order for visualization
    """
    # First get v2 result
    v2_result = compute_chain_aware_memory_permutation(action_probabilities, transition_weights, action_threshold, fade_no_incoming, min_weightsum_threshold)
    
    # Identify action groups in the v2 result
    M, A = action_probabilities.shape
    action_groups = {}
    mixed_nodes = []
    
    for m in range(M):
        max_prob_idx = np.argmax(action_probabilities[m])
        max_prob = action_probabilities[m, max_prob_idx]
        
        if max_prob >= action_threshold:
            if max_prob_idx not in action_groups:
                action_groups[max_prob_idx] = []
            action_groups[max_prob_idx].append(m)
        else:
            mixed_nodes.append(m)
    
    def determine_chain_direction(node_group, group_result_order):
        """
        Determine if a group should be reversed based on transition flow analysis.
        Returns True if the group should be reversed.
        """
        if len(node_group) <= 1:
            return False
        
        # Calculate net flow in the current ordering direction
        forward_flow = 0
        backward_flow = 0
        
        for i in range(len(group_result_order) - 1):
            current_node = group_result_order[i]
            next_node = group_result_order[i + 1]
            
            # Calculate flow from current to next (forward direction)
            for y in range(transition_weights.shape[0]):
                for a in range(transition_weights.shape[1]):
                    forward_weight = transition_weights[y, a, current_node, next_node]
                    backward_weight = transition_weights[y, a, next_node, current_node]
                    
                    forward_flow += forward_weight * action_probabilities[current_node, a]
                    backward_flow += backward_weight * action_probabilities[next_node, a]
        
        # If forward flow is stronger, we're going against the natural direction and should reverse
        return forward_flow > backward_flow
    
    # Apply directional correction to each action group
    corrected_result = v2_result.copy()
    
    for action_idx, nodes in action_groups.items():
        if len(nodes) <= 1:
            continue
            
        # Find the positions of this action group's nodes in v2_result
        group_positions = []
        for i, node in enumerate(v2_result):
            if node in nodes:
                group_positions.append(i)
        
        # Extract the group nodes in their v2 order
        group_order = [v2_result[i] for i in group_positions]
        
        # Determine if this group should be reversed
        should_reverse = determine_chain_direction(nodes, group_order)
        
        if should_reverse:
            # Reverse this action group
            reversed_group = group_order[::-1]
            
            # Rebuild the result with reversed group
            for i, pos in enumerate(group_positions):
                corrected_result[pos] = reversed_group[i]
    
    # Apply optional manual reversals at the end
    if reverse_left_part or reverse_right_part:
        # Identify the leftmost and rightmost action groups in the final result
        action_group_positions = {}
        
        for action_idx, nodes in action_groups.items():
            if len(nodes) <= 1:
                continue
                
            # Find positions of this group in corrected_result
            positions = []
            for i, node in enumerate(corrected_result):
                if node in nodes:
                    positions.append(i)
            
            if positions:
                action_group_positions[action_idx] = positions
        
        if action_group_positions:
            # Sort action groups by their leftmost position to identify left and right
            sorted_groups = sorted(action_group_positions.items(), key=lambda x: min(x[1]))
            
            if reverse_left_part and len(sorted_groups) > 0:
                # Reverse the leftmost action group
                leftmost_action, leftmost_positions = sorted_groups[0]
                leftmost_nodes = [corrected_result[i] for i in leftmost_positions]
                reversed_leftmost = leftmost_nodes[::-1]
                
                # Apply the reversal
                for i, pos in enumerate(leftmost_positions):
                    corrected_result[pos] = reversed_leftmost[i]
            
            if reverse_right_part and len(sorted_groups) > 0:
                # Reverse the rightmost action group
                rightmost_action, rightmost_positions = sorted_groups[-1]
                rightmost_nodes = [corrected_result[i] for i in rightmost_positions]
                reversed_rightmost = rightmost_nodes[::-1]
                
                # Apply the reversal
                for i, pos in enumerate(rightmost_positions):
                    corrected_result[pos] = reversed_rightmost[i]
    
    return corrected_result