import sys
import numpy as np
import time
# import psutil

# use the local causal-learn package
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
sys.path.insert(0, causal_learn_dir)

from gpucsl.pc.discover_skeleton_gaussian import discover_skeleton_gpu_gaussian
from gpucsl.pc.discover_skeleton_discrete import discover_skeleton_gpu_discrete
from gpucsl.pc.helpers import init_pc_graph
from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.PCUtils import SkeletonDiscovery, UCSepset, Meek
from typing import Tuple, Dict, Optional


# **GPU-Accelerated Skeleton Discovery**
def skeleton_discovery_gpu(data: np.ndarray, alpha: float, indep_test: str = 'fisherz') -> np.ndarray:
    """
    Perform skeleton discovery using `discover_skeleton_gpu_gaussian`.
    Returns the adjacency matrix of the skeleton.
    """
    graph = init_pc_graph(data)
    num_variables = data.shape[1]
    num_observations = data.shape[0]

    (
        (skeleton, _, _, _),
        _,
    ) = discover_skeleton_gpu_gaussian(
        graph, data, None, alpha, num_variables, num_observations
    )

    return skeleton  # Return adjacency matrix


# **Orient Edges Using UCSepset & Meek Rules**
def orient_edges(
    causal_graph: CausalGraph, c_indx: np.ndarray, alpha: float, uc_rule: int, uc_priority: int, background_knowledge=None
) -> CausalGraph:
    """
    Apply edge orientation rules using UCSepset and Meek algorithms.
    """
    c_indx_id = causal_graph.G.get_num_nodes() - 1  # Context index is the last node

    # **Orient the direction from c_indx to X**
    for i in causal_graph.G.get_adjacent_nodes(causal_graph.G.nodes[c_indx_id]):
        causal_graph.G.add_directed_edge(causal_graph.G.nodes[c_indx_id], i)


    # **UCSepset rule for collider orientation**
    if uc_rule == 0:
        if uc_priority != -1:
            causal_graph = UCSepset.uc_sepset(causal_graph, uc_priority, background_knowledge=background_knowledge)
        else:
            causal_graph = UCSepset.uc_sepset(causal_graph, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        if uc_priority != -1:
            causal_graph = UCSepset.maxp(causal_graph, uc_priority, background_knowledge=background_knowledge)
        else:
            causal_graph = UCSepset.maxp(causal_graph, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        if uc_priority != -1:
            causal_graph = UCSepset.definite_maxp(causal_graph, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            causal_graph = UCSepset.definite_maxp(causal_graph, alpha, background_knowledge=background_knowledge)

        # **Apply Meek's definite meek rules first**
        causal_graph = Meek.definite_meek(causal_graph, background_knowledge=background_knowledge)

    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")

    # **Apply Meek rules for additional edge orientation**
    causal_graph = Meek.meek(causal_graph, background_knowledge=background_knowledge)

    return causal_graph


# **Accelerated CDNOD using GPuCSL**
def accelerated_cdnod(
    data: np.ndarray, c_indx: np.ndarray, indep_test: str = 'fisherz', alpha: float = 0.05, depth: int = -1, uc_rule: int = 0, uc_priority: int = -1
) -> Tuple[np.ndarray, Dict, CausalGraph, float, float]:
    """
    Accelerated CDNOD algorithm that runs with GPuCSL acceleration.
    
    Args:
        data: Input dataset as a NumPy array.
        c_indx: Context index capturing domain shifts.
        alpha: Significance level.
        depth: Maximum conditioning set size.
        uc_rule: Rule for unshielded collider orientation.
        uc_priority: Rule for prioritizing collider conflicts.
        
    Returns:
        Tuple containing adjacency matrix, info dictionary, CausalGraph object, runtime, and memory usage.
    """
    # process = psutil.Process()
    # start_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    start_time = time.time()

    # Ensure `c_indx` is correctly reshaped
    c_indx = c_indx.reshape(-1, 1)

    # Append `c_indx` to data
    data_aug = np.concatenate((data, c_indx), axis=1)

    # **Perform GPU-Accelerated Skeleton Discovery**
    causal_graph = skeleton_discovery_gpu(data_aug, alpha, indep_test)

    # **Run CDNOD using Fisher-Z GPU acceleration**
    #directed_graph, info, causal_graph = fisherz_gpu_gpucsl(data_aug, alpha, depth)

    # **Orient edges using UCSepset and Meek**
    causal_graph = orient_edges(causal_graph, c_indx, alpha, uc_rule, uc_priority)

    # **Convert graph to adjacency matrix**
    # adj_matrix = convert_to_adjacency_matrix(causal_graph)

    runtime = time.time() - start_time
    # end_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    # memory_usage = end_memory - start_memory
    memory_usage = 0

    return causal_graph
