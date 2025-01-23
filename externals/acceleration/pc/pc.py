import numpy as np
import cupy as cp
import dask.array as da
from gpucsl.pc.pc import GaussianPC, DiscretePC

# use the local causal-learn package
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.PC import pc
from typing import Tuple, Dict


# GPU-accelerated KCI test using Dask + CuPy
def kci_dask_cupy(X: int, Y: int, Z: list, data: da.Array, alpha: float, gamma: float) -> bool:
    n_samples = data.shape[0]

    if len(Z) == 0:
        X_data = data[:, X].reshape(-1, 1)
        Y_data = data[:, Y].reshape(-1, 1)
    else:
        Z_data = data[:, Z]
        X_data = data[:, X]
        Y_data = data[:, Y]

        # Residualize X on Z
        beta_X = cp.linalg.lstsq(Z_data, X_data, rcond=None)[0]
        X_data = X_data - Z_data @ beta_X

        # Residualize Y on Z
        beta_Y = cp.linalg.lstsq(Z_data, Y_data, rcond=None)[0]
        Y_data = Y_data - Z_data @ beta_Y

    # Compute RBF kernels
    K_X = cp.exp(-gamma * cp.linalg.norm(X_data[:, None] - X_data, axis=2) ** 2)
    K_Y = cp.exp(-gamma * cp.linalg.norm(Y_data[:, None] - Y_data, axis=2) ** 2)

    # Center kernels
    H = cp.eye(n_samples) - cp.ones((n_samples, n_samples)) / n_samples
    Kc_X = H @ K_X @ H
    Kc_Y = H @ K_Y @ H

    # Compute HSIC statistic
    hsic_stat = cp.trace(Kc_X @ Kc_Y) / ((n_samples - 1) ** 2)

    return hsic_stat < alpha


# Fisher-Z with GPU acceleration using gpucsl
def fisherz_gpu_gpucsl(data: np.ndarray, alpha: float, depth: int) -> Tuple[np.ndarray, Dict, CausalGraph]:
    pc_result = GaussianPC(data, depth, alpha).set_distribution_specific_options().execute()
    directed_graph, separation_sets, _, _, _, _ = pc_result
    cg = CausalGraph(directed_graph)
    cg.sepset = separation_sets
    info = {
        'sepset': separation_sets,
        'PC_elapsed': pc_result.PC_elapsed,
    }
    return directed_graph, info, cg


# Chi-Square with GPU acceleration using gpucsl
def chi_square_gpu_gpucsl(data: np.ndarray, alpha: float, depth: int) -> Tuple[np.ndarray, Dict, CausalGraph]:
    pc_result = DiscretePC(data, depth, alpha).set_distribution_specific_options().execute()
    directed_graph, separation_sets, _, _, _, _ = pc_result
    cg = CausalGraph(directed_graph)
    cg.sepset = separation_sets
    info = {
        'sepset': separation_sets,
        'PC_elapsed': pc_result.PC_elapsed,
    }
    return directed_graph, info, cg


# Unified PC algorithm function
def accelerated_pc(
    data: np.ndarray, alpha: float = 0.05, indep_test: str = 'fisherz', depth: int = -1, gamma: float = 1.0
) -> Tuple[np.ndarray, Dict, CausalGraph]:
    """
    Accelerated PC algorithm that chooses between GPU-accelerated implementations
    based on the independence test.
    
    Args:
        data: Input dataset as a NumPy array.
        alpha: Significance level.
        indep_test: Independence test to use ('fisherz', 'chi_square', 'kci').
        depth: Maximum conditioning set size.
        gamma: Bandwidth for KCI test.
        
    Returns:
        Tuple containing adjacency matrix, info dictionary, and CausalGraph object.
    """
    if indep_test == 'fisherz':
        return fisherz_gpu_gpucsl(data, alpha, depth)

    elif indep_test == 'chi_square':
        return chi_square_gpu_gpucsl(data, alpha, depth)

    elif indep_test == 'kci':
        data_dask = da.from_array(cp.asarray(data), chunks=data.shape)
        pc_result = pc(
            data,
            alpha=alpha,
            indep_test=lambda X, Y, Z: kci_dask_cupy(X, Y, Z, data_dask, alpha, gamma),
            depth=depth,
        )
        directed_graph = pc_result.G
        cg = CausalGraph(directed_graph)
        return directed_graph, cg

    else:
        raise ValueError(f"Independence test '{indep_test}' not supported. Use 'fisherz', 'chi_square', or 'kci'.")
