import numpy as np
import pandas as pd
from typing import Dict, Tuple



# use the local lingam package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)



import tigramite
from tigramite.pcmci import PCMCI as PCMCI_model
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmisymb import CMIsymb

from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator


class PCMCI(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            """Construct a PCMCI model.
                Parameters
                ----------
                dataframe : data object
                    This is the Tigramite dataframe object. Among others, it has the
                    attributes dataframe.values yielding a numpy array of shape (
                    observations T, variables N) and optionally a mask of the same shape.
                cond_ind_test : conditional independence test object
                    This can be ParCorr or other classes from
                    ``tigramite.independence_tests`` or an external test passed as a
                    callable. This test can be based on the class
                    tigramite.independence_tests.CondIndTest.
                verbosity : int, optional (default: 0)
                    Verbose levels 0, 1, ... if larger than 1, print detailed info
            """
            
            'dataframe': None,
            'cond_ind_test': ParCorr(),
            'verbosity': 0,
            
            
            """
                selected_links : dict or None
                    Deprecated, replaced by link_assumptions
                link_assumptions : dict
                    Dictionary of form {j:{(i, -tau): link_type, ...}, ...} specifying
                    assumptions about links. This initializes the graph with entries
                    graph[i,j,tau] = link_type. For example, graph[i,j,0] = '-->'
                    implies that a directed link from i to j at lag 0 must exist.
                    Valid link types are 'o-o', '-->', '<--'. In addition, the middle
                    mark can be '?' instead of '-'. Then '-?>' implies that this link
                    may not exist, but if it exists, its orientation is '-->'. Link
                    assumptions need to be consistent, i.e., graph[i,j,0] = '-->'
                    requires graph[j,i,0] = '<--' and acyclicity must hold. If a link
                    does not appear in the dictionary, it is assumed absent. That is,
                    if link_assumptions is not None, then all links have to be specified
                    or the links are assumed absent.
                tau_min : int, optional (default: 0)
                    Minimum time lag to test. Note that zero-lags are undirected.
                tau_max : int, optional (default: 1)
                    Maximum time lag. Must be larger or equal to tau_min.
                save_iterations : bool, optional (default: False)
                    Whether to save iteration step results such as conditions used.
                pc_alpha : float, optional (default: 0.05)
                    Significance level in algorithm.
                max_conds_dim : int, optional (default: None)
                    Maximum number of conditions to test. If None is passed, this number
                    is unrestricted.
                max_combinations : int, optional (default: 1)
                    Maximum number of combinations of conditions of current cardinality
                    to test in PC1 step.
                max_conds_py : int, optional (default: None)
                    Maximum number of conditions of Y to use. If None is passed, this
                    number is unrestricted.
                max_conds_px : int, optional (default: None)
                    Maximum number of conditions of Z to use. If None is passed, this
                    number is unrestricted.
                alpha_level : float, optional (default: 0.05)
                    Significance level at which the p_matrix is thresholded to 
                    get graph.
                fdr_method : str, optional (default: 'fdr_bh')
                    Correction method, currently implemented is Benjamini-Hochberg
                    False Discovery Rate method. 
            """
            
            'selected_links': None,
            'link_assumptions': None,
            'tau_min': 0,
            'tau_max': 1,
            'save_iterations': False,
            'pc_alpha': 0.2,
            'max_conds_dim': None,
            'max_combinations': 1,
            'max_conds_py': None,
            'max_conds_px': None,
            'alpha_level': 0.05,
            'fdr_method': 'none',   
        }
        self._params.update(params)

    @property
    def name(self):
        return "PCMCI"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['dataframe', 'cond_ind_test', 'verbosity']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['selected_links', 'link_assumptions', 'tau_min', 'tau_max', 'save_iterations', 'pc_alpha', 'max_conds_dim', 'max_combinations', 'max_conds_py', 'max_conds_px', 'alpha_level', 'fdr_method']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, PCMCI_model]:
        # PCMCI
        
        pcmci = PCMCI_model(dataframe=data, cond_ind_test=self._params['cond_ind_test'])
        results = pcmci.run_pcmci(**self.get_secondary_params())
        
        """
        Returns
        -------
        graph : array of shape [N, N, tau_max+1]
            Causal graph, see description above for interpretation.
        val_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of test statistic values.
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values, optionally adjusted if fdr_method is
            not 'none'.
        conf_matrix : array of shape [N, N, tau_max+1,2]
            Estimated matrix of confidence intervals of test statistic values.
            Only computed if set in cond_ind_test, where also the percentiles
            are set.
        """
        # adj_matrices = results['graph']
        
        # Prepare additional information
        info = {
            'val_matrix': results['val_matrix'],
            'p_matrix': results['p_matrix'],
            'conf_matrix': results['conf_matrix'],
        }
        
        # create adj_matrices  # Note that zero-lags are undirected.
        matrices = (results['p_matrix'] < self._params['alpha_level']).astype(int)
        adj_matrices = np.array([matrices[:, :, lag].T for lag in range(matrices.shape[2])])
        
        return adj_matrices, info, pcmci
    
    
    
    def generate_data(self, n=5, T=1000, random_state=None):
        '''
        n = 5  # Number of variables
        T = 1000  # Number of time steps
        random_state = None  # Random seed (None means no fixed seed)
        '''
        
        np.random.seed(random_state)  # Set the seed (not effective here since it's None)

        T_spurious = 20  # Extra time steps used to remove startup artifacts # This adds extra time steps at the beginning to stabilize the system before collecting actual data.
        expon = 1.5  # Exponent used for non-Gaussian noise # Used later to generate non-Gaussian noise.
        
        '''
        Used for randomly generating causal adjacent coefficient matrix
        # Constructing B0 (Instantaneous Effect Matrix): Generates a random matrix B0 that defines the instantaneous causal relationships between the 5 variables.
        value = np.random.uniform(low=0.05, high=0.5, size=(n, n))  # Random weights (0.05 to 0.5)
        sign = np.random.choice([-1, 1], size=(n, n))  # Randomly assign positive or negative sign
        B0 = np.multiply(value, sign)  # Multiply values by their signs
        B0 = np.multiply(B0, np.random.binomial(1, 0.4, size=(n, n)))  # Apply sparsity (40% chance of being nonzero)
        B0 = np.tril(B0, k=-1)  # Keep only lower triangular part (ensuring acyclic structure)
        
        # Constructing B1 (Lagged Effect Matrix): Similar to B0, but represents lagged dependencies (i.e., how past values influence current values).
        value = np.random.uniform(low=0.05, high=0.5, size=(n, n))
        sign = np.random.choice([-1, 1], size=(n, n))
        B1 = np.multiply(value, sign)
        B1 = np.multiply(B1, np.random.binomial(1, 0.4, size=(n, n)))
        # M1 = np.dot(np.linalg.inv(np.eye(n) - B0), B1)  # Computes the lagged influence matrix M1 by accounting for instantaneous effects (B0) using matrix inversion.
        '''
        # manually set adjacent coefficient matrix
        B0 = [
            [0,-0.12,0,0,0],
            [0,0,0,0,0],
            [-0.41,0.01,0,-0.02,0],
            [0.04,-0.22,0,0,0],
            [0.15,0,-0.03,0,0],
        ]
        B1 = [
            [-0.32,0,0.12,0.32,0],
            [0,-0.35,-0.1,-0.46,0.4],
            [0,0,0.37,0,0.46],
            [-0.38,-0.1,-0.24,0,-0.13],
            [0,0,0,0,0],
        ]
        causal_order = [1, 0, 3, 2, 4]
        
        # Generating Non-Gaussian Noise # Generates non-Gaussian noise by raising absolute values to the power of expon (1.5).
        ee = np.empty((n, T + T_spurious))
        for i in range(n):
            ee[i, :] = np.random.normal(size=(1, T + T_spurious))  # Standard normal noise
            ee[i, :] = np.multiply(np.sign(ee[i, :]), abs(ee[i, :]) ** expon)  # Introduce non-Gaussianity
            ee[i, :] = ee[i, :] - np.mean(ee[i, :])  # Normalize mean to 0
            ee[i, :] = ee[i, :] / np.std(ee[i, :])  # Normalize standard deviation to 1

        # Adjusting Noise Influence
        std_e = np.random.uniform(size=(n,)) + 0.5  # Random noise standard deviation (0.5 to 1.5)
        nn = np.dot(np.dot(np.linalg.inv(np.eye(n) - B0), np.diag(std_e)), ee) # Adjusted noise that incorporates instantaneous effects (B0).
        
        
        # Simulating the Time Series (xx)
        xx = np.zeros((n, T + T_spurious))  # Initialize data storage
        xx[:, 0] = np.random.normal(size=(n,))  # Random initial state

        for t in range(1, T + T_spurious):
            # xx[:, t] = np.dot(M1, xx[:, t - 1]) + nn[:, t]      #The new value xx[:, t] is computed as: A function of past values (M1 * xx[:, t - 1]),  Plus new noise (nn[:, t])
            lag_influence = np.dot(B1, xx[:, t - 1])
            xx[:, t] = lag_influence + np.dot(B0, lag_influence) + nn[:, t]
            
        # Extracting Final Dataset
        data = xx[:, T_spurious + 1 : T_spurious + T]  # The first T_spurious time steps are removed to ensure a stable system.
        # df = pd.DataFrame(data.T, columns=["x1", "x2", "x3", "x4", "x5"])
        df = pp.DataFrame(data.T)
        return df
        
        
        
        
    def test_algorithm(self):
        # Generate sample data with linear relationships    # np.random.seed(42)    # n_samples = 1000
        df = self.generate_data(n=5, T=1000, random_state=42)

        print("Testing PCMCI algorithm with dataframe:")
        params = {
            'dataframe': None,
            'cond_ind_test': ParCorr,
            'verbosity': 0,
            'selected_links': None,
            'link_assumptions': None,
            'tau_min': 0,
            'tau_max': 1,
            'save_iterations': False,
            'pc_alpha': 0.2,
            'max_conds_dim': None,
            'max_combinations': 1,
            'max_conds_py': None,
            'max_conds_px': None,
            'alpha_level': 0.05,
            'fdr_method': 'none',  
        }
        adj_matrix, info, pcmci = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        
        print("\nAdditional Info:")
        # print("  Test Statistic Values (val_matrix):")
        # print("    ", info['val_matrix'])
        
        # print("  P-Values (p_matrix):")
        # print("    ", info['p_matrix'])
        
        # print("  Confidence Intervals (conf_matrix):")
        # print("    ", info['conf_matrix'])
        pcmci.print_significant_links(p_matrix=info['p_matrix'],
                                         val_matrix=info['val_matrix'],
                                         alpha_level=0.05)
  
  
        # Ground truth graph
        gt_graph = np.array([
            [
                [0,-0.12,0,0,0],
                [0,0,0,0,0],
                [-0.41,0.01,0,-0.02,0],
                [0.04,-0.22,0,0,0],
                [0.15,0,-0.03,0,0],
            ],
            [
                [-0.32,0,0.12,0.32,0],
                [0,-0.35,-0.1,-0.46,0.4],
                [0,0,0.37,0,0.46],
                [-0.38,-0.1,-0.24,0,-0.13],
                [0,0,0,0,0],
            ]
        ])


if __name__ == "__main__":
    pcmci_algo = PCMCI({})
    pcmci_algo.test_algorithm() 

