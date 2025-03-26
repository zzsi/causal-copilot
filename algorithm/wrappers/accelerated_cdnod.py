import numpy as np
import pandas as pd
from typing import Dict, Tuple

# use the local causal-learn package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)
sys.path.append(causal_learn_dir)

from externals.acceleration.cdnod.cdnod import accelerated_cdnod
from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator
from causallearn.graph.GraphClass import CausalGraph, GeneralGraph
from causallearn.utils.PCUtils import SkeletonDiscovery, UCSepset, Meek

class AcceleratedCDNOD(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'cmiknn',
            'depth': -1,
        }
        self._params.update(params)

    @property
    def name(self):
        return "AcceleratedCDNOD"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        secondary_param_keys = []
        return {k: v for k, v in self._params.items() if k in secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CausalGraph]:
        """
        Runs GPU-accelerated CDNOD, ensuring c_index is correctly handled.
        """
        node_names = list(data.columns)
        data_values = data.values

        # **Extract context index (c_indx)**
        if 'domain_index' not in data.columns:
            raise ValueError("Dataset must contain a 'domain_index' column for CDNOD.")
        c_indx = data['domain_index'].values.reshape(-1, 1)
        data = data.drop(columns=['domain_index'])

        # **Combine parameters**
        all_params = {**self.get_primary_params(), **self.get_secondary_params()}

        # **Run GPU-Accelerated CDNOD**
        cg = accelerated_cdnod(data, c_indx, **all_params)

        # **Convert graph to adjacency matrix**
        adj_matrix = self.convert_to_adjacency_matrix(cg)

        info = {
            'graph': cg,
        }

        return adj_matrix, info, cg

    def convert_to_adjacency_matrix(self, cg: CausalGraph) -> np.ndarray:
        adj_matrix = cg[:-1, :-1]
        inferred_flat = np.zeros_like(adj_matrix)
        indices = np.where(adj_matrix == 1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                # directed edge: j -> i
                inferred_flat[i, j] = 1
            elif adj_matrix[j, i] == 1:
                # bidirected edge: j <-> i
                if inferred_flat[j, i] == 0:
                    # keep asymmetric that only one entry is recorded
                    inferred_flat[i, j] = 3

        indices = np.where(adj_matrix == -1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                # undirected edge: j -- i
                if inferred_flat[j, i] == 0:
                    inferred_flat[i, j] = 2
        return inferred_flat


    def test_algorithm(self):
        # Generate sample data with linear relationships and multiple domains
        np.random.seed(42)
        n_samples = 1000
        n_domains = 3
        
        # Create base variables with different distributions per domain
        X1 = np.zeros(n_samples)
        X2 = np.zeros(n_samples)
        X3 = np.zeros(n_samples)
        X4 = np.zeros(n_samples)
        X5 = np.zeros(n_samples)
        
        # Create domain indices
        domain_indices = np.random.choice(range(n_domains), size=n_samples)
        
        # Generate data with domain-specific parameters
        for domain in range(n_domains):
            domain_mask = domain_indices == domain
            n_domain_samples = np.sum(domain_mask)
            
            # Domain-specific noise scales
            noise_scale = 0.5 + domain * 0.2
            
            # Domain-specific causal strengths
            coef_scale = 0.5 + domain * 0.1
            
            # Generate data for this domain
            X1[domain_mask] = np.random.normal(domain * 0.5, 1, n_domain_samples)
            X2[domain_mask] = coef_scale * X1[domain_mask] + np.random.normal(0, noise_scale, n_domain_samples)
            X3[domain_mask] = (0.3 + domain * 0.1) * X1[domain_mask] + (0.7 - domain * 0.1) * X2[domain_mask] + np.random.normal(0, noise_scale * 0.6, n_domain_samples)
            X4[domain_mask] = (0.6 + domain * 0.05) * X2[domain_mask] + np.random.normal(0, noise_scale * 0.8, n_domain_samples)
            X5[domain_mask] = (0.4 - domain * 0.05) * X3[domain_mask] + (0.5 + domain * 0.05) * X4[domain_mask] + np.random.normal(0, noise_scale * 0.4, n_domain_samples)
        
        # Create DataFrame with domain indices
        df = pd.DataFrame({
            'X1': X1, 
            'X2': X2, 
            'X3': X3, 
            'X4': X4, 
            'X5': X5, 
            'domain_index': domain_indices
        })
        
        print(f"Testing CDNOD algorithm with {n_domains} domains:")
        print(f"Domain distribution: {np.bincount(domain_indices)}")
        
        params = {
            'alpha': 0.05,
            'depth': 2,
            'indep_test': 'fisherz',
            'verbose': False,
            'show_progress': False
        }
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)

        # Ground truth graph
        gt_graph = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0]
        ])

        # Use GraphEvaluator to compute metrics
        evaluator = GraphEvaluator()
        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)

        print("\nMetrics:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"SHD: {metrics['shd']:.4f}")


if __name__ == "__main__":
    cdnod_algo = AcceleratedCDNOD({})
    cdnod_algo.test_algorithm()
