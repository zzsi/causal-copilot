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
sys.path.insert(0, causal_learn_dir)


from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.CDNOD import cdnod as cl_cdnod

from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator

class CDNOD(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'kci',
            'stable': True,
            'uc_rule': 0,
            'uc_priority': 2,
            'depth': 3, #-1,
            'mvcdnod': False,
            'correction_name': 'MV_Crtn_Fisher_Z',
            'background_knowledge': None,
            'verbose': False,
            'show_progress': False,
        }
        self._params.update(params)

    @property
    def name(self):
        return "CDNOD"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        secondary_param_keys = ['stable', 'uc_rule', 'uc_priority', 'mvcdnod', 'correction_name',
                                'background_knowledge', 'verbose', 'show_progress']
        return {k: v for k, v in self._params.items() if k in secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CausalGraph]:
        node_names = list(data.columns)
        # Extract c_indx (assuming it's the last column)
        c_indx = data['domain_index'].values.reshape(-1, 1)
        data = data.drop(columns=['domain_index'])
        data_values = data.values

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params(), 'node_names': node_names}

        # Run CD-NOD algorithm
        cg = cl_cdnod(data_values, c_indx, **all_params)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(cg)

        # Prepare additional information
        info = {
            'graph': cg,
            'PC_elapsed': cg.PC_elapsed,
        }

        return adj_matrix, info, cg

    def convert_to_adjacency_matrix(self, cg: CausalGraph) -> np.ndarray:
        adj_matrix = cg.G.graph
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

        # remove the domain index column
        inferred_flat = inferred_flat[:-1, :-1]
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
    cdnod = CDNOD()
    cdnod.test_algorithm()