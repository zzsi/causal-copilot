import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple

# use the local causal-learn package
import sys

sys.path.insert(0, 'causal-learn')

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.PC import pc as cl_pc
from causallearn.search.ConstraintBased.FCI import fci as cl_fci
from causallearn.search.ConstraintBased.CDNOD import cdnod as cl_cdnod
from causallearn.search.ScoreBased.GES import ges as cl_ges
from causallearn.search.FCMBased.lingam.direct_lingam import DirectLiNGAM as CLDirectLiNGAM
from causallearn.search.FCMBased.lingam.ica_lingam import ICALiNGAM as CLICALiNGAM

from .base import CausalDiscoveryAlgorithm


class PC(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'depth': -1,
            'stable': True,
            'uc_rule': 0,
            'uc_priority': -1,
            'mvpc': False,
            'correction_name': 'MV_Crtn_Fisher_Z',
            'background_knowledge': None,
            'verbose': False,
            'show_progress': False,
        }
        self._params.update(params)

    @property
    def name(self):
        return "PC"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['stable', 'uc_rule', 'uc_priority', 'mvpc', 'correction_name',
                                      'background_knowledge', 'verbose', 'show_progress']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data = data.values
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params(), 'node_names': node_names}

        # Run PC algorithm
        cg = cl_pc(data, **all_params)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(cg)

        # Prepare additional information
        info = {
            'sepset': cg.sepset,
            'definite_UC': cg.definite_UC,
            'definite_non_UC': cg.definite_non_UC,
            'PC_elapsed': cg.PC_elapsed,
        }

        return adj_matrix, info

    def convert_to_adjacency_matrix(self, cg: CausalGraph) -> np.ndarray:
        adj_matrix = np.zeros_like(cg.G.graph)
        for i in range(cg.G.graph.shape[0]):
            for j in range(cg.G.graph.shape[1]):
                if cg.G.graph[i, j] == 1 and cg.G.graph[j, i] == -1:
                    # only keep the determined arrows (j --> i)
                    adj_matrix[i, j] = 1
        return adj_matrix

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples, n_features = 1000, 5
        X = np.random.randn(n_samples, n_features)

        # Test with numpy array
        print("Testing PC algorithm with numpy array:")
        params = {
            'alpha': 0.05,
            'depth': 2,
            'indep_test': 'fisherz',
            'verbose': False,
            'show_progress': False
        }
        adj_matrix, info = self.fit(X)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("\nAdditional Info:")
        print(f"PC elapsed time: {info['PC_elapsed']:.4f} seconds")
        print(f"Number of definite unshielded colliders: {len(info['definite_UC'])}")
        print(f"Number of definite non-unshielded colliders: {len(info['definite_non_UC'])}")

        # Test with pandas DataFrame
        print("\nTesting PC algorithm with pandas DataFrame:")
        df = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_features)])
        adj_matrix, info = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("\nAdditional Info:")
        print(f"PC elapsed time: {info['PC_elapsed']:.4f} seconds")
        print(f"Number of definite unshielded colliders: {len(info['definite_UC'])}")
        print(f"Number of definite non-unshielded colliders: {len(info['definite_non_UC'])}")


class FCI(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'depth': -1,
            'max_path_length': -1,
            'verbose': False,
            'background_knowledge': None,
            'show_progress': False,
        }
        self._params.update(params)

    @property
    def name(self):
        return "FCI"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['max_path_length', 'verbose', 'background_knowledge', 'show_progress']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data = data.values
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params(), 'node_names': node_names}

        # Run FCI algorithm
        graph, edges = cl_fci(data, **all_params)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(graph)

        # Prepare additional information
        info = {
            'edges': edges,
            'graph': graph,
        }

        return adj_matrix, info

    def convert_to_adjacency_matrix(self, graph: CausalGraph) -> np.ndarray:
        adj_matrix = np.zeros_like(graph.graph, dtype=int)
        for i in range(graph.graph.shape[0]):
            for j in range(graph.graph.shape[1]):
                # only keep the determined arrows (j --> i)
                if graph.graph[i, j] == 1 and graph.graph[j, i] == -1:
                    adj_matrix[i, j] = 1  # j --> i
        return adj_matrix

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples, n_features = 1000, 5
        X = np.random.randn(n_samples, n_features)

        # Test with numpy array
        print("Testing FCI algorithm with numpy array:")
        params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'verbose': False,
            'show_progress': False
        }
        adj_matrix, info = self.fit(X)
        print("Adjacency Matrix:")
        print(adj_matrix)

        # Test with pandas DataFrame
        print("\nTesting FCI algorithm with pandas DataFrame:")
        df = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_features)])
        adj_matrix, info = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)


class CDNOD(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'stable': True,
            'uc_rule': 0,
            'uc_priority': 2,
            'depth': -1,
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

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data = data.values
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]

        # Extract c_indx (assuming it's the last column)
        c_indx = data[:, -1].reshape(-1, 1)
        data = data[:, :-1]

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params(), 'node_names': node_names}

        # Run CD-NOD algorithm
        cg = cl_cdnod(data, c_indx, **all_params)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(cg)

        # Prepare additional information
        info = {
            'graph': cg,
            'PC_elapsed': cg.PC_elapsed,
        }

        return adj_matrix, info

    def convert_to_adjacency_matrix(self, cg: CausalGraph) -> np.ndarray:
        adj_matrix = np.zeros_like(cg.G.graph, dtype=int)
        for i in range(cg.G.graph.shape[0]):
            for j in range(cg.G.graph.shape[1]):
                # only keep the determined arrows (j --> i)
                if cg.G.graph[i, j] == 1 and cg.G.graph[j, i] == -1:
                    adj_matrix[i, j] = 1  # j --> i
        return adj_matrix

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples, n_features = 1000, 5
        X = np.random.randn(n_samples, n_features)
        c_indx = np.random.randint(0, 2, size=(n_samples, 1))
        X_with_c_indx = np.hstack((X, c_indx))

        # Test with numpy array
        print("Testing CD-NOD algorithm with numpy array:")
        params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'verbose': False,
            'show_progress': False
        }
        adj_matrix, info = self.fit(X_with_c_indx)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print(f"CD-NOD elapsed time: {info['PC_elapsed']:.4f} seconds")

        # Test with pandas DataFrame
        print("\nTesting CD-NOD algorithm with pandas DataFrame:")
        df = pd.DataFrame(X_with_c_indx, columns=[f'X{i}' for i in range(n_features)] + ['c_indx'])
        adj_matrix, info = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print(f"CD-NOD elapsed time: {info['PC_elapsed']:.4f} seconds")


class GES(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'score_func': 'local_score_BIC',
            'maxP': None,
            'parameters': None,
        }
        self._params.update(params)

    @property
    def name(self):
        return "GES"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['score_func', 'maxP']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['parameters']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data = data.values
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params(), 'node_names': node_names}

        # Run GES algorithm
        record = cl_ges(data, **all_params)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(record['G'])

        # Prepare additional information
        info = {
            'score': record['score'],
            'update1': record['update1'],
            'update2': record['update2'],
        }

        return adj_matrix, info

    def convert_to_adjacency_matrix(self, G: CausalGraph) -> np.ndarray:
        adj_matrix = np.zeros_like(G.graph, dtype=int)
        for i in range(G.graph.shape[0]):
            for j in range(G.graph.shape[1]):
                if G.graph[i, j] == 1 and G.graph[j, i] == -1:
                    adj_matrix[i, j] = 1  # j --> i
        return adj_matrix

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples, n_features = 1000, 5
        X = np.random.randn(n_samples, n_features)

        # Test with numpy array
        print("Testing GES algorithm with numpy array:")
        params = {
            'score_func': 'local_score_BIC',
            'maxP': None,
        }
        adj_matrix, info = self.fit(X)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print(f"GES score: {info['score']:.4f}")

        # Test with pandas DataFrame
        print("\nTesting GES algorithm with pandas DataFrame:")
        df = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_features)])
        adj_matrix, info = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print(f"GES score: {info['score']:.4f}")


class DirectLiNGAM(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'random_state': None,
            'prior_knowledge': None,
            'apply_prior_knowledge_softly': False,
            'measure': 'pwling'
        }
        self._params.update(params)

    @property
    def name(self):
        return "DirectLiNGAM"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['measure']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['random_state', 'prior_knowledge', 'apply_prior_knowledge_softly']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data = data.values
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params()}

        # Run DirectLiNGAM algorithm
        model = CLDirectLiNGAM(**all_params)
        model.fit(data)

        # Convert the graph to adjacency matrix
        adj_matrix = model.adjacency_matrix_

        # Prepare additional information
        info = {
            'causal_order': model.causal_order_
        }

        return adj_matrix, info

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples, n_features = 1000, 5
        X = np.random.randn(n_samples, n_features)

        # Test with numpy array
        print("Testing DirectLiNGAM algorithm with numpy array:")
        params = {
            'measure': 'pwling',
            'random_state': 42
        }
        adj_matrix, info = self.fit(X)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("Causal Order:")
        print(info['causal_order'])

        # Test with pandas DataFrame
        print("\nTesting DirectLiNGAM algorithm with pandas DataFrame:")
        df = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_features)])
        adj_matrix, info = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("Causal Order:")
        print(info['causal_order'])


class ICALiNGAM(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'random_state': None,
            'max_iter': 1000
        }
        self._params.update(params)

    @property
    def name(self):
        return "ICALiNGAM"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['max_iter']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['random_state']
        return {}

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data = data.values
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]

        # Run ICALiNGAM algorithm
        model = CLICALiNGAM(**self.get_primary_params(), **self.get_secondary_params())
        model.fit(data)

        # Convert the graph to adjacency matrix
        adj_matrix = model.adjacency_matrix_

        # Prepare additional information
        info = {
            'causal_order': model.causal_order_
        }

        return adj_matrix, info

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples, n_features = 1000, 5
        X = np.random.randn(n_samples, n_features)

        # Test with numpy array
        print("Testing ICALiNGAM algorithm with numpy array:")
        params = {
            'random_state': 42,
            'max_iter': 1000
        }
        adj_matrix, info = self.fit(X)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("Causal Order:")
        print(info['causal_order'])

        # Test with pandas DataFrame
        print("\nTesting ICALiNGAM algorithm with pandas DataFrame:")
        df = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_features)])
        adj_matrix, info = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("Causal Order:")
        print(info['causal_order'])


if __name__ == "__main__":
    # pc_algo = PC({})
    # pc_algo.test_algorithm()

    # fci_algo = FCI({})
    # fci_algo.test_algorithm()

    # cdnod_algo = CDNOD({})
    # cdnod_algo.test_algorithm()

    # ges_algo = GES({})
    # ges_algo.test_algorithm()

    direct_lingam_algo = DirectLiNGAM({})
    direct_lingam_algo.test_algorithm()

    ica_lingam_algo = ICALiNGAM({})
    ica_lingam_algo.test_algorithm()