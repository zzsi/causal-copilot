import numpy as np
import pandas as pd
from typing import Set, Tuple, List, Callable, Dict, Optional
import networkx as nx
import matplotlib.pyplot as plt

# use the local causal-learn package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.insert(0, causal_learn_dir)

from causallearn.utils.cit import CIT, fisherz, kci, chisq, gsq


def MB2CPDAG(data: pd.DataFrame, mb_dict: Dict[int, List[int]], indep_test: str = 'fisherz', alpha: float = 0.05, n_jobs: int = 1) -> np.ndarray:
    """
    Convert Markov blanket results to a CPDAG by running PC algorithm on each target and its MB nodes.
    
    Args:
        data: pandas DataFrame containing the data
        mb_dict: Dictionary mapping each target node index to list of its MB node indices
        is_discrete: Whether the data is discrete (True) or continuous (False)
        alpha: Significance level for independence tests
        
    Returns:
        numpy array representing the merged CPDAG adjacency matrix
    """
    from causallearn.search.ConstraintBased.PC import pc
    
    n_vars = data.shape[1]
    merged_cpdag = np.zeros((n_vars, n_vars))
    
    def compute_local_cpdag(target, mb_nodes):
        local_indices = [target] + mb_nodes
        if len(local_indices) < 2:
            return None
        if len(local_indices) == 2:
            local_adj = np.zeros((2,2))
            local_adj[0,1] = 2
            return (local_indices, local_adj)
        local_data = data.iloc[:, local_indices].values
        node_names = [str(i) for i in local_indices]
        cg = pc(local_data,
                alpha=alpha,
                indep_test=indep_test,
                node_names=node_names,
                verbose=False)
        local_adj = cg.G.graph
        return (local_indices, local_adj)

    results = []
    for target, mb_nodes in mb_dict.items():
        result = compute_local_cpdag(target, mb_nodes)
        results.append(result)

    for res in results:
        if res is None:
            continue
        local_indices, local_adj = res
        for i in range(len(local_indices)):
            for j in range(len(local_indices)):
                if local_adj[i, j] == 1:
                    if local_adj[j, i] == -1:
                        merged_cpdag[local_indices[i], local_indices[j]] = 1
                    elif local_adj[j, i] == 1:
                        if merged_cpdag[local_indices[j], local_indices[i]] == 0:
                            merged_cpdag[local_indices[i], local_indices[j]] = 2
                elif local_adj[i, j] == -1 and local_adj[j, i] == -1:
                    if merged_cpdag[local_indices[j], local_indices[i]] == 0:
                        merged_cpdag[local_indices[i], local_indices[j]] = 2

    # Remove symmetries by keeping only one entry for undirected edges
    indices = np.where(merged_cpdag == 2)
    for i, j in zip(indices[0], indices[1]):
        if merged_cpdag[i,j] == merged_cpdag[j,i] == 2:
            merged_cpdag[i,j] = 2
            merged_cpdag[j,i] = 0
    indices = np.where(merged_cpdag == 1)
    for i, j in zip(indices[0], indices[1]):
        if merged_cpdag[i,j] == merged_cpdag[j,i] == 1:
            merged_cpdag[i,j] = 2
            merged_cpdag[j,i] = 0 

    return merged_cpdag


# class MB2CPDAG:
#     def __init__(self, data: pd.DataFrame, ci_test: str = 'fisherz'):
#         """
#         Initialize the classifier with a dataset and conditional independence test
        
#         Args:
#             data: DataFrame containing observations of all variables
#             ci_test: Conditional independence test to use ('fisherz', 'kci', 'chisq', 'gsq')
#         """
#         self.data = data.to_numpy()
#         self.column_names = data.columns.tolist()
#         self.ci_test = CIT(self.data, method=ci_test)
        
#     def conditional_independence_test(self, A: str, B: str, conditioning_set: List[str], alpha: float = 0.05) -> bool:
#         """
#         Test if A and B are conditionally independent given conditioning_set using the specified CI test
        
#         Args:
#             A: First variable name
#             B: Second variable name 
#             conditioning_set: List of conditioning variable names
#             alpha: Significance level for independence test
            
#         Returns:
#             bool: True if independent, False if dependent
#         """
#         A_idx = self.column_names.index(A)
#         B_idx = self.column_names.index(B)
#         conditioning_indices = [self.column_names.index(X) for X in conditioning_set]
        
#         print(f"A: {A_idx}, B: {B_idx}, conditioning_set: {conditioning_indices}")
#         pvalue = self.ci_test(A_idx, B_idx, conditioning_indices)
#         print(f"A: {A}, B: {B}, conditioning_set: {conditioning_set}, pvalue: {pvalue}, alpha: {alpha}")
#         return pvalue >= alpha
    
#     def classify_blanket_nodes(self, T: str, MB: List[str]) -> Dict[str, str]:
#         """
#         Classify Markov blanket nodes into spouses and parent/child nodes
        
#         Args:
#             T: Target node
#             MB: List of nodes in T's Markov blanket
            
#         Returns:
#             Dict[str, str]: Dictionary mapping each MB node to either 'spouse' or 'parent_or_child'
#         """
#         classification = {}
#         MB_set = set(MB)
        
#         for X in MB:
#             # Condition on all other MB nodes except X
#             conditioning_set = list(MB_set - {X})
            
#             # Test if T and X are independent given MB\{X}
#             # For parent or child, T and X are dependent whether conditioning on MB\{X} or not
#             # For spouse, T and X are independent before conditioning on MB\{X}, but dependent after conditioning on MB\{X}
#             is_independent_before = self.conditional_independence_test(T, X, [])
#             is_independent_after = self.conditional_independence_test(T, X, conditioning_set)
#             # If T ⟂ X | (MB(T)\{X}), then X is a spouse
#             # else X is a direct neighbor (parent or child)
#             if is_independent_before and not is_independent_after:
#                 classification[X] = 'spouse'
#             else:
#                 classification[X] = 'parent_or_child'
                
#         return classification
    
    

# def generate_test_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, Dict[str, str]]:
#     """
#     Generate synthetic data from a known Bayesian network structure
    
#     Returns:
#         Tuple[pd.DataFrame, Dict[str, str]]: 
#             - Generated data
#             - True classification of nodes
#     """
#     # Generate parent nodes
#     np.random.seed(42)
#     data = pd.DataFrame({
#         'P1': np.random.normal(0, 1, n_samples),
#         'P2': np.random.normal(0, 1, n_samples),
#         'S1': np.random.normal(0, 1, n_samples),
#         'S2': np.random.normal(0, 1, n_samples)
#     })
    
#     # Generate target node X based on parents
#     data['X'] = (0.9 * data['P1'] + 
#                  0.8 * data['P2'] + 
#                  np.random.normal(0, 0.5, n_samples))
    
#     # Generate child node based on X and spouse nodes
#     data['C1'] = (0.8 * data['X'] + 
#                   0.7 * data['S1'] + 
#                   0.6 * data['S2'] + 
#                   np.random.normal(0, 0.5, n_samples))
    
#     true_classification = {
#         'P1': 'parent_or_child',
#         'P2': 'parent_or_child', 
#         'S1': 'spouse',
#         'S2': 'spouse',
#         'C1': 'parent_or_child'
#     }
    
#     return data, true_classification

# def visualize_network(classification: Dict[str, str], T: str):
#     """
#     Visualize the discovered network structure
#     """
#     G = nx.DiGraph()
    
#     # Add nodes
#     G.add_node(T, color='lightblue')
#     for node, node_type in classification.items():
#         if node_type == 'spouse':
#             G.add_node(node, color='lightgreen')
#         else:
#             G.add_node(node, color='lightcoral')
    
#     # Add edges (simplified visualization)
#     for node, node_type in classification.items():
#         if node_type == 'parent_or_child':
#             if node.startswith('P'):
#                 G.add_edge(node, T)
#             else:
#                 G.add_edge(T, node)
    
#     # Draw the graph
#     pos = nx.spring_layout(G)
#     colors = [G.nodes[node]['color'] for node in G.nodes()]
#     plt.figure(figsize=(10, 8))
#     nx.draw(G, pos, node_color=colors, with_labels=True, 
#             node_size=1500, font_size=12, font_weight='bold')
#     plt.title('Discovered Network Structure')
#     plt.show()





def convert_names_to_indices(data: pd.DataFrame, name_mapping: Dict[str, List[str]]) -> Dict[int, List[int]]:
    """
    Convert a mapping of column names to a mapping of column indices.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input dataset with named columns.
    name_mapping : Dict[str, List[str]]
        A dictionary mapping column names to lists of their highly correlated column names.
    
    Returns
    -------
    Dict[int, List[int]]
        A dictionary mapping column indices to lists of their highly correlated column indices.
    """
    # Get column name to index mapping
    col_to_idx = {col: idx for idx, col in enumerate(data.columns)}
    
    # Convert name mapping to index mapping
    idx_mapping = {}
    for col_name, corr_cols in name_mapping.items():
        if col_name in col_to_idx:
            idx = col_to_idx[col_name]
            idx_mapping[idx] = [col_to_idx[c] for c in corr_cols if c in col_to_idx]
    
    return idx_mapping


def convert_indices_to_names(data: pd.DataFrame, idx_mapping: Dict[int, List[int]]) -> Dict[str, List[str]]:
    """
    Convert a mapping of column indices to a mapping of column names.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input dataset with named columns.
    idx_mapping : Dict[int, List[int]]
        A dictionary mapping column indices to lists of their highly correlated column indices.
    
    Returns
    -------
    Dict[str, List[str]]
        A dictionary mapping column names to lists of their highly correlated column names.
    """
    # Get index to column name mapping
    idx_to_col = {idx: col for idx, col in enumerate(data.columns)}
    
    # Convert index mapping to name mapping
    name_mapping = {}
    for idx, corr_idxs in idx_mapping.items():
        if idx in idx_to_col:
            col_name = idx_to_col[idx]
            name_mapping[col_name] = [idx_to_col[i] for i in corr_idxs if i in idx_to_col]
    
    return name_mapping


def remove_highly_correlated_features(data: pd.DataFrame, high_corr_feature_groups: Dict[str, List[str]]) -> Tuple[pd.DataFrame, Dict[int, List[int]], List[int]]:
    """
    Remove highly correlated features from the dataset based on the provided mapping.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input dataset with all features.
    high_corr_feature_groups : Dict[str, List[str]]
        A dictionary mapping column names to lists of their highly correlated column names.
        Example: {'X0': ['X5', 'X8']} means columns 'X5' and 'X8' are highly correlated with 'X0'.
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict[int, List[int]], List[int]]
        - The dataset with highly correlated features removed
        - The mapping with indices for the reduced dataset
        - List of original indices kept in the reduced dataset
    """
    if not high_corr_feature_groups:
        return data, {}, list(range(data.shape[1]))
    
    # Convert name-based mapping to index-based mapping
    idx_mapping = convert_names_to_indices(data, high_corr_feature_groups)
    
    # Identify all nodes to be removed
    nodes_to_remove = []
    for source_node, correlated_nodes in idx_mapping.items():
        nodes_to_remove.extend(correlated_nodes)
    
    # Create a list of nodes to keep
    all_nodes = list(range(data.shape[1]))
    nodes_to_keep = [node for node in all_nodes if node not in nodes_to_remove]
    
    # Create mapping from original indices to new indices in reduced dataset
    orig_to_reduced = {orig_idx: reduced_idx for reduced_idx, orig_idx in enumerate(nodes_to_keep)}
    
    # Create adjusted mapping for the reduced dataset
    adjusted_mapping = {}
    for source_node, correlated_nodes in idx_mapping.items():
        if source_node in nodes_to_keep:  # Only include if source node is kept
            adjusted_mapping[orig_to_reduced[source_node]] = correlated_nodes
    
    # Remove correlated features from the dataset
    reduced_data = data.iloc[:, nodes_to_keep]
    
    return reduced_data, adjusted_mapping, nodes_to_keep


def add_correlated_nodes_to_graph(graph: np.ndarray, correlated_nodes_map: Dict[str, List[str]], data: pd.DataFrame) -> np.ndarray:
    """
    Add highly correlated nodes back to the graph after causal discovery.
    
    This function takes a graph adjacency matrix and a mapping of nodes to their highly correlated nodes,
    then adds the correlated nodes back to the graph with the same relationships as their corresponding nodes.
    Additionally, it adds undirected edges between each source node and its correlated nodes.
    
    Parameters
    ----------
    graph : np.ndarray
        The adjacency matrix of shape (N, N) representing the causal graph.
        Edge types follow the convention in GraphEvaluator.EDGE_TYPES:
        - 0: no_edge
        - 1: directed (j->i)
        - 2: undirected
        - 3: bidirected
        - 4: partial_directed
        - 5: partial_undirected
        - 6: partial_unknown
    
    correlated_nodes_map : Dict[str, List[str]]
        A dictionary mapping column names to lists of their highly correlated column names
        that were removed before causal discovery.
        Example: {'X0': ['X5', 'X8']} means columns 'X5' and 'X8' were highly correlated with 'X0'.
    
    data : pd.DataFrame
        The original dataset with all features, used to map column names to indices.
    
    Returns
    -------
    np.ndarray
        The expanded adjacency matrix including the correlated nodes with the same relationships.
    
    Examples
    --------
    >>> graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # X0->X1->X2
    >>> correlated_map = {'X0': ['X3', 'X4'], 'X2': ['X5']}  # X3,X4 correlated with X0; X5 with X2
    >>> data = pd.DataFrame(np.random.randn(100, 6), columns=['X0', 'X1', 'X2', 'X3', 'X4', 'X5'])
    >>> expanded_graph = add_correlated_nodes_to_graph(graph, correlated_map, data)
    """
    if not correlated_nodes_map:
        return graph
    
    # Convert name-based mapping to index-based mapping
    idx_mapping = convert_names_to_indices(data, correlated_nodes_map)
    
    # Find the maximum node index in the correlated nodes map
    all_correlated_nodes = [node for sublist in idx_mapping.values() for node in sublist]
    if not all_correlated_nodes:
        return graph
    
    max_node_idx = max(max(all_correlated_nodes), graph.shape[0] - 1)
    
    # Create expanded graph with space for correlated nodes
    expanded_size = max_node_idx + 1
    expanded_graph = np.zeros((expanded_size, expanded_size), dtype=graph.dtype)
    
    # Copy original graph to expanded graph
    n = graph.shape[0]
    expanded_graph[:n, :n] = graph
    
    # Add correlated nodes with the same relationships
    for node_idx, correlated_indices in idx_mapping.items():
        if node_idx >= n:
            continue  # Skip if the reference node is outside the original graph
            
        node_name = data.columns[node_idx] if node_idx < len(data.columns) else f"X{node_idx}"
        
        for corr_idx in correlated_indices:
            corr_name = data.columns[corr_idx] if corr_idx < len(data.columns) else f"X{corr_idx}"
            
            # Copy incoming edges (parents)
            for i in range(n):
                if graph[i, node_idx] != 0:  # If i has an edge to node_idx
                    expanded_graph[i, corr_idx] = graph[i, node_idx]
                    
            # Copy outgoing edges (children)
            for j in range(n):
                if graph[node_idx, j] != 0:  # If node_idx has an edge to j
                    expanded_graph[corr_idx, j] = graph[node_idx, j]
            
            # If there are bidirectional or undirected edges, copy those too
            for k in range(n):
                if k != node_idx:
                    # Check for bidirectional edges (both i->j and j->i exist)
                    if graph[node_idx, k] != 0 and graph[k, node_idx] != 0:
                        expanded_graph[corr_idx, k] = graph[node_idx, k]
                        expanded_graph[k, corr_idx] = graph[k, node_idx]
            
            # Add undirected edge between source node and its correlated node
            expanded_graph[node_idx, corr_idx] = 2
            expanded_graph[corr_idx, node_idx] = 0  # Following the convention where only one side is marked
    
    return expanded_graph


def restore_original_node_indices(reduced_graph: np.ndarray, original_indices: List[int], correlated_nodes_map: Dict[int, List[int]]) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Restore the original node indices in the graph and correlated nodes map.
    
    Parameters
    ----------
    reduced_graph : np.ndarray
        The graph adjacency matrix with reduced indices.
    original_indices : List[int]
        The list of original indices that were kept in the reduced dataset.
    correlated_nodes_map : Dict[int, List[int]]
        A dictionary mapping reduced node indices to lists of original indices of their highly correlated nodes.
    
    Returns
    -------
    Tuple[np.ndarray, Dict[int, List[int]]]
        - The graph with original node indices
        - The correlated nodes map with original indices
    """
    if not original_indices:
        return reduced_graph, correlated_nodes_map
    
    # Create mapping from reduced indices to original indices
    reduced_to_orig = {i: original_indices[i] for i in range(len(original_indices))}
    
    # Create corrected mapping with original indices
    corrected_map = {}
    for reduced_idx, corr_nodes in correlated_nodes_map.items():
        if reduced_idx < len(original_indices):
            orig_idx = reduced_to_orig[reduced_idx]
            corrected_map[orig_idx] = corr_nodes
    
    return reduced_graph, corrected_map

def test_full_pipeline_with_names():
    """
    Test the full pipeline with name-based mapping.
    """
    print("\nTesting full pipeline with name-based mapping...")
    
    # Create a synthetic dataset with 6 features and custom column names
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 6)
    column_names = ['Income', 'Age', 'Education', 'Earnings', 'SchoolYears', 'Salary']
    data = pd.DataFrame(X, columns=column_names)
    
    # Make Earnings, Salary correlated with Income, and SchoolYears correlated with Education
    data['Earnings'] = data['Income'] + 0.1 * np.random.randn(n_samples)
    data['Salary'] = data['Income'] + 0.2 * np.random.randn(n_samples)
    data['SchoolYears'] = data['Education'] + 0.1 * np.random.randn(n_samples)
    
    # Define high correlation mapping using column names
    high_corr_feature_groups = {
        'Income': ['Earnings', 'Salary'],      # Income is correlated with Earnings and Salary
        'Education': ['SchoolYears']           # Education is correlated with SchoolYears
    }
    
    print(f"Original data columns: {data.columns.tolist()}")
    print(f"High correlation mapping: {high_corr_feature_groups}")
    
    # Step 1: Remove highly correlated features
    reduced_data, adjusted_mapping, original_indices = remove_highly_correlated_features(data, high_corr_feature_groups)
    
    print(f"Reduced data columns: {reduced_data.columns.tolist()}")
    
    # Step 2: Simulate running a causal discovery algorithm (create a simple graph)
    reduced_graph = np.zeros((3, 3))
    reduced_graph[0, 1] = 1  # Income -> Age
    reduced_graph[1, 2] = 1  # Age -> Education
    
    print("\nReduced graph (from causal discovery):")
    print(reduced_graph)
    
    # Step 3: Add back the highly correlated features
    final_graph = add_correlated_nodes_to_graph(reduced_graph, high_corr_feature_groups, data)
    
    print("\nFinal graph with correlated features added back:")
    print(final_graph)
    
    # Verify results
    assert final_graph.shape == (6, 6), "Final graph should be 6x6"
    
    # Get indices for each column
    col_to_idx = {col: idx for idx, col in enumerate(data.columns)}
    
    # Check parent-child relationships
    Income_idx = col_to_idx['Income']
    Age_idx = col_to_idx['Age']
    Education_idx = col_to_idx['Education']
    Earnings_idx = col_to_idx['Earnings']
    Salary_idx = col_to_idx['Salary']
    SchoolYears_idx = col_to_idx['SchoolYears']
    
    assert final_graph[Income_idx, Age_idx] == 1, "Income should have directed edge to Age"
    assert final_graph[Age_idx, Education_idx] == 1, "Age should have directed edge to Education"
    assert final_graph[Earnings_idx, Age_idx] == 1, "Earnings should have same edge as Income"
    assert final_graph[Salary_idx, Age_idx] == 1, "Salary should have same edge as Income"
    
    # Check undirected edges between source and correlated nodes
    assert final_graph[Income_idx, Earnings_idx] == 2, "Income and Earnings should have undirected edge"
    assert final_graph[Income_idx, Salary_idx] == 2, "Income and Salary should have undirected edge"
    assert final_graph[Education_idx, SchoolYears_idx] == 2, "Education and SchoolYears should have undirected edge"
    
    print("Full pipeline with name-based mapping test passed!")
    return True


if __name__ == "__main__":
    """
    Run all tests to verify the functionality of the highly correlated feature handling.
    """
    print("Running tests for highly correlated feature handling...")
    test_full_pipeline_with_names()
    print("\nAll tests passed!")

