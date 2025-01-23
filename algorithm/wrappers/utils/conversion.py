import numpy as np
import pandas as pd
from typing import Set, Tuple, List, Callable, Dict
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


def MB2CPDAG(data: pd.DataFrame, mb_dict: Dict[int, List[int]], is_discrete: bool = False, alpha: float = 0.05) -> np.ndarray:
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
    
    # Set independence test based on data type
    indep_test = 'gsq' if is_discrete else 'fisherz'
    
    # For each target and its MB nodes
    for target, mb_nodes in mb_dict.items():
        # Get indices for this local graph
        local_indices = [target] + mb_nodes
        
        if len(local_indices) < 2:
            continue
        if len(local_indices) == 2:
            merged_cpdag[local_indices[0], local_indices[1]] = 2
            continue

        # Extract relevant data columns
        local_data = data.iloc[:, local_indices].values
        node_names = [str(i) for i in local_indices]
        
        # Run PC algorithm on local graph
        cg = pc(local_data,
                alpha=alpha,
                indep_test=indep_test,
                node_names=node_names,
                verbose=False)
        
        print(cg.G.graph)
                
        # Convert PC result to adjacency matrix format
        local_adj = cg.G.graph
        
        # Map local results back to full adjacency matrix
        for i in range(len(local_indices)):
            for j in range(len(local_indices)):
                if local_adj[i,j] == 1:
                    if local_adj[j,i] == -1:
                        # Directed edge i->j
                        merged_cpdag[local_indices[i], local_indices[j]] = 1
                    elif local_adj[j,i] == 1:
                        # Undirected edge i--j
                        if merged_cpdag[local_indices[j], local_indices[i]] == 0:
                            merged_cpdag[local_indices[i], local_indices[j]] = 2
                elif local_adj[i,j] == -1 and local_adj[j,i] == -1:
                    # Undirected edge i--j
                    if merged_cpdag[local_indices[j], local_indices[i]] == 0:
                        merged_cpdag[local_indices[i], local_indices[j]] = 2
                        
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

def main():
    # Generate test data
    print("Generating test data...")
    data, true_classification = generate_test_data()
    
    # Initialize classifier with Fisher Z test
    classifier = MarkovBlanketClassifier(data, ci_test='fisherz')
    
    # Define target and Markov blanket nodes
    T = 'X'
    MB = ['P1', 'P2', 'S1', 'S2', 'C1']
    
    print("\nClassifying Markov blanket nodes...")
    classification = classifier.classify_blanket_nodes(T, MB)
    
    # Print results
    print("\nResults:")
    print(f"Discovered classification: {classification}")
    print(f"True classification: {true_classification}")
    
    # Calculate accuracy
    correct = sum(1 for node in MB if classification[node] == true_classification[node])
    accuracy = correct / len(MB)
    
    print(f"\nClassification accuracy: {accuracy:.2%}")
    
    # Visualize the network
    # visualize_network(classification, T)
if __name__ == "__main__":
    main()
