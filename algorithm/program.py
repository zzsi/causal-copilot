import json
import algorithm.wrappers as wrappers
from algorithm.wrappers.utils.tab_utils import remove_highly_correlated_features, add_correlated_nodes_to_graph, restore_original_node_indices

class Programming(object):
    def __init__(self, args):
        self.args = args

    def forward(self, global_state):
        # Process data to remove highly correlated features if they exist
        if global_state.user_data.high_corr_feature_groups is not None:
            # Remove highly correlated features before running algorithm
            reduced_data, adjusted_mapping, original_indices = remove_highly_correlated_features(
                global_state.user_data.processed_data, 
                global_state.user_data.high_corr_feature_groups
            )
            
            # Run algorithm on reduced dataset
            algo_func = getattr(wrappers, global_state.algorithm.selected_algorithm)
            graph, info, raw_result = algo_func(global_state.algorithm.algorithm_arguments).fit(reduced_data)
            
            # Restore original indices in the mapping if needed
            restored_graph, restored_mapping = restore_original_node_indices(
                graph, original_indices, adjusted_mapping
            )
            
            # Add back the highly correlated features to the graph
            final_graph = add_correlated_nodes_to_graph(
                restored_graph, 
                global_state.user_data.high_corr_feature_groups,
                global_state.user_data.processed_data
            )
            
            # Store original and expanded results
            global_state.results.raw_result = raw_result
            global_state.results.converted_graph = final_graph
            info['original_graph'] = graph  # Store the original graph before adding correlated nodes
            info['high_corr_features_removed'] = original_indices
            global_state.results.raw_info = info
        else:
            # Run algorithm on the full dataset
            algo_func = getattr(wrappers, global_state.algorithm.selected_algorithm)
            graph, info, raw_result = algo_func(global_state.algorithm.algorithm_arguments).fit(global_state.user_data.processed_data)
            
            global_state.results.raw_result = raw_result
            global_state.results.converted_graph = graph
            global_state.results.raw_info = info
            
        # Handle time-series specific data
        if global_state.statistics.data_type=="Time-series":
            if 'lag_matrix' in info:
                global_state.results.lagged_graph = info['lag_matrix']
            else:
                global_state.results.lagged_graph = None
                
        return global_state
