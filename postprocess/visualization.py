import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pywhy_graphs import PAG
# from pywhy_graphs.viz import draw
import sys
# sys.path.append('causal-learn')
# from causallearn.search.FCMBased.lingam.utils import make_dot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocess.draw import draw


def get_layout(g):
    """Generate layout positions for a graph using a two-stage approach.
    
    First applies ForceAtlas2 for global structure, then refines with Fruchterman-Reingold.
    Falls back to spring_layout if fa2 is not available.
    
    Args:
        g: A networkx graph object
        
    Returns:
        dict: Node positions mapping
    """
    try:
        # First stage: ForceAtlas2 layout for initial global structure
        import fa2
        forceatlas2 = fa2.ForceAtlas2(
            # Behavior alternatives
            outboundAttractionDistribution=True,  # Dissuade hubs
            linLogMode=False,  # NOT Lin-log mode
            adjustSizes=False,  # Prevent overlap
            edgeWeightInfluence=1.0,
            
            # Performance
            jitterTolerance=1.0,  # Tolerance
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            
            # Tuning
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,
            
            # Log
            verbose=False
        )
        
        # Initial positions with ForceAtlas2
        initial_pos = forceatlas2.forceatlas2_networkx_layout(g, pos=None, iterations=500)
        
        # Second stage: Fruchterman-Reingold for fine-tuning
        pos = nx.fruchterman_reingold_layout(g, pos=initial_pos, iterations=50, seed=42)
    except ImportError:
        # Fallback if fa2 is not available
        pos = nx.spring_layout(g, seed=42)
        print("Warning: fa2 package not found. Using spring_layout instead.")
    return pos

class Visualization(object):
    def __init__(self, global_state, threshold: float=0.95):
        """
        :param global_state: a dict containing global variables and information
        :param args: arguments for the report generation
        :param threshold: threshold for the bootstrap probability to accept an edge.
        """
        self.global_state = global_state
        
        intersection_features = list(set(global_state.user_data.processed_data.columns).intersection(
            set(global_state.user_data.visual_selected_features)))
        self.data = global_state.user_data.processed_data[intersection_features]
        self.data_idx = [global_state.user_data.processed_data.columns.get_loc(var) for var in intersection_features]
        self.bootstrap_prob = global_state.results.bootstrap_probability
        self.save_dir = global_state.user_data.output_graph_dir
        self.threshold = threshold

    def get_pos(self, mat):
        """Get node positions for graph visualization."""
        adj_matrix = (mat != 0).astype(int).T
        g = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        # Relabel nodes with variable names from data columns
        mapping = {i: self.data.columns[i] for i in range(len(self.data.columns))}
        g = nx.relabel_nodes(g, mapping)
        
        # Get layout positions
        pos = get_layout(g)
        return pos

    def plot_pdag(self, mat, save_path, pos=None, relation=False):
        print(mat)
        algo = self.global_state.algorithm.selected_algorithm
        path = os.path.join(self.save_dir, save_path)
        data_idx = [self.global_state.user_data.processed_data.columns.get_loc(var) for var in self.global_state.user_data.visual_selected_features]
        mat = mat[data_idx, :][:, data_idx]
        edges_dict = convert_to_edges(algo, self.data.columns, mat)
        pag = PAG()
        for edge in edges_dict['certain_edges']:
            try:
                pag.add_edge(edge[0], edge[1], pag.directed_edge_name)
            except:
                pag.remove_edge(edge[1], edge[0], pag.directed_edge_name)
                pag.add_edge(edge[0], edge[1], pag.bidirected_edge_name)
        for edge in edges_dict['uncertain_edges']:
            pag.add_edge(edge[0], edge[1], pag.undirected_edge_name)
        for edge in edges_dict['bi_edges']:
            pag.add_edge(edge[0], edge[1], pag.bidirected_edge_name)
        for edge in edges_dict['half_certain_edges']:
            pag.add_edge(edge[0], edge[1], pag.directed_edge_name)
            pag.add_edge(edge[1], edge[0], pag.circle_edge_name)
        for edge in edges_dict['half_uncertain_edges']:
            pag.add_edge(edge[0], edge[1], pag.undirected_edge_name)
            pag.add_edge(edge[1], edge[0], pag.circle_edge_name)
        for edge in edges_dict['none_edges']:
            pag.add_edge(edge[0], edge[1], pag.circle_edge_name)
            pag.add_edge(edge[1], edge[0], pag.circle_edge_name)

        if pos is not None:
            dot_graph = draw(pag, full_node_names=list(pos.keys()), pos=pos, shape='circle')  
            pos_G = pos              
        else:
            dot_graph = draw(pag, shape='circle')
            pos_G = None
        dot_graph.render(outfile=path, cleanup=True)
        
        return pos_G
       
    def boot_heatmap_plot(self):
        
        boot_prob_dict = {k: v for k, v in self.bootstrap_prob.items() if v is not None and sum(v.flatten())>0}
        name_map = {'certain_edges': 'Directed Edge', #(->)
                    'uncertain_edges': 'Undirected Edge', #(-)
                    'bi_edges': 'Bi-Directed Edge', #(<->)
                    'half_certain_edges': 'Directed Non-Ancestor Edge', #(o->)
                    'half_uncertain_edges': 'Undirected Non-Ancestor Edge', #(o-)
                    'none_edges': 'No D-Seperation Edge', #(o-o)
                    'none_existence':'No Edge'}
        paths = []
        for key in boot_prob_dict.keys():
            prob_mat = boot_prob_dict[key]
            name = name_map[key]
            # Create a heatmap
            plt.figure(figsize=(8, 6))
            #plt.rcParams['font.family'] = 'Times New Roman'
            sns.heatmap(prob_mat[self.data_idx, :][:, self.data_idx], annot=True, cmap='Reds', fmt=".2f", square=True, cbar_kws={"shrink": .8},
                        xticklabels=self.global_state.user_data.visual_selected_features,
                        yticklabels=self.global_state.user_data.visual_selected_features)            
            plt.title(f'Confidence Heatmap for {name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            # Save the plot
            save_path_conf = os.path.join(self.save_dir, f'{key}_confidence_heatmap.jpg')
            plt.savefig(fname=save_path_conf, dpi=1000)
            paths.append(save_path_conf)

        return paths

    def metrics_plot(self, original_metrics, revised_metrics):
        # Sample data
        shd_revised = revised_metrics.pop('shd')
        shd_original = original_metrics.pop('shd')

        # Extract keys and values
        metrics = list(revised_metrics.keys())
        revised_values = list(revised_metrics.values())
        original_values = list(original_metrics.values())

        # Set the bar width
        bar_width = 0.35

        # Set position of bars on the X axis
        r1 = np.arange(len(metrics))
        r2 = [x + bar_width for x in r1]

        # Create the figure and the first axis
        fig, ax1 = plt.subplots()

        # Create bars for revised metrics on the first y-axis
        ax1.bar(r1, original_values, color='#FFB266', width=bar_width, label='Original Metrics')
        ax1.bar(r2, revised_values, color='#0080FF', width=bar_width, label='Revised Metrics')

        # Add labels and title for the first y-axis
        ax1.set_xlabel('Metrics', fontweight='bold')
        ax1.set_ylabel('Values (Precision, Recall, F1)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # Set y-axis limits for the first y-axis
        ax1.set_ylim(0, 1)

        # Create a second y-axis
        ax2 = ax1.twinx()

        # Plot the shd metric on the second y-axis
        ax2.bar(r1[-1] + 1, shd_original, color='#FFB266', width=bar_width, label='Original SHD')
        ax2.bar(r2[-1] + 1, shd_revised, color='#0080FF', width=bar_width, label='Revised SHD')

        # Set the second y-axis label
        ax2.set_ylabel('SHD Values', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Set x-ticks
        ax1.set_xticks([r + bar_width / 2 for r in range(len(metrics) + 1)])
        metrics.append('SHD')
        ax1.set_xticklabels(metrics)

        # Add legends
        ax1.legend(loc='upper left')

        # Set the title
        plt.title('Metrics Comparison of Original and Revised Graphs')

        # Show the plot
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'metrics.jpg')
        plt.savefig(fname=save_path, dpi=1000)

        return save_path

def convert_to_edges(algo, variables, mat):
    labels = {i: variables[i] for i in range(len(variables))}

    certain_edges = [] # ->
    uncertain_edges = []  # -
    bi_edges = []    #<->
    half_certain_edges = []  # o->
    half_uncertain_edges = []  # o-
    none_edges = []  # o-o

    for ind_i in range(mat.shape[0]):
        for ind_j in range(mat.shape[0]):
            if ind_i == ind_j: continue
            elif mat[ind_i, ind_j] == 1: 
                certain_edges.append((ind_j, ind_i))
            elif mat[ind_i, ind_j] == 2: 
                uncertain_edges.append((ind_i, ind_j))
            elif mat[ind_i, ind_j] == 3: 
                bi_edges.append((ind_i, ind_j))
            elif mat[ind_i, ind_j] == 4: 
                half_certain_edges.append((ind_j, ind_i))
            elif mat[ind_i, ind_j] == 5: 
                half_uncertain_edges.append((ind_j, ind_i))
            elif mat[ind_i, ind_j] == 6: 
                none_edges.append((ind_j, ind_i))

    uncertain_edges = list({tuple(sorted(t)) for t in uncertain_edges})
    none_edges = list({tuple(sorted(t)) for t in none_edges})
    all_edges = certain_edges.copy() + uncertain_edges.copy() + bi_edges.copy() + half_certain_edges.copy() + half_uncertain_edges.copy() + none_edges.copy()

    all_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in all_edges]
    certain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in certain_edges]
    uncertain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in uncertain_edges]
    bi_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in bi_edges]
    half_certain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in half_certain_edges]
    half_uncertain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in half_uncertain_edges]
    none_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in none_edges]
    edges_dict = {
        'all_edges': all_edges_names,
        'certain_edges': certain_edges_names,
        'uncertain_edges': uncertain_edges_names,
        'bi_edges': bi_edges_names,
        'half_certain_edges': half_certain_edges_names,
        'half_uncertain_edges': half_uncertain_edges_names,
        'none_edges': none_edges_names
    }
    return edges_dict
   
def test_fixed_pos():
    # Create a DAG (Directed Acyclic Graph)
    n_nodes = 50
    # Initialize with zeros
    dag = np.zeros((n_nodes, n_nodes))
    
    # Create a simple chain structure: 0->1->2->3->4
    for i in range(n_nodes-1):
        dag[i, i+1] = 1
    
    # Add some additional edges to make it more complex
    # But ensure it remains acyclic (only connect from lower to higher indices)
    dag[0, 2] = 1  # 0->2
    dag[1, 3] = 1  # 1->3
    dag[0, 4] = 1  # 0->4
    
    # Create a sparser version of the DAG by removing some edges
    sparse_dag = dag.copy()
    # Remove edge 0->2
    sparse_dag[0, 2] = 0
    # Remove edge 1->3
    sparse_dag[1, 3] = 0

    # Use spring layout to get positions
    import networkx as nx
    G_full = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    G_sparse = nx.from_numpy_array(sparse_dag, create_using=nx.DiGraph)
    
    pos_full = get_layout(G_full)
    pos_sparse = get_layout(G_sparse)

    # Plot using the visualization functions
    draw(G_full, pos=pos_full, full_node_names=list(pos_full.keys()), shape='circle').render(outfile='fully_connected.pdf', cleanup=True)
    draw(G_sparse, pos=pos_sparse, full_node_names=list(pos_sparse.keys()), shape='circle').render(outfile='sparse_dag.pdf', cleanup=True)
    

if __name__ == '__main__':
    test_fixed_pos()
    # my_visual_initial = Visualization(global_state)
    # if global_state.results.raw_pos is None:
    #     data_idx = [global_state.user_data.processed_data.columns.get_loc(var) for var in global_state.user_data.visual_selected_features]
    #     pos = my_visual_initial.get_pos(global_state.results.converted_graph[data_idx, :][:, data_idx])
    #     global_state.results.raw_pos = pos
    # if global_state.user_data.ground_truth is not None:
    #     my_visual_initial.plot_pdag(global_state.user_data.ground_truth, f'{global_state.algorithm.selected_algorithm}_true_graph.jpg', global_state.results.raw_pos)
    #     my_visual_initial.plot_pdag(global_state.user_data.ground_truth, f'{global_state.algorithm.selected_algorithm}_true_graph.pdf', global_state.results.raw_pos)
    #     chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/{global_state.algorithm.selected_algorithm}_true_graph.jpg',)))
    #     yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
    # if global_state.results.converted_graph is not None:
    #     my_visual_initial.plot_pdag(global_state.results.converted_graph, f'{global_state.algorithm.selected_algorithm}_initial_graph.jpg', global_state.results.raw_pos)
    #     my_visual_initial.plot_pdag(global_state.results.converted_graph, f'{global_state.algorithm.selected_algorithm}_initial_graph.pdf', global_state.results.raw_pos)