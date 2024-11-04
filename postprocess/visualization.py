import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pywhy_graphs import PAG
from pywhy_graphs.viz import draw
from causallearn.search.FCMBased.lingam.utils import make_dot

class Visualization(object):
    def __init__(self, global_state, threshold: float=0.95):
        """
        :param global_state: a dict containing global variables and information
        :param args: arguments for the report generation
        :param threshold: threshold for the bootstrap probability to accept an edge.
        """
        self.global_state = global_state
        self.data = global_state.user_data.raw_data
        self.bootstrap_prob = global_state.results.bootstrap_probability
        self.save_dir = global_state.user_data.output_graph_dir
        self.threshold = threshold

    def convert_to_edges_truth(self, mat):
        variables = self.data.columns
        labels = {i: variables[i] for i in range(len(variables))}
        certain_edges = []
        bi_certain_edges = []
        indices = np.where(mat == 1)
        for i, j in zip(indices[0], indices[1]):
            if (mat[j, i] == 0) or (mat[j, i] == -1):
                certain_edges.append((j, i))
            else:
                bi_certain_edges.append((j, i))
        certain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in certain_edges]
        bi_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in bi_certain_edges]
        edges_dict = {
            'all_edges': [],
            'certain_edges': certain_edges_names,
            'uncertain_edges': [],
            'bi_edges': bi_edges_names,
            'half_edges': [],
            'none_edges': []
        }
        
        return edges_dict 

    def convert_to_edges(self, g):
        if isinstance(g, np.ndarray):
            adj_matrix = g
        else:
            if self.global_state.algorithm.selected_algorithm == 'FCI':
                g = g[0]
            elif self.global_state.algorithm.selected_algorithm == 'GES':
                g = g['G']
            try:
                adj_matrix = g.graph
            except:
                adj_matrix = g.G.graph
        #print('adj_matrix:', adj_matrix)

        variables = self.data.columns
        labels = {i: variables[i] for i in range(len(variables))}

        certain_edges = [] # ->
        uncertain_edges = []  # -
        bi_edges = []    #<->
        half_edges = []  # o->
        none_edges = []  # o-o

        indices = np.where(adj_matrix == 1)

        for i, j in zip(indices[0], indices[1]):
            # save the determined edges (j -> i)
            if adj_matrix[j, i] == -1:
                certain_edges.append((j, i))
            # save the bidirected edges (j <-> i)
            elif adj_matrix[j, i] == 1:
                bi_edges.append((j, i))
            # save the half determined edges (j o-> i)
            elif adj_matrix[j, i] == 2:
                half_edges.append((j, i))
        indices = np.where(adj_matrix == 2)
        for i, j in zip(indices[0], indices[1]):
            # save the non determined edges (j o-o i)
            if adj_matrix[j, i] == 2:
                none_edges.append((j, i))
        indices = np.where(adj_matrix == -1)
        for i, j in zip(indices[0], indices[1]):
            # save the uncertain edges (j - i)
            if adj_matrix[j, i] == -1:
                uncertain_edges.append((j, i))

        uncertain_edges = list({tuple(sorted(t)) for t in uncertain_edges})
        none_edges = list({tuple(sorted(t)) for t in none_edges})
        all_edges = certain_edges.copy() + uncertain_edges.copy() + bi_edges.copy() + half_edges.copy() + none_edges.copy()

        all_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in all_edges]
        certain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in certain_edges]
        uncertain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in uncertain_edges]
        bi_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in bi_edges]
        half_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in half_edges]
        none_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in none_edges]
        edges_dict = {
            'all_edges': all_edges_names,
            'certain_edges': certain_edges_names,
            'uncertain_edges': uncertain_edges_names,
            'bi_edges': bi_edges_names,
            'half_edges': half_edges_names,
            'none_edges': none_edges_names
        }
        return edges_dict

    # def process_boot_mat(self, boot_prob_mat: np.array, full_graph: np.array):
    #     # causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
    #     boot_prob_mat = boot_prob_mat.T
    #     np.fill_diagonal(boot_prob_mat, 0)
    #     # Get the probability dictionary
    #     boot_dict = {index: value for index, value in np.ndenumerate(boot_prob_mat)}

    #     return boot_dict

    def plot_pdag(self, g, save_path, pos=None, relation=False):
        algo = self.global_state.algorithm.selected_algorithm
        path = os.path.join(self.save_dir, save_path)

        if algo in ['PC', 'FCI', 'CDNOD', 'GES'] or relation:
            if isinstance(g, np.ndarray):
                edges_dict = self.convert_to_edges_truth(g)
            else:
                edges_dict = self.convert_to_edges(g)
            pag = PAG()
            for edge in edges_dict['certain_edges']:
                pag.add_edge(edge[0], edge[1], pag.directed_edge_name)
            for edge in edges_dict['uncertain_edges']:
                pag.add_edge(edge[0], edge[1], pag.undirected_edge_name)
            for edge in edges_dict['bi_edges']:
                pag.add_edge(edge[0], edge[1], pag.bidirected_edge_name)
            for edge in edges_dict['half_edges']:
                pag.add_edge(edge[0], edge[1], pag.directed_edge_name)
                pag.add_edge(edge[1], edge[0], pag.circle_edge_name)
            for edge in edges_dict['none_edges']:
                pag.add_edge(edge[0], edge[1], pag.circle_edge_name)
                pag.add_edge(edge[1], edge[0], pag.circle_edge_name)

            if pos is None:
                pos_G = nx.spring_layout(pag)
                
                dot_graph = draw(pag, pos=pos_G, shape='circle')                
            else:
                dot_graph = draw(pag, pos=pos, shape='circle')
                pos_G = None
            dot_graph.render(outfile=path, cleanup=True)
            
            return pos_G
            
        elif algo in ['DirectLiNGAM', 'ICALiNGAM', 'NOTEARS']:
            labels = [f'{col}' for i, col in enumerate(self.data.columns)]
            variables = self.data.columns
            if isinstance(g, np.ndarray):
                mat = g
            else:
                if algo in ['NOTEARS']:
                    mat = np.zeros((len(variables), len(variables)))
                    for parent, child in g.edges():
                        i = variables.get_loc(parent)
                        j = variables.get_loc(child) 
                        mat[j, i] = g.get_edge_data(parent, child)['weight']
                else:
                    mat = g.adjacency_matrix_
            pyd = make_dot(mat, labels=labels)  
            #pyd = make_dot_highlight(mat, labels=labels, cmap="cool", vmargin=2, hmargin=2)         
            pyd.render(outfile=path, cleanup=True)
            return None

    def boot_heatmap_plot(self):
        
        boot_prob_dict = {k: v for k, v in self.bootstrap_prob.items() if v is not None and sum(v.flatten())>0}
        name_map = {'certain_edges': 'Directed Edge', #(->)
                    'uncertain_edges': 'Undirected Edge', #(-)
                    'bi_edges': 'Bi-Directed Edge', #(<->)
                    'half_edges': 'Non-Ancestor Edge', #(o->)
                    'non_edges': 'No D-Seperation Edge', #(o-o)
                    'non_existence':'No Edge'}
        paths = []
        for key in boot_prob_dict.keys():
            prob_mat = boot_prob_dict[key]
            name = name_map[key]
            # Create a heatmap
            plt.figure(figsize=(8, 6))
            plt.rcParams['font.family'] = 'Times New Roman'
            sns.heatmap(prob_mat, annot=True, cmap='Reds', fmt=".2f", square=True, cbar_kws={"shrink": .8},
                        xticklabels=self.data.columns,
                        yticklabels=self.data.columns)            
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
