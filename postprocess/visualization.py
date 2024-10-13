import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

class Visualization(object):
    def __init__(self, y, save_dir, threshold=0.95):
        """
        :param y: Name of the result variable y.
        :param save_dir: the save directory of the output graph.
        :param threshold: threshold for the bootstrap probability to accept an edge.
        """
        self.y = y
        self.threshold = threshold
        self.save_dir = save_dir

    def convert_mat(self, full_graph: np.array):
            converted_graph = full_graph.T
            return converted_graph

    def process_boot_mat(self, boot_prob_mat: np.array):
        # causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
        boot_prob_mat = boot_prob_mat.T
        # Get the indices of non-zero elements
        non_zero_indices = np.nonzero(boot_prob_mat)
        # Get the values of non-zero elements
        non_zero_values = boot_prob_mat[non_zero_indices]
        # Get the probability dictionary
        boot_dict = {}
        for index, value in zip(zip(*non_zero_indices), non_zero_values):
            boot_dict[index] = value
        return boot_dict


    def mat_to_graph(self, full_graph: np.array, ori_graph: np.array = None,
                     labels: str = None, edge_labels: str = None, title: str = None):
            '''
            :param full_graph: An adjacent matrix in Numpy Ndarray format -
                               causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
            :param ori_graph: It's used when we need to plot the new graph based on the position of the original graph
            :param labels: Node labels in a dictionary of text labels keyed by node.
            :param edge_labels: Edge labels in a dictionary of labels keyed by edge two-tuple.
                                Only labels for the keys in the dictionary are drawn.
            :param title: graph title.
            :return: Path of the output graph
            '''

            converted_graph = self.convert_mat(full_graph)
            G = nx.from_numpy_array(converted_graph, parallel_edges=False, create_using=nx.DiGraph)
            if labels is None:
                    labels = {n:n for n in G}

            plt.subplots(figsize=(8, 8))
            plt.axis("off")
            if ori_graph is None:
                nx.draw(G,
                        with_labels=True,
                        pos=nx.planar_layout(G),
                        ###node###
                        node_shape="o",
                        node_size=1000,
                        linewidths=3,
                        edgecolors="#4a90e2d9",
                        node_color=['#b45b1f' if str(node)==str(self.y) else '#1f78b4' for node in G.nodes],
                        ###edge###
                        edge_color="gray",
                        width=2,
                        ###labels###
                        labels=labels,
                        font_weight="bold",
                        font_family="Helvetica",
                        font_size=11
                        )
            else:
                converted_graph_ori = self.convert_mat(ori_graph)
                G_ori = nx.from_numpy_array(converted_graph_ori, parallel_edges=False, create_using=nx.DiGraph)
                nx.draw_networkx_nodes(G_ori,
                        pos=nx.planar_layout(G_ori),
                        ###node###
                        node_shape="o",
                        node_size=1000,
                        linewidths=3,
                        edgecolors="#4a90e2d9",
                        node_color=['#b45b1f' if str(node) == str(self.y) else '#1f78b4' for node in G.nodes]
                        )
                nx.draw_networkx_labels(G_ori,
                                       pos=nx.planar_layout(G_ori),
                                       ###labels###
                                       labels=labels,
                                       font_weight="bold",
                                       font_family="Helvetica",
                                       font_size=11
                                       )
                nx.draw_networkx_edges(G,
                                        pos=nx.planar_layout(G_ori),
                                        ###edge###
                                        edge_color="gray",
                                        width=2,
                                       node_size=1000
                                        )

            if edge_labels is not None:
                boot_dict_exist = {key:value for key, value in edge_labels.items() if value >= self.threshold}
                boot_dict_delete = {key:value for key, value in edge_labels.items() if value < self.threshold}
                nx.draw_networkx_edge_labels(G,
                                             pos=nx.planar_layout(G),
                                             edge_labels=boot_dict_exist,
                                             font_size=10,
                                             font_weight="bold",
                                             verticalalignment='baseline',
                                             bbox=dict(boxstyle='square',
                                             ec=(1.0, 1.0, 1.0, 0),
                                             fc=(1.0, 1.0, 1.0, 0))
                                             )
                nx.draw_networkx_edge_labels(G,
                                             pos=nx.planar_layout(G),
                                             edge_labels=boot_dict_delete,
                                             font_size=10,
                                             font_weight="bold",
                                             font_color='r',
                                             verticalalignment='baseline',
                                             bbox=dict(boxstyle='square',
                                                       ec=(1.0, 1.0, 1.0, 0),
                                                       fc=(1.0, 1.0, 1.0, 0))
                                             )
            #ax.set_title(title)
            save_path = os.path.join(self.save_dir, f'{title.replace(" ", "_")}.jpg')
            plt.savefig(fname=save_path, dpi=1000)

            return save_path




true_mat = np.load('postprocess/test_data/20241007_184921_base_nodes8_samples1500/base_graph.npy')
result_mat = np.load('postprocess/test_data/20241007_184921_base_nodes8_samples1500/initial_graph.npy')
boot_probability = np.load('postprocess/test_data/20241007_184921_base_nodes8_samples1500/boot_prob.npy')
revised_mat = np.load('postprocess/test_data/20241007_184921_base_nodes8_samples1500/revised_graph.npy')

my_visual = Visualization(y='0', save_dir='postprocess/test_data/20241007_184921_base_nodes8_samples1500/output_graph', threshold=0.95)
true_fig_path = my_visual.mat_to_graph(full_graph=true_mat,
                                 labels=None,
                                 edge_labels=None,
                                 title='True Graph')

boot_dict = my_visual.process_boot_mat(boot_probability)
result_fig_path = my_visual.mat_to_graph(full_graph=result_mat,
                                 labels=None,
                                 edge_labels=boot_dict,
                                 title='Initial Graph')

revised_fig_path = my_visual.mat_to_graph(full_graph=revised_mat,
                                ori_graph=result_mat,
                                 labels=None,
                                 edge_labels=None,
                                 title='Revised Graph')





