import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class Visualization(object):
    def __init__(self, global_state, args, threshold: float=0.95):
        """
        :param global_state: a dict containing global variables and information
        :param args: arguments for the report generation
        :param threshold: threshold for the bootstrap probability to accept an edge.
        """
        self.data = global_state.user_data.raw_data
        self.bootstrap_prob = global_state.results.bootstrap_probability
        #self.y = y
        self.save_dir = args.output_graph_dir
        self.threshold = threshold

    def convert_mat(self, full_graph: np.array):
            converted_graph = full_graph.T
            return converted_graph

    def process_boot_mat(self, boot_prob_mat: np.array, full_graph: np.array):
        # causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
        boot_prob_mat = boot_prob_mat.T
        np.fill_diagonal(boot_prob_mat, 0)
        # Get the probability dictionary
        boot_dict = {index: value for index, value in np.ndenumerate(boot_prob_mat)}

        return boot_dict


    def mat_to_graph(self, full_graph: np.array, ori_graph: np.array = None,
                     edge_labels: str = None, title: str = None):
            '''
            :param full_graph: An adjacent matrix in Numpy Ndarray format -
                               causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
            :param ori_graph: It's used when we need to plot the new graph based on the position of the original graph
            :param edge_labels: Edge labels in a dictionary of labels keyed by edge two-tuple.
                                Only labels for the keys in the dictionary are drawn.
            :param title: graph title.
            :return: Path of the output graph
            '''

            # Convert adjacent matrix into graph
            converted_graph = self.convert_mat(full_graph)
            G = nx.from_numpy_array(converted_graph, parallel_edges=False, create_using=nx.DiGraph)
            # Dict of node labels
            variables = self.data.columns[:full_graph.shape[0]]
            labels = {i: variables[i] for i in range(len(variables))}

            plt.subplots(figsize=(8, 8))
            plt.axis("off")
            if ori_graph is None:
                nx.draw(G,
                        with_labels=True,
                        pos=nx.nx_pydot.graphviz_layout(G),
                        ###node###
                        node_shape="o",
                        node_size=1500,
                        linewidths=3,
                        edgecolors="#4a90e2d9",
                        #node_color=['#b45b1f' if str(variable) in self.y else '#1f78b4' for variable in labels.values()],
                        ###edge###
                        edge_color="gray",
                        width=2,
                        ###labels###
                        labels=labels,
                        font_weight="bold",
                        font_family="Helvetica",
                        font_size=12
                        )
            else:
                converted_graph_ori = self.convert_mat(ori_graph)
                G_ori = nx.from_numpy_array(converted_graph_ori, parallel_edges=False, create_using=nx.DiGraph)
                nx.draw_networkx_nodes(G_ori.copy(),
                        pos=nx.nx_pydot.graphviz_layout(G_ori),
                        ###node###
                        node_shape="o",
                        node_size=1500,
                        linewidths=3,
                        edgecolors="#4a90e2d9",
                        #node_color=['#b45b1f' if str(variable) in self.y else '#1f78b4' for variable in labels.values()]
                        )
                nx.draw_networkx_labels(G_ori.copy(),
                                       pos=nx.nx_pydot.graphviz_layout(G_ori),
                                       ###labels###
                                       labels=labels,
                                       font_weight="bold",
                                       font_family="Helvetica",
                                       font_size=12
                                       )
                nx.draw_networkx_edges(G.copy(),
                                        pos=nx.spectral_layout(G_ori),
                                        ###edge###
                                        edge_color="gray",
                                        width=2,
                                       node_size=1500
                                        )

            # if edge_labels is not None:
            #     force_threshold = self.threshold
            #     forbid_threshold = 1-self.threshold
            #     boot_dict_exist = {key:value for key, value in edge_labels.items() if (value >= force_threshold or key in G.edges)}
            #     boot_dict_delete = {key:value for key, value in edge_labels.items() if (value < forbid_threshold and key in G.edges)}
            #     nx.draw_networkx_edge_labels(G.copy(),
            #                                  pos=nx.spectral_layout(G),
            #                                  edge_labels=boot_dict_exist,
            #                                  font_size=10,
            #                                  font_weight="bold",
            #                                  verticalalignment='baseline',
            #                                  bbox=dict(boxstyle='square',
            #                                  ec=(1.0, 1.0, 1.0, 0),
            #                                  fc=(1.0, 1.0, 1.0, 0))
            #                                  )
            #     nx.draw_networkx_edge_labels(G.copy(),
            #                                  pos=nx.spectral_layout(G),
            #                                  edge_labels=boot_dict_delete,
            #                                  font_size=10,
            #                                  font_weight="bold",
            #                                  font_color='r',
            #                                  verticalalignment='baseline',
            #                                  bbox=dict(boxstyle='square',
            #                                            ec=(1.0, 1.0, 1.0, 0),
            #                                            fc=(1.0, 1.0, 1.0, 0))
            #                                  )
            #ax.set_title(title)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            save_path = os.path.join(self.save_dir, f'{title.replace(" ", "_")}.jpg')
            plt.savefig(fname=save_path, dpi=1000)

            return save_path

    def boot_heatmap_plot(self):
        # Create a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.bootstrap_prob, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8},
                    xticklabels=self.data.columns,
                    yticklabels=self.data.columns)
        plt.title('Confidence Heatmap')
        # Save the plot
        save_path_conf = os.path.join(self.save_dir, 'confidence_heatmap.jpg')
        plt.savefig(fname=save_path_conf, dpi=1000)

        return save_path_conf

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
