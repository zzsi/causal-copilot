from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.DAG2CPDAG import dag2cpdag

def array2cpdag(adj_array, node_names):
    # for methods return cpdag
    g = GeneralGraph([])
    node_map = {}
    num_nodes = adj_array.shape[0]
    
    # Create nodes
    for i in range(num_nodes):
        node_name = node_names[i]
        node_map[node_name] = GraphNode(node_name)
        g.add_node(node_map[node_name])
    
    # Create edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_array[i, j] == 1:
                node1 = node_map[node_names[i]]
                node2 = node_map[node_names[j]]
                edge = Edge(node1, node2, Endpoint.TAIL, Endpoint.ARROW)
                g.add_edge(edge)
    
    truth_cpdag = dag2cpdag(g)
    return truth_cpdag


class Judge(object):
    def __init__(self, global_state, args):
        self.global_state = global_state
        self.args = args

    def quality_judge(self, data, full_graph, algorithm, hyperparameters, knowledge_docs, boot_num):
        '''
        :param data: Given Tabular Data in Pandas DataFrame format
        :param full_graph: An adjacent matrix in Numpy Ndarray format -
                           causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
        :param algorithm: String representing the algorithm name
        :param hyperparameters: Dictionary of hyperparameter names and values
        :param knowledge_docs: A doc containing all necessary domain knowledge information from GPT-4.
        :param boot_num: Number of bootstrap iterations..
        :return: obvious errors in causal analysis results,
                 bootstrap probability of directed edges,
                 revised causal graph based on errors.
        '''

        from postprocess.judge_functions import bootstrap, llm_evaluation

        # Statistics Perspective: Bootstrapping to get probability of edges using the selected algorithm.
        errors_stat, boot_probability = bootstrap(data=data, full_graph=full_graph, algorithm=algorithm, hyperparameters=hyperparameters,
                                                  boot_num=boot_num, ts=False, parallel=self.args.parallel)
        print("Errors from Bootstrap method: ", errors_stat)
        print("Bootstrap Probability: ", boot_probability)

        # LLM perspective: errors based on domain knowledge from GPT-4
        if len(knowledge_docs) == 0 or "no knowledge" in knowledge_docs[0].lower():
            conversation, errors_llm = {}, {}
            print("No Errors are found by LLM, due to No Knowledge")
        else:
            conversation, errors_llm = llm_evaluation(data=data, full_graph=full_graph, args=self.args, knowledge_docs=knowledge_docs)
            print("Errors from LLMs: ", errors_llm)

        # Combine error obtained from both statistics and LLM perspectives
        errors = {**errors_stat, **errors_llm}

        # Revise causal graph based on errors
        revised_graph = full_graph

        for key in errors.keys():
            # i -> j
            split_key = key.split("->")
            i = data.columns.get_loc(split_key[0])
            j = data.columns.get_loc(split_key[1])

            if errors[key] == "Forced":
                revised_graph[j, i] = 1

            if errors[key] == "Forbidden":
                revised_graph[j, i] = 0

        ###### New Version Revision ######
        from postprocess.judge_functions import llm_direction
        llm_directions, revised_graph = llm_direction(self.global_state, self.args)

        return conversation, errors_llm, errors_stat, boot_probability, revised_graph, llm_directions


    def forward(self, global_state):
        (conversation,
         global_state.results.llm_errors,
         global_state.results.bootstrap_errors,
         global_state.results.bootstrap_probability,
         global_state.results.revised_graph,
         global_state.results.llm_directions) = self.quality_judge(
            data=global_state.user_data.processed_data,
            full_graph=global_state.results.converted_graph,
            algorithm=global_state.algorithm.selected_algorithm,
            hyperparameters=global_state.algorithm.algorithm_arguments,
            knowledge_docs=global_state.user_data.knowledge_docs,
            boot_num=global_state.statistics.boot_num
        )
        global_state.logging.knowledge_conversation.append(conversation)
        return global_state


    def evaluation(self, global_state):
        '''
        :param est_graph: estimated adjacent matrix of causal graph in Panda Ndarray format
        :param ground_truth: ground truth, represented by adjacent matrix in Panda Ndarray format - Matrix[i,j] indicates j->i
        :return: Structural Hamming Distance, precision, recall, F1 score.
        '''

        import numpy as np
        from sklearn.metrics import precision_score, recall_score, f1_score
        from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
        from causallearn.graph.SHD import SHD

        if global_state.algorithm.selected_algorithm in ['PC', 'FCI', 'GES', 'CDNOD']:
            print('Selected Algorithm: ', global_state.algorithm.selected_algorithm)
            if global_state.algorithm.selected_algorithm == 'PC':
                est_graph = global_state.results.raw_result.G
            elif global_state.algorithm.selected_algorithm == 'CDNOD':
                est_graph = global_state.results.raw_result.G
                # remove the domain index node
            elif global_state.algorithm.selected_algorithm == 'GES':
                est_graph = global_state.results.raw_result['G']
            elif global_state.algorithm.selected_algorithm == 'FCI':
                # TODO: improve for better handling edge o-o, o->, o-, currently ignore this part
                est_graph = global_state.results.raw_result[0]

            if global_state.statistics.domain_index != None:
                est_graph.remove_node(list(est_graph.node_map.keys())[-1])
            ground_truth = array2cpdag(global_state.user_data.ground_truth.transpose(), 
                                       node_names=global_state.user_data.processed_data.columns)
            shd = SHD(ground_truth, est_graph).get_shd()
            adj = AdjacencyConfusion(ground_truth, est_graph)
            precision = adj.get_adj_precision()
            recall = adj.get_adj_recall()
        else:
            ground_truth_flat = global_state.user_data.ground_truth.flatten()  
            est_graph_flat = global_state.results.converted_graph.flatten()
            shd = np.sum(np.abs(ground_truth_flat - est_graph_flat))

            adj_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

            for truth, est in zip(ground_truth_flat, est_graph_flat):
                if truth == 1 and est == 0:
                    adj_metrics['fn'] += 1
                elif truth == 0 and est == 1:
                    adj_metrics['fp'] += 1
                elif truth == 1 and est == 1:
                    adj_metrics['tp'] += 1
                else:
                    adj_metrics['tn'] += 1

            precision = adj_metrics['tp'] / (adj_metrics['tp'] + adj_metrics['fp']) if (adj_metrics['tp'] + adj_metrics['fp']) > 0 else 0
            recall = adj_metrics['tp'] / (adj_metrics['tp'] + adj_metrics['fn']) if (adj_metrics['tp'] + adj_metrics['fn']) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {'shd': shd, 'precision': precision, 'recall': recall, 'f1': f1}




