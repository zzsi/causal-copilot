import sys
sys.path.insert(0, 'causal-learn')

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

        from postprocess.judge_functions import bootstrap, llm_evaluation, llm_direction, llm_evaluation_justification

        # Statistics Perspective: Bootstrapping to get probability of edges using the selected algorithm.
        edge_recom, boot_probability = bootstrap(data=data, full_graph=full_graph, algorithm=algorithm, hyperparameters=hyperparameters,
                                                  boot_num=boot_num, ts=False, parallel=self.args.parallel)
        print("Edge Recommendations from Bootstrap method: ", edge_recom)
        #print("Bootstrap Probability: ", boot_probability)

        ############Edge Pruning with Bootstrap############
        print('Bootstrap Pruning Decisioning')
        revised_graph = full_graph.copy()
        fixed_pairs = []
        bootstrap_pruning_record = []
        for k in edge_recom.keys():
            (i, j) = tuple(map(int, k.split('-')))
            edge = edge_recom[k].split('(')[0]
            prob = float(edge_recom[k].split('(')[1].strip(')'))
            # Change edges according to recommendation if bootstrap probability>0.95
            if prob > 0.95:
                fixed_pairs.append((i,j))
                fixed_pairs.append((j,i))
                text = f'{data.columns[j]} {edge} {data.columns[i]}'
                bootstrap_pruning_record.append(text)
                if edge == '->':
                    revised_graph[i, j] = 1
                    revised_graph[j, i] = -1
                elif edge == '-':
                    revised_graph[i, j] = revised_graph[j, i] = -1
                elif edge == '<->':
                    revised_graph[i, j] = revised_graph[j, i] = 1
                elif edge == 'o->':
                    revised_graph[i, j] = 1
                    revised_graph[j, i] = 2
                elif edge == 'o-o':
                    revised_graph[i, j] = revised_graph[j, i] = 2
                else:
                    revised_graph[i, j] = revised_graph[j, i] = 0
        #print(bootstrap_pruning_record)
        ########################

        ############ Edge Pruning with LLM ############
        print('LLM Pruning Decisioning')
        force_ind, forbid_ind = llm_evaluation(data, self.args, knowledge_docs, self.global_state.results.converted_graph)
        llm_pruning_record = {}
        force_variables = []
        forbid_variables = []
        for force_pair in force_ind:
            i, j = force_pair[0], force_pair[1]
            # force it if it doesn't exist in original graph and not fixed by bootstrap
            if revised_graph[i, j]==0 and revised_graph[j, i]==0 and (i, j) not in fixed_pairs:
                revised_graph[i, j] = 1
                revised_graph[j, i] = -1
                force_variables.append((data.columns[j],data.columns[i]))
        json_forces = llm_evaluation_justification(self.args, knowledge_docs, force_variables, force=True)                                                               
        for forbid_pair in forbid_ind:
            i, j = forbid_pair[0], forbid_pair[1]
            # forbid it if it exists in original graph and not fixed by bootstrap
            if revised_graph[i, j]!=0 or revised_graph[j, i]!=0 and (i, j) not in fixed_pairs:
                revised_graph[i, j] = revised_graph[j, i] = 0
                forbid_variables.append((data.columns[j],data.columns[i]))
        #print('forbid_variables:', forbid_variables)
        json_forbids = llm_evaluation_justification(self.args, knowledge_docs, forbid_variables, force=False) 
        llm_pruning_record={
            'force_record': json_forces,
            'forbid_record': json_forbids
        }
        #print(llm_pruning_record)
        ########################
        
        ###### Edge Direction with LLM ######
        print('LLM Direction Decisioning')
        llm_directions_record, revised_graph = llm_direction(self.global_state, self.args, revised_graph)

        return {}, bootstrap_pruning_record, boot_probability, llm_pruning_record, llm_directions_record, revised_graph


    def forward(self, global_state):
        
        if self.global_state.algorithm.selected_algorithm in ['DirectLiNGAM', 'ICALiNGAM', 'NOTEARS']:
            adj_matrix = global_state.results.raw_result
        else:
            if self.global_state.algorithm.selected_algorithm == 'FCI':
                g = global_state.results.raw_result[0]
            elif self.global_state.algorithm.selected_algorithm == 'GES':
                g = global_state.results.raw_result['G']
            else:
                g = global_state.results.raw_result
            try:
                adj_matrix = g.graph
            except:
                adj_matrix = g.G.graph

        
        (conversation,
         global_state.results.bootstrap_errors,
         global_state.results.bootstrap_probability,
         global_state.results.llm_errors,
         global_state.results.llm_directions,
         global_state.results.revised_graph
        )= self.quality_judge(
        data=global_state.user_data.processed_data,
        full_graph=adj_matrix,
        algorithm=global_state.algorithm.selected_algorithm,
        hyperparameters=global_state.algorithm.algorithm_arguments,
        knowledge_docs=global_state.user_data.knowledge_docs,
        boot_num=global_state.statistics.boot_num
        )
        global_state.logging.knowledge_conversation.append(conversation)
        return global_state


    def evaluation(self, global_state, revise=False):
        '''
        :param est_graph: estimated adjacent matrix of causal graph in Panda Ndarray format
        :param ground_truth: ground truth, represented by adjacent matrix in Panda Ndarray format - Matrix[i,j] indicates j->i
        :return: Structural Hamming Distance, precision, recall, F1 score.
        '''

        import numpy as np
        from sklearn.metrics import precision_score, recall_score, f1_score
        from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
        from causallearn.graph.SHD import SHD

        if global_state.algorithm.selected_algorithm in ['PC', 'FCI', 'GES', 'CDNOD'] and not revise:
            print('Selected Algorithm: ', global_state.algorithm.selected_algorithm)
            if global_state.algorithm.selected_algorithm == 'PC':
                est_graph = global_state.results.raw_result.G
            elif global_state.algorithm.selected_algorithm == 'CDNOD':
                est_graph = global_state.results.raw_result.G
            elif global_state.algorithm.selected_algorithm == 'GES':
                est_graph = global_state.results.raw_result['G']
            elif global_state.algorithm.selected_algorithm == 'FCI':
                # TODO: improve for better handling edge o-o, o->, o-, currently ignore this part
                est_graph = global_state.results.raw_result[0]

            if global_state.statistics.domain_index is not None:
                est_graph.remove_node(list(est_graph.node_map.keys())[-1])
            ground_truth = array2cpdag(global_state.user_data.ground_truth.transpose(), 
                                       node_names=global_state.user_data.processed_data.columns)
            shd = SHD(ground_truth, est_graph).get_shd()
            adj = AdjacencyConfusion(ground_truth, est_graph)
            precision = adj.get_adj_precision()
            recall = adj.get_adj_recall()
        else:
            if global_state.statistics.domain_index is not None:
                global_state.results.converted_graph = global_state.results.converted_graph[:-1, :-1]
            ground_truth_flat = global_state.user_data.ground_truth.flatten() 
            if revise:
                 est_graph_flat = np.where(global_state.results.revised_graph==1, 1, 0).flatten()
            else:
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




