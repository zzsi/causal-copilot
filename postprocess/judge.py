class Judge(object):
    def __init__(self, args):
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
                                                  boot_num=boot_num, ts=False)
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

        return conversation, errors_llm, errors_stat, boot_probability, revised_graph


    def forward(self, global_state):
        (conversation,
         global_state.results.llm_errors,
         global_state.results.bootstrap_errors,
         global_state.results.bootstrap_probability,
         global_state.results.revised_graph) = self.quality_judge(
            data=global_state.user_data.processed_data,
            full_graph=global_state.results.converted_graph,
            algorithm=global_state.algorithm.selected_algorithm,
            hyperparameters=global_state.algorithm.algorithm_arguments,
            knowledge_docs=global_state.user_data.knowledge_docs,
            boot_num=global_state.statistics.boot_num
        )
        global_state.logging.knowledge_conversation.append(conversation)
        return global_state


    def evaluation(self, est_graph, ground_truth):
        '''
        :param est_graph: estimated adjacent matrix of causal graph in Panda Ndarray format
        :param ground_truth: ground truth, represented by adjacent matrix in Panda Ndarray format - Matrix[i,j] indicates j->i
        :return: Structural Hamming Distance, precision, recall, F1 score.
        '''

        import numpy as np
        from sklearn.metrics import precision_score, recall_score, f1_score

        if est_graph.shape[0] - 1 == ground_truth.shape[0]:
            # drop the domain index column
            est_graph = est_graph[:-1, :-1]
        ground_truth_flat = ground_truth.flatten()  
        est_graph_flat = est_graph.flatten()
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




