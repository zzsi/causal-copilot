class Judge(object):
    def __init__(self, args):
        self.args = args

    def quality_judge(self, data, full_graph, algorithm, hyperparameters, knowledge_docs):
        '''
        :param data: Given Tabular Data in Pandas DataFrame format
        :param full_graph: An adjacent matrix in Numpy Ndarray format -
                           causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
        :param algorithm: String representing the algorithm name
        :param hyperparameters: Dictionary of hyperparameter names and values
        :param knowledge_docs: A doc containing all necessary domain knowledge information from GPT-4.
        :return: obvious errors in causal analysis results,
                 bootstrap probability of directed edges,
                 revised causal graph based on errors.
        '''

        from postprocess.judge_functions import bootstrap, llm_evaluation

        # Statistics Perspective: Bootstrapping to get probability of edges using the selected algorithm.
        errors_stat, boot_probability = bootstrap(data=data, full_graph=full_graph, algorithm=algorithm, hyperparameters=hyperparameters,
                                                  boot_num=self.args.boot_num, ts=self.args.ts)
        print("Errors from Bootstrap method: ", errors_stat)
        print("Bootstrap Probability: ", boot_probability)

        # LLM perspective: errors based on domain knowledge from GPT-4
        if "No Knowledge" in knowledge_docs:
            errors_llm = []
            print("No Errors are found by LLM, due to No Knowledge")
        else:
            errors_llm = llm_evaluation(data=data, full_graph=full_graph, args=self.args, knowledge_docs=knowledge_docs)
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

        return errors, boot_probability, revised_graph

    # def report_generation(self, llm_setup, data, graph, statics_dict, algorithm_setup,
    #                       knowledge_docs, boot_probability, boot_prob_common_cause):
    #     '''
    #     generate and save the report
    #     :param llm_setup: information of configurations of GPT-4
    #     :param data: Given Tabular Data in Pandas DataFrame format
    #     :param graph: graph generated using causal discovery algorithm
    #     :param statics_dict: A dict containing all necessary statics information
    #     :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
    #     :param knowledge_docs: A doc containing all necessary domain knowledge information from GPT-4
    #     :param boot_probability: bootstrap probability of directed edges, e.g., i -> j
    #     :param boot_prob_common_cause: bootstrap probability of common cause, e.g., i <-> j
    #     :return: Str: A technique report explaining all the results for readers
    #     '''
    #     from openai import OpenAI
    #     import numpy as np
    #
    #     client = OpenAI(organization=llm_setup.organization, project=llm_setup.project, api_key=llm_setup.apikey)
    #
    #     # Data information
    #     table_columns = '\t'.join(data.columns)
    #
    #     # Data property prompt
    #     if statics_dict.get("Stationary") == "non time-series":
    #         missing = "has missing values," if statics_dict.get("Missingness") else "does not have missing values,"
    #         data_type = f"is {statics_dict.get('Data Type')} data,"
    #         linear = "satisfies the linearity assumption," if statics_dict.get("Linearity") else "violates the linearity assumption,"
    #         gaussian = ",and satisfies the Gaussian error assumption" if statics_dict.get("Gaussian Error") else ",and violates the Gaussian error assumption"
    #         data_prop_prompt = missing + data_type + linear + gaussian
    #     else:
    #         data_prop_prompt = f"is {'stationary' if statics_dict.get('Stationary') else 'non-stationary'} time-series data"
    #
    #     # Graph prompt
    #     graph_prompt = graph_effect_prompts(
    #         column_names=data.columns,
    #         graph=graph,
    #         boot_probability=boot_probability,
    #         boot_prob_common_cause=boot_prob_common_cause
    #     )
    #
    #     prompt = (f"I have conduct causal discovery on the Tabular Dataset containing the following columns: \n\n"
    #               f"{table_columns}\n\n. "
    #               f"In terms of the property of the dataset, it {data_prop_prompt}.\n\n"
    #               f"Based on the property of the dataset mentioned above, we think {algorithm_setup} is suitable for "
    #               f"running causal discovery on this dataset. \n\n"
    #               f"{graph_prompt}"
    #               f"Based on the context and knowledge {knowledge_docs}, and all of the results and information above, "
    #               f"write a comprehensive technical report, the report should be easily readable and understandable to audience who may not "
    #               f"be familiar with causal discovery. If there is common cause between variables, you should point out what may be "
    #               f"the potential common cause.")
    #
    #     response = client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant."},
    #             {"role": "user", "content": prompt}
    #         ]
    #     )
    #     report_doc = response.choices[0].message.content
    #
    #     return report_doc

    def forward(self, data, full_graph, algorithm, hyperparameters, knowledge_docs):
        '''
        :param data: Given Tabular Data in Pandas DataFrame format
        :param full_graph: An adjacent matrix in Numpy Ndarray format -
                           causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
        :param algorithm: String representing the algorithm name
        :param hyperparameters: Dictionary of hyperparameter names and values
        :param knowledge_docs: A doc containing all necessary domain knowledge information from GPT-4.
        :return: A dict containing the revised algorithm and its hyperparameter settings
        '''
        errors, boot_probability, revised_graph = self.quality_judge(data=data, full_graph=full_graph, algorithm = algorithm,
                                                                     hyperparameters = hyperparameters, knowledge_docs=knowledge_docs)

        if len(errors) == 0:
            return True, errors, boot_probability, revised_graph
        else:
            return False, errors, boot_probability, revised_graph


    def evaluation(self, est_graph, ground_truth):
        '''
        :param est_graph: estimated adjacent matrix of causal graph in Panda Ndarray format
        :param ground_truth: ground truth, represented by adjacent matrix in Panda Ndarray format - Matrix[i,j] indicates j->i
        :return: Structural Hamming Distance, precision, recall, F1 score.
        '''

        import numpy as np
        from sklearn.metrics import precision_score, recall_score, f1_score

        # Structural Hamming Distance (SHD)
        diff = np.abs(est_graph - ground_truth)
        # Count how many edges are different
        shd = np.sum(diff)

        # Precision, Recall and F1-score
        precision = precision_score(ground_truth, est_graph, average=None)
        recall = recall_score(ground_truth, est_graph, average=None)
        f1 = f1_score(ground_truth, est_graph, average=None)

        return shd, precision, recall, f1
