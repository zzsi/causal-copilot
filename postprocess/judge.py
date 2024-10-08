class Judge(object):
    def __init__(self, args):
        pass

    def quality_judge(self, data, algorithm_setup, knowledge_docs, ts: bool = False):
        '''
        :param data: Given Tabular Data in Pandas DataFrame format
        :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
        :param knowledge_docs: A doc containing all necessary domain knowledge information from GPT-4.
        :param ts: indicator of time-series data
        :return: obvious errors in causal analysis results,
                bootstrap probability of directed edges and common cause.
        '''
        from algorithm.program import Programming

        # Run algorithm (using algorithm_setup) with the full dataset
        cg_boot = Programming(data, algorithm_setup)
        full_graph = cg_boot.G.graph

        # Statistics Perspective: Bootstrapping to get probability of edges using the selected algorithm.
        errors_stat, boot_probability, boot_prob_common_cause = bootstrap(
            data=data,
            algorithm_setup=algorithm_setup,
            full_graph=full_graph,
            ts=ts,
            boot_num=500
        )

        # LLM perspective: errors based on domain knowledge from GPT-4
        errors_llm = llm_evaluation(
            data=data,
            full_graph=full_graph,
            llm_setup=llm_setup,
            knowledge_docs=knowledge_docs
        )

        # Combine error obtained from both statistics and LLM perspectives
        errors = {**errors_stat, **errors_llm}

        return errors, boot_probability, boot_prob_common_cause

    def report_generation(self, llm_setup, data, graph, statics_dict, algorithm_setup,
                          knowledge_docs, boot_probability, boot_prob_common_cause):
        '''
        generate and save the report
        :param llm_setup: information of configurations of GPT-4
        :param data: Given Tabular Data in Pandas DataFrame format
        :param graph: graph generated using causal discovery algorithm
        :param statics_dict: A dict containing all necessary statics information
        :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
        :param knowledge_docs: A doc containing all necessary domain knowledge information from GPT-4
        :param boot_probability: bootstrap probability of directed edges, e.g., i -> j
        :param boot_prob_common_cause: bootstrap probability of common cause, e.g., i <-> j
        :return: Str: A technique report explaining all the results for readers
        '''
        from openai import OpenAI
        import numpy as np

        client = OpenAI(organization=llm_setup.organization, project=llm_setup.project, api_key=llm_setup.apikey)

        # Data information
        table_columns = '\t'.join(data.columns)

        # Data property prompt
        if statics_dict.get("Stationary") == "non time-series":
            missing = "has missing values," if statics_dict.get("Missingness") else "does not have missing values,"
            data_type = f"is {statics_dict.get('Data Type')} data,"
            linear = "satisfies the linearity assumption," if statics_dict.get("Linearity") else "violates the linearity assumption,"
            gaussian = ",and satisfies the Gaussian error assumption" if statics_dict.get("Gaussian Error") else ",and violates the Gaussian error assumption"
            data_prop_prompt = missing + data_type + linear + gaussian
        else:
            data_prop_prompt = f"is {'stationary' if statics_dict.get('Stationary') else 'non-stationary'} time-series data"

        # Graph prompt
        graph_prompt = graph_effect_prompts(
            column_names=data.columns,
            graph=graph,
            boot_probability=boot_probability,
            boot_prob_common_cause=boot_prob_common_cause
        )

        prompt = (f"I have conduct causal discovery on the Tabular Dataset containing the following columns: \n\n"
                  f"{table_columns}\n\n. "
                  f"In terms of the property of the dataset, it {data_prop_prompt}.\n\n"
                  f"Based on the property of the dataset mentioned above, we think {algorithm_setup} is suitable for "
                  f"running causal discovery on this dataset. \n\n"
                  f"{graph_prompt}"
                  f"Based on the context and knowledge {knowledge_docs}, and all of the results and information above, "
                  f"write a comprehensive technical report, the report should be easily readable and understandable to audience who may not "
                  f"be familiar with causal discovery. If there is common cause between variables, you should point out what may be "
                  f"the potential common cause.")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        report_doc = response.choices[0].message.content

        return report_doc

    def forward(self, data, results, statics_dict, algorithm_setup, knowledge_docs):
        '''
        :param data: Given Tabular Data in Pandas DataFrame format
        :param results: A dict containing all algorithm candidates
        :param statics_dict: A dict containing all necessary statics information
        :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
        :param knowledge_docs: A list containing all necessary domain knowledge information from GPT-4
        :return: A dict containing the revised algorithm and its hyperparameter settings
        '''
        errors, _, _ = self.quality_judge(data, results, statics_dict, algorithm_setup, knowledge_docs)
        
        if len(errors) == 0:
            return True, algorithm_setup
        else:
            new_algorithm_setup = {}
            return False, new_algorithm_setup
