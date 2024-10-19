class Reranker(object):
    # Kun Zhou Implemented
    def __init__(self, args):
        self.args = args

    def statistics_dict2string(self, statistics_dict):
        return str(statistics_dict)

    def algo_can2string(self, algo_candidates, hp_context):
        algo_candidate_string = ''
        algo2des_cond_hyper = {}
        for algo_name in algo_candidates:
            if algo_name in hp_context:
                algo_des, algo_justify = algo_candidates[algo_name]['description'], algo_candidates[algo_name]['justification']
                algo_string = algo_name + ':\n' + algo_des + '\nJustification: ' + algo_justify
                algo2des_cond_hyper[algo_name] = algo_string
                algo_candidate_string += algo_string + "\n\n"
        return algo_candidate_string, algo2des_cond_hyper

    def extract(self, output, start_str, end_str):
        if start_str in output and end_str in output:
            try:
                algo = output.split(start_str)[1].split(end_str)[0]
            except:
                algo = ''
            return algo
        else:
            return ''

    # workflow
    # 1. select candidate algorithms -> algorithm names and their descriptions
    # 2. rerank the algorithms based on more criteria -> pick the best one
    # 3. for the best algorithm, select the best hyperparameters
    #   a. give suggestions for the primary hyperparameters, use default values of the secondary hyperparameters
    #     i. for each primary hyperparameters, we might need to have some context about their meaning and possible values
    # 4. return the selected algorithm and its hyperparameters
    def forward(self, global_state):
        '''
        :param global_state: The global state containing the processed data, algorithm candidates, statistics description, and knowledge documents
        :return: A doc containing the selected algorithm and its hyperparameter settings
        '''
        from openai import OpenAI
        client = OpenAI(organization=self.args.organization, project=self.args.project, api_key=self.args.apikey)
        # Set up the Hyperparameters
        # Load hyperparameters prompt template
        import json
        import algorithm.wrappers as wrappers

        data = global_state.user_data.processed_data
        algo_candidates = global_state.algorithm.algorithm_candidates
        statistics_desc = global_state.statistics.description
        knowledge_docs = global_state.user_data.knowledge_docs

        if global_state.algorithm.selected_algorithm is None:
            with open("algorithm/context/hyperparameters_prompt.txt", "r") as f:
                hp_prompt = f.read()

            # Load hyperparameters context
            with open("algorithm/context/hyperparameters.json", "r") as f:
                hp_context = json.load(f)

            # Load additional context files for parameters that have them
            for algo in hp_context:
                for param in hp_context[algo]:
                    if 'context_file' in hp_context[algo][param]:
                        context_file_path = hp_context[algo][param]['context_file']
                        with open(context_file_path, "r") as cf:
                            hp_context[algo][param]['context_content'] = cf.read()

            table_name = self.args.data_file
            table_columns = '\t'.join(data.columns._data)
            knowledge_info = '\n'.join(knowledge_docs)
            statistics_info = statistics_desc
            algo_info, algo2des_cond_hyper = self.algo_can2string(algo_candidates, hp_context)

            # Select the Best Algorithm
            prompt = ("I will conduct causal discovery on the Tabular Dataset %s containing the following Columns:\n\n"
                    "%s\n\nThe Detailed Background Information is listed below:\n\n"
                    "%s\n\nThe Statistics Information about the dataset is:\n\n"
                    "%s\n\nBased on the above information, please select the best-suited algorithm from the following candidate:\n\n"
                    "%s\n\nPlease highlight the selected algorithm name using the following template <Algo>Name</Algo> in the ending of the output") % (table_name, table_columns, knowledge_info, statistics_info, algo_info)
            selected_algo = ''
            print("Keys in algo2des_cond_hyper:", algo2des_cond_hyper.keys())
            while selected_algo not in algo2des_cond_hyper:
                print("The used prompt for rerank is: -------------------------------------------------------------------------")
                print(prompt)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                output = response.choices[0].message.content
                print("The received answer for rerank is: -------------------------------------------------------------------------")
                print(output)
                selected_algo = self.extract(output, '<Algo>', '</Algo>')
            global_state.algorithm.selected_algorithm = selected_algo
        else:
            print("User has already selected the algorithm, skip the reranking process.")
            print("Selected Algorithm: ", global_state.algorithm.selected_algorithm)

        if global_state.algorithm.selected_algorithm is not None and global_state.algorithm.algorithm_arguments is None:
            # Get algorithm description and hyperparameters

            algo_description = algo2des_cond_hyper[selected_algo]
            primary_params = getattr(wrappers, selected_algo)().get_primary_params()

            # Prepare hyperparameter information
            hp_info_str = str(hp_context[selected_algo])

            # Create the hyperparameter selection prompt
            hp_prompt = hp_prompt.replace("[COLUMNS]", table_columns)
            hp_prompt = hp_prompt.replace("[KNOWLEDGE_INFO]", knowledge_info)
            hp_prompt = hp_prompt.replace("[STATISTICS INFO]", statistics_desc)
            hp_prompt = hp_prompt.replace("[ALGORITHM_NAME]", selected_algo)
            hp_prompt = hp_prompt.replace("[ALGORITHM_DESCRIPTION]", algo_description)
            hp_prompt = hp_prompt.replace("[PRIMARY_HYPERPARAMETERS]", str(primary_params))
            hp_prompt = hp_prompt.replace("[HYPERPARAMETER_INFO]", hp_info_str)

            # Get hyperparameter suggestions from GPT-4
            print("Hyperparameter Prompt: ", hp_prompt)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a causal discovery expert. Provide your response in JSON format."},
                    {"role": "user", "content": hp_prompt}
                ],
                response_format={"type": "json_object"}
            )

            print("Hyperparameter Response: ", response.choices[0].message.content)
            hyper_suggest = json.loads(response.choices[0].message.content)

            # only use the hyperparameter keys and values, explanation is added later
            hyper_suggest = {k: v['value'] for k, v in hyper_suggest['hyperparameters'].items() if k in primary_params}

            global_state.algorithm.algorithm_arguments = hyper_suggest
            print("Selected Algorithm: ", selected_algo)
            print("Hyperparameter Suggestions: ", hyper_suggest)
        else:
            print("User has already selected the hyperparameters, skip the hyperparameter selection process.")
            print("Selected Hyperparameters: ", global_state.algorithm.algorithm_arguments)

        global_state.logging.
        return global_state
