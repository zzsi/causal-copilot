class Reranker(object):
    # Kun Zhou Implemented
    def __init__(self, args):
        self.args = args

    def statics_dict2string(self, statics_dict):
        return str(statics_dict)

    def algo_can2string(self, algo_candidates):
        algo_candidate_string = ''
        algo2des_cond_hyper = {}
        for algo_name in algo_candidates:
            # todo hyper_dict processing
            algo_des, algo_justify = algo_candidates[algo_name]['description'], algo_candidates[algo_name]['justification']
            algo_string = algo_name + ':\nDescription: ' + algo_des + '\nJustification: ' + algo_justify
            algo_cond = "Independence Test or Score Function Requirement: " + algo_candidates[algo_name]['independence_test_or_score_function']
            algo_hyper = "Hyper-parameters Requirements:\n"
            for id, hyper_name in enumerate(algo_candidates[algo_name]['hyperparameters']):
                algo_hyper += str(id) + ": " + hyper_name + ': ' + algo_candidates[algo_name]['hyperparameters'][hyper_name] + '\n'
            algo2des_cond_hyper[algo_name] = (algo_string, algo_cond, algo_hyper)
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

    def forward(self, data, algo_candidates, statics_dict, knowledge_docs):
        '''

        :param data: Given Tabular Data in Pandas DataFrame format
        :param algo_candidates: A list containing all algorithm candidates and their description
        :param statics_dict: A dict containing all necessary statics information
        :param knowledge_docs: A list containing all necessary domain knowledge information from GPT-4
        :return: A doc containing the selected algorithm and its hyperparameter settings
        '''
        from openai import OpenAI
        client = OpenAI(organization=self.args.organization, project=self.args.project, api_key=self.args.apikey)
        table_name = self.args.data_file
        table_columns = '\t'.join(data.columns._data)
        knowledge_info = '\n'.join(knowledge_docs)
        statics_info = self.statics_dict2string(statics_dict)
        algo_info, algo2des_cond_hyper = self.algo_can2string(algo_candidates)

        # Select the Best Algorithm
        prompt = ("I will conduct causal discovery on the Tabular Dataset %s containing the following Columns:\n\n"
                  "%s\n\nThe Detailed Background Information is listed below:\n\n"
                  "%s\n\nThe Statics Information about the dataset is:\n\n"
                  "%s\n\nBased on the above information, please select the best-suited algorithm from the following candidate:\n\n"
                  "%s\n\nPlease highlight the selected algorithm name using the following template <Algo>Name</Algo> in the ending of the output") % (table_name, table_columns, knowledge_info, statics_info, algo_info)
        selected_algo = ''
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

        # Set up the Hyperparameters
        algo_des, algo_cond, algo_hyper = algo2des_cond_hyper[selected_algo]
        prompt = ("I will conduct causal discovery on the Tabular Dataset %s containing the following Columns:\n\n"
                  "%s\n\nThe Detailed Background Information is listed below:\n\n"
                  "%s\n\nThe Statics Information about the dataset is:\n\n"
                  "%s\n\nWe have determined to use the algorithm %s, whose description is: "
                  "%s\nIts condition requirements are: "
                  "%s\nIts hyperparameters are listed below:"
                  "%s\n\nPlease give suggestions about the hyperparameter setup based on the above information. If not very confident, please use the default value."
                  "Finally, please generate the hyperparameter suggestions in the ending of the output, using the following template <Hyper>hyperparameter suggestions</Hyper>") % (
                 table_name, table_columns, knowledge_info, statics_info, selected_algo, algo_des, algo_cond, algo_hyper)
        hyper_suggest = ''
        while hyper_suggest == '':
            print("The used prompt for hyper-parameter suggestion is: -------------------------------------------------------------------------")
            print(prompt)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            output = response.choices[0].message.content
            print("The received answer for hyper-parameter suggestion is: -------------------------------------------------------------------------")
            print(output)
            hyper_suggest = self.extract(output, "<Hyper>", "</Hyper>")
        return selected_algo, hyper_suggest