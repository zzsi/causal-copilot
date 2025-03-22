import json
import os
from openai import OpenAI
import algorithm.wrappers as wrappers
from algorithm.llm_client import LLMClient

class HyperparameterSelector:
    def __init__(self, args):
        self.args = args
        self.llm_client = LLMClient(args)

    def forward(self, global_state):
        selected_algo = global_state.algorithm.selected_algorithm
        hp_context = self.load_hp_context(selected_algo)
        algorithm_description = global_state.algorithm.algorithm_candidates[selected_algo]['description']
        algorithm_optimum_reason = global_state.algorithm.algorithm_optimum['reason']
        algorithm_optimum_reason = algorithm_description + "\n" + algorithm_optimum_reason

        # Select hyperparameters
        hyper_suggest = self.select_hyperparameters(global_state, selected_algo, hp_context, algorithm_optimum_reason)
        global_state.algorithm.algorithm_arguments = hyper_suggest
        return global_state
        

    def load_hp_context(self, selected_algo):
        # Load hyperparameters context
        # hp_context = {}
        # hyperparameters_folder = "algorithm/context/hyperparameters"
        return open(f"algorithm/context/hyperparameters/{selected_algo}.json", "r").read()
            
        # # Load additional context files for parameters that have them
        # # for algo in hp_context:
        # #     for param in hp_context[algo]:
        # #         if 'context_file' in hp_context[algo][param]:
        # #             context_file_path = hp_context[algo][param]['context_file']
        # #             with open(context_file_path, "r") as cf:
        # #                 hp_context[algo][param]['context_content'] = cf.read()
        
        # return hp_context

    def create_prompt(self, global_state, selected_algo, hp_context, algorithm_optimum_reason):
        with open("algorithm/context/hyperparameter_select_prompt.txt", "r") as f:
            hp_prompt = f.read()
        
        print(selected_algo)
        primary_params = getattr(wrappers, selected_algo)().get_primary_params()
        hp_info_str = json.dumps(hp_context)
        table_columns = '\t'.join(global_state.user_data.processed_data.columns._data)
        knowledge_info = '\n'.join(global_state.user_data.knowledge_docs)
        
        hp_prompt = hp_prompt.replace("[COLUMNS]", table_columns)
        hp_prompt = hp_prompt.replace("[KNOWLEDGE_INFO]", knowledge_info)
        hp_prompt = hp_prompt.replace("[STATISTICS INFO]", global_state.statistics.description)
        hp_prompt = hp_prompt.replace("[ALGORITHM_NAME]", selected_algo)
        hp_prompt = hp_prompt.replace("[ALGORITHM_DESCRIPTION]", algorithm_optimum_reason)
        # hp_prompt = hp_prompt.replace("[PRIMARY_HYPERPARAMETERS]", str(primary_params))
        hp_prompt = hp_prompt.replace("[HYPERPARAMETER_INFO]", hp_info_str)

        with open(f"algorithm/context/hp_rerank_prompt_test.txt", "w", encoding="utf-8") as f:
            f.write(hp_prompt)

        return hp_prompt, primary_params
        
    def select_hyperparameters(self, global_state, selected_algo, hp_context, algorithm_optimum_reason):
        if global_state.algorithm.algorithm_arguments is not None:
            print("User has already selected the hyperparameters, skip the hyperparameter selection process.")
            return global_state.algorithm.algorithm_arguments
        
        hp_prompt, primary_params = self.create_prompt(global_state, selected_algo, hp_context, algorithm_optimum_reason)

        # if selected_algo == 'CDNOD' and global_state.statistics.linearity == False:
        #     kci_prompt = (f'\nAs it is nonlinear data, it is suggested to use kci for indep_test. '
        #                 f'As the user can wait for {global_state.algorithm.waiting_minutes} minutes for the algorithm execution. If kci can not exceed it, we MUST select it:\n\n'
        #                 f'The estimated time costs of CDNOD algorithms using the two indep_test settings are: {time_info_cdnod}')
        #     hp_prompt = hp_prompt + kci_prompt
        
        response = self.llm_client.chat_completion(hp_prompt,
                                                    system_prompt="You are a causal discovery expert. Provide your response in JSON format.", json_response=True)

        hyper_suggest = response
        global_state.algorithm.algorithm_arguments_json = hyper_suggest
        print(hyper_suggest)
        hyper_suggest = {k: v['value'] for k, v in hyper_suggest['hyperparameters'].items() if k in primary_params}

        global_state.logging.argument_conversation.append({
            "prompt": hp_prompt,
            "response": response
        })

        print("-"*25, "\n", "Hyperparameter Response: ", "\n", hyper_suggest)

        return hyper_suggest 