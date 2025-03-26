from .wrappers import __all__ as all_algos
from .hyperparameter_selector import HyperparameterSelector
from .runtime_estimators.runtime_estimator import RuntimeEstimator
from .llm_client import LLMClient

class Reranker:
    def __init__(self, args):
        self.args = args
        self.hp_selector = HyperparameterSelector(args)
        self.llm_client = LLMClient(args)

    def forward(self, global_state):
        algo_candidates = self.filter_algo_candidates(global_state)
        algo_info, algo2des_cond_hyper = self.algo_can2string(algo_candidates)

        # Select algorithm
        selected_algo = self.select_algorithm(global_state, algo_candidates, algo_info, algo2des_cond_hyper)
        global_state.algorithm.selected_algorithm = selected_algo

        # # Get time info for CDNOD if needed
        # time_info_cdnod = ""
        # if selected_algo == 'CDNOD':
        #     _, time_info_cdnod = self.algo_cans2time_string([selected_algo], global_state.statistics.sample_size, global_state.statistics.feature_number)

        return global_state
    
    def create_prompt(self, global_state, algo_info, time_info):
        algorithm_guidelines = open(f"algorithm/context/algos/guidelines.txt", "r").read()
        prompt_template = open(f"algorithm/context/algo_rerank_prompt.txt", "r").read()

        # load the candidates' profiles
        algorithm_profiles = ""
        for algo in global_state.algorithm.algorithm_candidates:
            profile_path = f"algorithm/context/algos/{algo}.txt"
            with open(profile_path, "r", encoding="utf-8") as f:
                algorithm_profiles += f"======================================\n\n" + f"# {algo}\n\n" + f.read() + "\n\n"


        replacements = {
            "[TABLE_NAME]": self.args.data_file,
            "[COLUMNS]": '\t'.join(global_state.user_data.processed_data.columns._data),
            "[KNOWLEDGE_INFO]": str(global_state.user_data.knowledge_docs),
            "[STATISTICS_INFO]": global_state.statistics.description,
            "[ALGORITHM_CANDIDATES]": str(global_state.algorithm.algorithm_candidates.keys()),
            "[WAIT_TIME]": str(global_state.algorithm.waiting_minutes),
            "[TIME_INFO]": time_info,
            # "[ALGORITHM_GUIDELINES]": algorithm_guidelines,
            "[ALGORITHM_PROFILES]": algorithm_profiles
        }

        for placeholder, value in replacements.items():
            prompt_template = prompt_template.replace(placeholder, value)

        with open(f"algorithm/context/algo_rerank_prompt_test.txt", "w", encoding="utf-8") as f:
            f.write(prompt_template)

        return prompt_template
    
    def filter_algo_candidates(self, global_state):
        # filter out CDNOD if data is not heterogeneous, hard-coded for now
        # if not global_state.statistics.heterogeneous:
        #     if global_state.algorithm.selected_algorithm == 'CDNOD':
        #         print("Sorry! As the data is not heterogeneous, CDNOD algorithm should not be used! "
        #               "Causality-Copilot will continue to select the best-suited algorithm for you!")
        #         global_state.algorithm.selected_algorithm = None
        #     algo_candidates = {algo: global_state.algorithm.algorithm_candidates[algo] 
        #             for algo in global_state.algorithm.algorithm_candidates if algo != 'CDNOD'}
        # else:
        #     algo_candidates = global_state.algorithm.algorithm_candidates

        # filter out algorithm candidates that are not in the hp_context
        algo_candidates = global_state.algorithm.algorithm_candidates
        algo_candidates = {algo: algo_candidates[algo] for algo in algo_candidates if algo in all_algos}

        # if user has already selected an algorithm, only keep the selected algorithm in the algo_candidates
        if global_state.algorithm.selected_algorithm is not None:
            algo_candidates = {global_state.algorithm.selected_algorithm: {'description': '', 'justification': ''}}

        return algo_candidates
    
    def select_algorithm(self, global_state, algo_candidates, algo_info, algo2des_cond_hyper):
        if global_state.algorithm.selected_algorithm is not None:
            print(f"User has already selected the algorithm: {global_state.algorithm.selected_algorithm}, skip the reranking process.")
            global_state.algorithm.algorithm_optimum = {
                "reasoning": "",
                "reason": f"The user has already selected the algorithm: {global_state.algorithm.selected_algorithm}",
                "algorithm": global_state.algorithm.selected_algorithm
            }

            return global_state.algorithm.selected_algorithm
        
        time_info = self.runtime_estimate(algo_candidates, global_state.statistics.sample_size, global_state.statistics.feature_number)
        
        prompt = self.create_prompt(global_state, algo_info, time_info)
        output = self.llm_client.chat_completion(prompt, json_response=True, model="gpt-4o")
        print("-"*25, "\n", "The received answer for rerank is: ", "\n", output)
        selected_algo = output['algorithm']
        global_state.algorithm.algorithm_optimum = output

        global_state.logging.select_conversation.append({
            "prompt": prompt,
            "response": output
        })

        return selected_algo
    
    def runtime_estimate(self, algo_candidates, n_sample, feature_number):
        """Estimate runtime for given algorithms using UnifiedEstimator."""        
        runtime_estimates = {}
        for algo in algo_candidates:
            try:
                estimator = RuntimeEstimator(algo)
                runtime = estimator.predict_runtime(
                    samples=n_sample,
                    variables=feature_number
                )
                runtime_estimates[algo] = runtime
            except Exception as e:
                print(f"Warning: Failed to estimate runtime for {algo}: {e}")
                runtime_estimates[algo] = float('inf')

        # Format results into string
        runtime_strings = []
        for algo, runtime in runtime_estimates.items():
            runtime_str = f"{runtime/60:.1f} minutes"
            runtime_strings.append(f"{algo}: {runtime_str}")

        print("-"*25, "\n", "Runtime Estimates: ", "\n", runtime_strings)
        return "\n".join(runtime_strings)
    
    def algo_can2string(self, algo_candidates):
        algo_strings = []
        algo2des_cond_hyper = {}
        
        for algo_name, algo_info in algo_candidates.items():
            algo_string = (f"{algo_name}:\n"
                          f"{algo_info['description']}\n"
                          f"Justification: {algo_info['justification']}")
            
            algo2des_cond_hyper[algo_name] = algo_string
            algo_strings.append(algo_string)
            
        return "\n\n".join(algo_strings), algo2des_cond_hyper
