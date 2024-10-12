# Kun Zhou Implemented
from data.simulation.simulation import SimulationManager
from preprocess.dataset import load_data, statics_info, knowledge_info
from algorithm.filter import Filter
from algorithm.program import Programming
from algorithm.rerank import Reranker
from postprocess.judge import Judge

import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

    # Input data file
    parser.add_argument(
        '--data-file',
        type=str,
        default="simulated_data/20240918_224140_base_nodes4_samples1000",
        help='Path to the input dataset file (e.g., CSV format)'
    )

    # Target variable
    parser.add_argument(
        '--target-variable',
        type=str,
        help='Name of the target variable in the dataset'
    )

    # Covariates or features
    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        help='List of feature names to include in the analysis'
    )

    # Causal model selection
    parser.add_argument(
        '--model',
        type=str,
        choices=['linear_regression', 'propensity_score_matching', 'causal_forest', 'do_calculus'],
        help='Causal inference model to use for the analysis'
    )

    # Hyperparameters for the model
    parser.add_argument(
        '--hyperparameters',
        type=str,
        help='JSON string or path to JSON file containing hyperparameters for the chosen model'
    )

    # Output file for results
    parser.add_argument(
        '--output-file',
        type=str,
        default='results.txt',
        help='File path to save the analysis results'
    )

    # Data preprocessing options
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Apply normalization to the dataset'
    )
    parser.add_argument(
        '--impute-missing',
        action='store_true',
        help='Impute missing values in the dataset'
    )

    # Data Preprocess Hyper-parameters
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.5,
        help=''
    )
    parser.add_argument(
        '--ts',
        type=bool,
        default=False,
        help=''
    )
    parser.add_argument(
        '--num_test',
        type=int,
        default=100,
        help=''
    )
    # Verbosity level
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='Enable verbose output during analysis'
    )

    # Max Deliberation Round
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='The maximum number of iterations to run the algorithm'
    )

    # OpenAI Settings
    parser.add_argument(
        '--organization',
        type=str,
        default="org-5NION61XDUXh0ib0JZpcppqS",
        help='Organization ID'
    )

    parser.add_argument(
        '--project',
        type=str,
        default="proj_Ry1rvoznXAMj8R2bujIIkhQN",
        help='Project ID'
    )

    parser.add_argument(
        '--apikey',
        type=str,
        default="sk-l4ETwy_5kOgNvt5OzHf_YtBevR1pxQyNrlW8NRNPw2T3BlbkFJdKpqpbcDG0IhInYcsS3CXdz_EMHkJO7s1Bo3e4BBcA",
        help='API Key'
    )

    parser.add_argument(
        '--simulation_mode',
        type=str,
        default="online",
        help='Simulation mode: online or offline'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debugging mode'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.simulation_mode == "online":
        simulation_manager = SimulationManager(args)
        config, data, graph = simulation_manager.generate_dataset()
    elif args.simulation_mode == "offline":
        config, data, graph = load_data(args.data_file)
    else:
        raise ValueError("Invalid simulation mode. Please choose 'online' or 'offline'.")

    # background info collection
    print("Original Data: ", data)
    
    if args.debug:
        # Fake statistics_dict and knowledge_docs for debugging
        statistics_dict = {
            "Missingness": False,
            "Data Type": "Continuous",
            "Linearity": True,
            "Gaussian Error": True,
            "Stationary": "non time-series"
        }
        statistics_dict = json.dumps(statistics_dict, indent=4)
        preprocessed_data = data  # For simplicity, use original data
        knowledge_docs = ["This is fake domain knowledge for debugging purposes."]
    else:
        statistics_dict, preprocessed_data = statics_info(args, data)
        knowledge_docs = knowledge_info(args, preprocessed_data)
    
    print("Preprocessed Data: ", preprocessed_data)
    print("Statics Info: ", statistics_dict)
    print("Knowledge Info: ", knowledge_docs)

    # algorithm selection and deliberation initialization
    filter = Filter(args)
    algo_candidates = filter.forward(preprocessed_data, statistics_dict)
    print(algo_candidates)

    reranker = Reranker(args)
    algorithm, hyper_suggest = reranker.forward(preprocessed_data, algo_candidates, statistics_dict, knowledge_docs)
    print(algorithm)
    print(hyper_suggest)

    programmer = Programming(args)
    code, results = programmer.forward(preprocessed_data, algorithm, hyper_suggest)
    print(results)

    judge = Judge(args)
    flag, hyper_suggest = judge.forward(preprocessed_data, code, results, statistics_dict, hyper_suggest, knowledge_docs)

    # algorithm selection process
    '''
    round = 0
    flag = False

    while round < args.max_iterations and flag == False:
        code, results = programmer.forward(preprocessed_data, algorithm, hyper_suggest)
        flag, algorithm_setup = judge(preprocessed_data, code, results, statistics_dict, algorithm_setup, knowledge_docs)
    '''

    judge.report_generation(preprocessed_data, results, statistics_dict, hyper_suggest, knowledge_docs)

if __name__ == '__main__':
    main()