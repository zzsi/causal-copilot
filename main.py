# Kun Zhou Implemented
from preprocess.dataset import load_data, statics_info, knowledge_info
from algorithm.filter import Filter
from algorithm.program import Programming
from algorithm.rerank import Reranker
from postprocess.judge import Judge


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

    # Input data file
    parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Path to the input dataset file (e.g., CSV format)'
    )

    # Target variable
    parser.add_argument(
        '--target-variable',
        type=str,
        required=True,
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
        required=True,
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

    # Verbosity level
    parser.add_argument(
        '--verbose',
        action='store_true',
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

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = load_data(args.data_file)

    # background info collection
    statics_dict = statics_info(args, data)
    knowledge_docs = knowledge_info(args, data)

    # algorithm selection and deliberation initialization
    filter = Filter(args)
    reranker = Reranker(args)
    programmer = Programming(args)
    judge = Judge(args)

    # algorithm selection process
    round = 0
    flag = False
    algo_candidates = filter(data, statics_dict)
    algorithm, algorithm_setup = reranker(data, algo_candidates, statics_dict, knowledge_docs)

    while round < args.max_iterations and flag == False:
        results = programmer(data, algorithm, algorithm_setup)
        flag, algorithm_setup = judge(data, results, statics_dict, algorithm_setup, knowledge_docs)

    judge.report_generation(data, results, statics_dict, algorithm_setup, knowledge_docs)