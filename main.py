# Kun Zhou Implemented
from data.simulation.simulation import SimulationManager
from preprocess.dataset import load_data, statistics_info, convert_stat_info_to_text,knowledge_info
from algorithm.filter import Filter
from algorithm.program import Programming
from algorithm.rerank import Reranker
from postprocess.judge import Judge
from postprocess.visualization import Visualization
from postprocess.report_generation import Report_generation
from global_setting.Initialize_state import global_state_initialization, GlobalState, load_data

import json
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

    # Input data file
    parser.add_argument(
        '--data-file',
        type=str,
        default="test_data/20241018_020318_base_nodes10_samples2000",
        help='Path to the input dataset file (e.g., CSV format or directory location)'
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
        '--output-report-dir',
        type=str,
        default='test_data/20241018_020318_base_nodes10_samples2000/output_report',
        help='Directory to save the output report'
    )

    # Output directory for graphs
    parser.add_argument(
        '--output-graph-dir',
        type=str,
        default='test_data/20241018_020318_base_nodes10_samples2000/output_graph',
        help='Directory to save the output graph'
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
        '--domain_index',
        type=str,
        default='domain_index',
        help='Name of the column which indicates the domain index'
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

    # Postprocess options
    parser.add_argument(
        '--boot_num',
        type=int,
        default=5,
        help='Number of bootstrap iterations'
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
        default="offline",
        help='Simulation mode: online or offline'
    )

    parser.add_argument(
        '--data_mode',
        type=str,
        default="simulated",
        help='Data mode: real or simulated'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Enable debugging mode'
    )

    parser.add_argument(
        '--initial_query',
        type=str,
        default="",
        help='Initial query for the algorithm'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    global_state = global_state_initialization(args.data_file, args.initial_query)
    global_state = load_data(global_state, args)

    # background info collection
    print("Original Data: ", global_state.user_data.raw_data)

    if args.debug:
        # Fake statistics for debugging
        global_state.statistics.missingness = False
        global_state.statistics.data_type = "Continuous"
        global_state.statistics.linearity = True
        global_state.statistics.gaussian_error = True
        global_state.statistics.stationary = "non time-series"
        preprocessed_data = global_state.user_data.raw_data
        global_state.user_data.knowledge_docs = "This is fake domain knowledge for debugging purposes."
    else:
        global_state = statistics_info(global_state)
        global_state.user_data.knowledge_docs = knowledge_info(args, global_state.user_data.preprocessed_data)

    # Convert statistics to text
    global_state.statistics.description = convert_stat_info_to_text(global_state.)
    
    print("Preprocessed Data: ", global_state.user_data.processed_data)
    print("Statistics Info: ", global_state.statistics.description)
    print("Knowledge Info: ", global_state.user_data.knowledge_docs)
    
    # Algorithm selection and deliberation
    filter = Filter(args)
    algo_candidates = filter.forward(global_state)
    global_state.algorithm.algorithm_candidates = algo_candidates

    reranker = Reranker(args)
    algorithm, hyper_suggest, prompt, hp_prompt = reranker.forward(
        global_state.user_data.processed_data, 
        global_state.algorithm.algorithm_candidates, 
        global_state.statistics.description, 
        global_state.user_data.knowledge_docs
    )
    global_state.algorithm.selected_algorithm = algorithm
    global_state.algorithm.algorithm_arguments = hyper_suggest

    programmer = Programming(args)
    code, results = programmer.forward(global_state.user_data.processed_data, algorithm, hyper_suggest)
    global_state.results.raw_result = results

    judge = Judge(args)
    mat_ground_truth = global_state.user_data.ground_truth
    print("Original Graph: ", results)
    print("Mat Ground Truth: ", mat_ground_truth)

    shd, precision, recall, f1 = judge.evaluation(results, mat_ground_truth)
    print(shd, precision, recall, f1)
    original_metrics = {'shd': shd,
                       'precision': precision,
                       'recall': recall,
                       'f1': f1}

    flag, _, boot_probability, revised_graph = judge.forward(preprocessed_data, results, algorithm, hyper_suggest, global_state.user_data.knowledge_docs)
    #print(flag)
    #print(boot_probability)

    print("Revised Graph: ", revised_graph)
    print("Mat Ground Truth: ", mat_ground_truth)
    shd, precision, recall, f1 = judge.evaluation(revised_graph, mat_ground_truth)
    print(shd, precision, recall, f1)
    revised_metrics = {'shd': shd,
                       'precision': precision,
                       'recall': recall,
                       'f1': f1}

    #############Visualization###################
    my_visual = Visualization(data=global_state.user_data.raw_data,
                              y=['MEDV'],
                              save_dir=args.output_graph_dir,
                              threshold=0.95)
    if global_state.user_data.ground_truth is not None:
        true_fig_path = my_visual.mat_to_graph(full_graph=mat_ground_truth,
                                               edge_labels=None,
                                               title='True Graph')

    boot_dict = my_visual.process_boot_mat(boot_probability, global_state.results.raw_result)

    result_fig_path = my_visual.mat_to_graph(full_graph=global_state.results.raw_result,
                                                 edge_labels=boot_dict,
                                                 title='Initial Graph')

    revised_fig_path = my_visual.mat_to_graph(full_graph=revised_graph,
                                              ori_graph=global_state.results.raw_result,
                                              edge_labels=None,
                                              title='Revised Graph')

    metrics_fig_path = my_visual.matrics_plot(original_metrics.copy(), revised_metrics.copy())

    ################################
    
    # algorithm selection process
    '''
    round = 0
    flag = False

    while round < args.max_iterations and flag == False:
        code, results = programmer.forward(preprocessed_data, algorithm, hyper_suggest)
        flag, algorithm_setup = judge(preprocessed_data, code, results, statistics_dict, algorithm_setup, knowledge_docs)
    '''

    #############Report Generation###################
    my_report = Report_generation(args, global_state.user_data.raw_data,
                                  global_state.statistics.description, global_state.user_data.knowledge_docs, 
                                  prompt, hp_prompt,
                                  global_state.results.revised_graph, 
                                  original_metrics, revised_metrics,
                                  visual_dir=args.output_graph_dir)
    report = my_report.generation()
    my_report.save_report(report, save_path=args.output_report_dir)
    ################################
   

if __name__ == '__main__':
    main()
