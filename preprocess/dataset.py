def load_data(directory):
    # Xinyue Wang Implemented
    '''
    :param path: str for data path or filename
    :return: pandas dataframe
    '''
    import json
    import numpy as np
    import pandas as pd
    import os

    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")
    
    config_path = f"{directory}/config.json"
    data_path = f"{directory}/base_data.csv"
    graph_path = f"{directory}/base_graph.npy"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = None
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        raise FileNotFoundError(f"The data file {data_path} does not exist.")
    if os.path.exists(graph_path):
        graph = np.load(graph_path)
    else:
        graph = None

    return config, data, graph


def statics_info(args, data):
    # Fang Nan Implemented
    '''
    :param args: a class containing pre-specified information - an indicator of time-series data,
                 missing ratio for data cleaning,
                 significance level (default 0.1),
                 maximum number of tests (default 1000).
    :param data: Given Tabular Data in Pandas DataFrame format
    :return: A dict containing all necessary statics information
    '''
    from preprocess.stat_info_functions import stat_info_collection

    statics_dict = stat_info_collection(args=args, data=data)

    return statics_dict


def knowledge_info(args, data):
    # Kun Zhou Implemented
    '''
    :param args: configurations
    :param data: Given Tabular Data in Pandas DataFrame format
    :return: A list containing all necessary domain knowledge information from GPT-4
    '''
    from openai import OpenAI
    client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
    table_name = args.data_file
    table_columns = '\t'.join(data.columns._data)
    prompt = ("I will conduct causal discovery on the Tabular Dataset %s containing the following Columns: \n\n"
              "%s\n\nPlease list the following information with clear format and accurate expression:"
              "\n1.Detailed Explanation about the Variables (columns);"
              "\n2.Possible Causal Relations among these variables;"
              "\n3.Other Background Domain Knowledge that may be helpful for experts to design causal discovery algorithms") % (
             table_name, table_columns)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    knowledge_doc = response.choices[0].message.content
    knowledge_docs = [knowledge_doc]
    return knowledge_docs


if __name__ == '__main__':
    data = load_data('../data/20240918_224141_base_nodes20_samples5000/base_data.csv')

    import argparse


    def parse_args():
        parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

        # Input data file
        parser.add_argument(
            '--data-file',
            type=str,
            default="data/20240918_224141_base_nodes20_samples5000/base_data.csv",
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

        args = parser.parse_args()
        return args
    args = parse_args()
    statics_dict = statics_info(args, data)
    print(statics_dict)