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
    import numpy as np

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
        graph = graph.T
    else:
        graph = None

    return config, data, graph


def statistics_info(args, data):
    # Fang Nan Implemented
    '''
    :param args: configurations.
    :param data: Given Tabular Data in Pandas DataFrame format
    :return: A dict containing all necessary statistics information
    '''
    from preprocess.stat_info_functions import stat_info_collection

    statistics_dict, preprocessed_data = stat_info_collection(args=args, data=data)

    return statistics_dict, preprocessed_data


def convert_stat_info_to_text(stat_info):
    """
    Convert the statistical information dictionary to natural language.
    
    :param stat_info: Dictionary containing statistical information about the dataset.
    :return: A string describing the dataset characteristics in natural language.
    """
    import json
    info = json.loads(stat_info)
    
    text = f"The dataset has the following characteristics:\n\n"
    text += f"The sample size is {info['Sample Size']} with {info['Number of Features']} features. "
    text += f"This dataset {'is' if info['Time-series'] else 'is not'} time-series data. "
    
    text += f"Data Type: The overall data type is {info['Data Type']}.\n\n"
    text += f"Data Quality: {'There are' if info['Missingness'] else 'There are no'} missing values in the dataset.\n\n"
    
    if not info['Time-series']:
        text += "Statistical Properties:\n"
        text += f"- Linearity: The relationships between variables {'are' if info['Linearity'] else 'are not'} predominantly linear.\n"
        text += f"- Gaussian Errors: The errors in the data {'do' if info['Gaussian Error'] else 'do not'} follow a Gaussian distribution.\n"
        text += f"- This dataset {'is' if info['Heterogeneity'] else 'is not'} heterogeneous. \n\n"

        text += "Implications for Analysis:\n"
        if info['Linearity'] and info['Gaussian Error']:
            text += "1. The data is well-suited for linear modeling techniques.\n"
        else:
            if not info['Linearity']:
                text += "1. Non-linear modeling techniques may be more appropriate.\n"
            if not info['Gaussian Error']:
                text += "2. Robust statistical methods or transformations might be necessary.\n"
        
        if info['Missingness']:
            text += "3. Imputation techniques should be considered during preprocessing.\n"
    else:
        text += "Time Series Properties:\n"
        text += f"- Stationarity: The time series {'is' if info['Stationary'] else 'is not'} stationary.\n\n"
        
        text += "Implications for Analysis:\n"
        if info['Stationary']:
            text += "1. Standard time series analysis techniques can be applied.\n"
        else:
            text += "1. Differencing or other stationarity-inducing transformations may be necessary.\n"
        
        if info['Missingness']:
            text += "2. Time-series specific imputation methods should be considered.\n"

    if 'Domain Index' in info and info['Domain Index'] is not None:
        text += f"If the data is heterogeneous, the column/variable {info['Domain Index']} is the domain index indicating the heterogeneity. "
        text += f"If the data is not heterogeneous, then the existed domain index is constant.\n\n"
    else:
        text += "\n\n"
        
    return text


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
              "%s\n\nIf the Variables Names are Meaningful, Please list the following information with clear format and accurate expression:"
              "\n1.Detailed Explanation about the Variables;"
              "\n2.Possible Causal Relations among these variables;"
              "\n3.Other Background Domain Knowledge that may be helpful for experts to design causal discovery algorithms\n\n"
              "Otherwise, if the Variable Names are just Symbols (like x1, y1), Please Return 'No Knowledge'") % (
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
    statistics_dict = statistics_info(args, data)
    print(statistics_dict)