def json_type(x):
    import json
    import argparse
    try:
        return json.loads(x)
    except ValueError as ve:
        raise argparse.ArgumentTypeError("Invalid JSON: " + str(ve))

def global_configuration():
    import argparse
    import pandas as pd
    import numpy as np
    import json

    parser = argparse.ArgumentParser(description='ALL INFORMATION')

    # User
    parser.add_argument(
        '--raw_data',
        type=pd.DataFrame,
        default=None,
        help=''
    )

    parser.add_argument(
        '--ground_truth',
        type=np.array,
        default=None,
        help=''
    )

    parser.add_argument(
        '--processed_data',
        type=pd.DataFrame,
        default=None,
        help=''
    )

    parser.add_argument(
        '--initial_query',
        type=str,
        default=None,
        help=''
    )

    parser.add_argument(
        '--knowledge_docs',
        type=str,
        default=None,
        help=''
    )

    # Hyperparameters for statistical analysis
    parser.add_argument(
        '--boot_num',
        type=int,
        default=100,
        help='Number of bootstrap iterations'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='Significance level'
    )

    parser.add_argument(
        '--num_test',
        type=int,
        default=100,
        help='Maximum number of testing'
    )

    parser.add_argument(
        '--ratio',
        type=float,
        default=0.5,
        help='Threshold for missingness ratio'
    )

    #Statistical information
    stat_info = {"sample_size": None , "feature_number": None , "heterogeneity": None, "domain_index": None ,
                 "missingness": None, "data_type": None, "linearity": None, "gaussian_error": None}

    parser.add_argument(
        '--stat_info',
        type=json_type,
        default=json.dumps(stat_info),
        help='Language description of all statistics'
    )

    parser.add_argument(
        '--description',
        type=str,
        default=None,
        help='Language description of all statistics'
    )

    # Logging
    parser.add_argument(
        '--query_conversation',
        nargs='+',
        type=str,
        help=''
    )

    parser.add_argument(
        '--knowledge_conversation',
        nargs='+',
        type=str,
        help=''
    )

    parser.add_argument(
        '--filter_conversation',
        nargs='+',
        type=str,
        help=''
    )

    parser.add_argument(
        '--select_conversation',
        nargs='+',
        type=str,
        help=''
    )

    parser.add_argument(
        '--argument_conversation',
        nargs='+',
        type=str,
        help=''
    )

    parser.add_argument(
        '--errors_conversion',
        nargs='+',
        type=str,
        help=''
    )

    # Algorithm Selection
    parser.add_argument(
        '--algorithm_candidates',
        nargs='+',
        type=str,
        help=''
    )

    parser.add_argument(
        '--selected_algorithm',
        nargs='+',
        type=str,
        help=''
    )

    parser.add_argument(
        '--selected_reason',
        nargs='+',
        type=str,
        help=''
    )

    parser.add_argument(
        '--algorithm_arguments',
        nargs='+',
        type=str,
        help=''
    )

    # Results
    parser.add_argument(
        '--raw_result',
        type=np.array,
        default=None,
        help=''
    )

    parser.add_argument(
        '--bootstrap_probability',
        type=np.array,
        default=None,
        help=''
    )

    parser.add_argument(
        '--llm_errors',
        nargs='+',
        type=str,
        help=''
    )

    parser.add_argument(
        '--bootstrap_errors',
        nargs='+',
        type=str,
        help=''
    )

    parser.add_argument(
        '--revised_graph',
        type=np.array,
        default=None,
        help=''
    )

    args = parser.parse_args()
    return args



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


def global_config_initialization(data_path, user_query = None):

    global_config = global_configuration()

    # Load data & ground truth
    _, data, graph = load_data(data_path)

    global_config.raw_data = data
    global_config.ground_truth = graph

    return global_config


