from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any
import argparse
import pandas as pd
import ast
import numpy as np
import os
import json

from global_setting.state import GlobalState
from data.simulation.simulation import SimulationManager

# Update logic and priority of the state initialization
# 1. All values are intialized to None, later would be set its valid values by information exaction from the user query
# 2. The priority of the state initialization is from the user query > from the data > default values, if previously value is set, the later corresponding operation would be skipped

@dataclass
class UserData:
    raw_data: Optional[pd.DataFrame] = None
    processed_data: Optional[pd.DataFrame] = None
    ground_truth: Optional[np.ndarray] = None
    initial_query: Optional[str] = None
    knowledge_docs: Optional[str] = None

@dataclass
class Statistics:
    linearity: Optional[bool] = None
    gaussian_error: Optional[bool] = None
    missingness: Optional[bool] = None
    sample_size: Optional[int] = None
    feature_number: Optional[int] = None
    boot_num: int = 100
    alpha: float = 0.1
    num_test: int = 100
    ratio: float = 0.5
    data_type: Optional[str] = None
    heterogeneous: Optional[bool] = None
    domain_index: Optional[str] = None
    description: Optional[str] = None

@dataclass
class Logging:
    query_conversation: List[Dict] = field(default_factory=list)
    knowledge_conversation: List[Dict] = field(default_factory=list)
    filter_conversation: List[Dict] = field(default_factory=list)
    select_conversation: List[Dict] = field(default_factory=list)
    argument_conversation: List[Dict] = field(default_factory=list)
    errors_conversion: List[Dict] = field(default_factory=list)

@dataclass
class Algorithm:
    selected_algorithm: Optional[str] = None
    selected_reason: Optional[str] = None
    algorithm_candidates: Optional[Dict] = field(default_factory=dict)
    algorithm_arguments: Dict = field(default_factory=dict)

@dataclass
class Results:
    raw_result: Optional[object] = None
    raw_info: Optional[Dict] = None
    converted_graph: Optional[str] = None
    metrics: Optional[Dict] = None
    revised_graph: Optional[np.ndarray] = None
    revised_metrics: Optional[Dict] = None
    bootstrap_probability: Optional[np.ndarray] = None
    llm_errors: List[Dict] = field(default_factory=list)
    bootstrap_errors: List[Dict] = field(default_factory=list)

@dataclass
class GlobalState:
    user_data: UserData = field(default_factory=UserData)
    statistics: Statistics = field(default_factory=Statistics)
    logging: Logging = field(default_factory=Logging)
    algorithm: Algorithm = field(default_factory=Algorithm)
    results: Results = field(default_factory=Results)

def global_state_initialization(data_path: str, user_query: str = None) -> GlobalState:
    global_state = GlobalState()
    global_state.user_data.initial_query = user_query
    # TODO: initial the state with the exaction information from user query

    return global_state

def load_data(global_state: GlobalState, args: argparse.Namespace):
    if args.simulation_mode == "online":
        simulation_manager = SimulationManager(args)
        config, data, graph = simulation_manager.generate_dataset()
    elif args.simulation_mode == "offline":
        if args.data_mode == "simulated":
            config, data, graph = load_local_data(args.data_file)
        elif args.data_mode == "real":
            data = pd.read_csv(args.data_file)
            graph = None
        else:
            raise ValueError("Invalid data mode. Please choose 'real' or 'simulated'.")
    else:
        raise ValueError("Invalid simulation mode. Please choose 'online' or 'offline'.")
    
    global_state.user_data.raw_data = data
    global_state.user_data.ground_truth = graph

    # hard-coded heterogeneous and domain index, later would be set by the user query
    if 'domain_index' in data.columns:
        if data.nunique(axis=0)['domain_index'] > 1:
            global_state.statistics.heterogeneous = True
        else:
            global_state.statistics.heterogeneous = False
        global_state.statistics.domain_index = 'domain_index'

    return global_state

def load_local_data(directory: str):
    # Xinyue Wang Implemented
    '''
    :param directory: str for data directory
    :return: tuple of (config, data, graph)
    '''
    import json
    import os
    import pandas as pd
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
    else:
        graph = None

    return config, data, graph


def load_data(global_state: GlobalState, args: argparse.Namespace):
    if args.simulation_mode == "online":
        simulation_manager = SimulationManager(args)
        config, data, graph = simulation_manager.generate_dataset()
    elif args.simulation_mode == "offline":
        if args.data_mode == "simulated":
            config, data, graph = load_local_data(args.data_file)
        elif args.data_mode == "real":
            data = pd.read_csv(args.data_file)
            graph = None
        else:
            raise ValueError("Invalid data mode. Please choose 'real' or 'simulated'.")
    else:
        raise ValueError("Invalid simulation mode. Please choose 'online' or 'offline'.")

    global_state.user_data.raw_data = data
    global_state.user_data.ground_truth = graph

    # hard-coded heterogeneous and domain index, later would be set by the user query
    if 'domain_index' in data.columns:
        if data.nunique(axis=0)['domain_index'] > 1:
            global_state.statistics.heterogeneous = True
        else:
            global_state.statistics.heterogeneous = False
        global_state.statistics.domain_index = 'domain_index'

    return global_state

def global_state_initialization(args: argparse.Namespace = None) -> GlobalState:
    user_query = args.initial_query
    global_state = GlobalState()
    #user_query = "I’m analyzing a biological dataset with 2000 samples and 12 features. The data has some missing values, and I assume the relationships between the variables are linear. I believe the error terms follow a Gaussian distribution. The dataset is homogeneous, and there’s no specific domain index. I’d like to use the NOTEARS algorithm for causal discovery. Can you guide me through the process?"

    global_state.user_data.initial_query = user_query

    # Extract information from user queries
    from openai import OpenAI

    # organization = "org-5NION61XDUXh0ib0JZpcppqS"
    # project = "proj_Ry1rvoznXAMj8R2bujIIkhQN"
    # apikey = "sk-l4ETwy_5kOgNvt5OzHf_YtBevR1pxQyNrlW8NRNPw2T3BlbkFJdKpqpbcDG0IhInYcsS3CXdz_EMHkJO7s1Bo3e4BBcA"

    client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
    prompt = (f"Using the query that I provided: {user_query} \n\n, "
              "and extract the following information for me based on the format of the example I provided. "
              "For example, if my query is: \n\n "
              "The relationship between the variables are assumed to be linear (for this kind of information, you just need to extract if the relationship is linear or non-linear).\n\n "
              "The errors are assumed to be Gaussian (for this kind of information, you just need to extract if the error follows Gaussian distribution or not).\n\n "
              "There is missingness in the dataset (for this kind of information, you just need to extract if there is missing values or not in the dataset).\n\n "
              "The suggested significant level for in the analysis is 0.05 (it should be a value that is larger than 0 and smaller than 1).\n\n "
              "The datatype is mixture, which means it contain both continuous and categorical variables. Other options for data type is 'Continuous' or 'Category'.\n\n "
              "The data is heterogeneous (for this kind of information, you just need to extract if the data is heterogeneous or not).\n\n "
              "The feature which represents the domain index is 'sea_level' in the dataset. (for this kind of information, you just need to extract the column name that stands for the domain index in the dataset).\n\n"
              "I would like to use PC to do causal discovery. Other options for algorithm include FCI, CDNOD, GES, NOTEARS, DirectLiNGAM, ICALiNGAM. \n\n"
              "Then you should give me the output that is in a json format: \n\n"
              "{'linearity': True, 'gaussian_error': True, 'missingness': True, 'alpha': 0.05, 'data_type': 'Mixture', 'heterogeneous': True, 'domain_index':'sea_level', 'selected_algorithm': 'PC'}.  \n\n"
              "If the information does not match the options I provided above or the queries do not provide such information, you would give value of None for such key !"
              "Just give me the output in a dict format, do not provide other information! \n\n")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # The output from GPT is str
    info_extracted = response.choices[0].message.content
    info_extracted = json.loads(info_extracted)


    # Assign extracted information from user queries to global_stat
    global_state.statistics.linearity = info_extracted["linearity"]
    global_state.statistics.gaussian_error = info_extracted["gaussian_error"]
    global_state.statistics.missingness = info_extracted["missingness"]
    global_state.statistics.alpha = info_extracted["alpha"]
    global_state.statistics.data_type = info_extracted["data_type"]
    global_state.statistics.heterogeneous = info_extracted["heterogeneous"]
    global_state.statistics.domain_index = info_extracted["domain_index"]
    global_state.algorithm.selected_algorithm = info_extracted["selected_algorithm"]

    # Load Data
    # _, data, graph = load_local_data(data_path)
    # global_state.user_data.raw_data = data
    # global_state.user_data.ground_truth = graph

    # Statistics Information Collection


    return global_state





