from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import argparse
import pandas as pd
import numpy as np
import os
import json
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
    algorithm_candidates: Dict[Dict] = field(default_factory=dict)
    algorithm_candidate_name: Optional[str] = None
    algorithm_candidate_desc: Optional[str] = None
    algorithm_candidate_just: Optional[str] = None
    algorithm_arguments: Dict = field(default_factory=dict)

@dataclass
class Results:
    raw_result: Optional[np.ndarray] = None
    metrics: Optional[Dict] = None
    bootstrap_probability: Optional[np.ndarray] = None
    llm_errors: List[Dict] = field(default_factory=list)
    bootstrap_errors: List[Dict] = field(default_factory=list)
    revised_graph: Optional[np.ndarray] = None

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
