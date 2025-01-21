import os
import numpy as np
import pandas as pd 
from causal_analysis.inference import Analysis
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pydantic import BaseModel
from causal_analysis.help_functions import LLM_parse_query

def main(global_state, args):
    """
    Modify the main function to call attribute_anomalies and save the results in ./auto_mpg_output.
    """
    print("Welcome to the Causal Analysis Demo using the sachs dataset.\n")
    analysis = Analysis(global_state, args)
    message = input("Is there any causal analysis problem I can help you? \n")

    class InfList(BaseModel):
                tasks: list[str]
                descriptions: list[str]
                key_node: list[str]
    columns = global_state.user_data.processed_data.columns
    with open('causal_analysis/context/query_prompt.txt', 'r') as file:
        query_prompt = file.read()
        query_prompt = query_prompt.replace('[COLUMNS]', columns)
    
    global_state.logging.downstream_discuss.append({"role": "user", "content": message})
    parsed_response = LLM_parse_query(args, InfList, query_prompt, message)
    tasks_list, descs_list, key_node_list = parsed_response.tasks, parsed_response.descriptions, parsed_response.key_node
    print(tasks_list, descs_list, key_node_list)
    #tasks_list, descs_list, key_node_list = ['Treatment Effect Estimation'], ['Analyze the treatment effect of PIP2 to PIP3.'], ['PIP3']
    for i, (task, desc, key_node) in enumerate(zip(tasks_list, descs_list, key_node_list)):
        print(task, desc, key_node)
        response, figs = analysis.forward(task, desc, key_node)
        print(response)
        for file_name in figs:
            img = mpimg.imread(f'{global_state.user_data.output_graph_dir}/{file_name}')  # Read the image
            plt.imshow(img)  
    
    
if __name__ == '__main__':
    import argparse
    import pickle
    def parse_args():
        parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

        # Input data file
        parser.add_argument(
            '--data-file',
            type=str,
            default="dataset/sachs/sachs.csv",
            help='Path to the input dataset file (e.g., CSV format or directory location)'
        )

        # Output file for results
        parser.add_argument(
            '--output-report-dir',
            type=str,
            default='causal_analysis/test_result',
            help='Directory to save the output report'
        )

        # Output directory for graphs
        parser.add_argument(
            '--output-graph-dir',
            type=str,
            default='causal_analysis/test_result',
            help='Directory to save the output graph'
        )

        # OpenAI Settings
        parser.add_argument(
            '--organization',
            type=str,
            default="org-gw7mBMydjDsOnDlTvNQWXqPL",
            help='Organization ID'
        )

        parser.add_argument(
            '--project',
            type=str,
            default="proj_SIDtemBJMHUWG7CPdU7yRjsn",
            help='Project ID'
        )

        parser.add_argument(
            '--apikey',
            type=str,
            default=None,
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
            default="real",
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
            default="selected algorithm: PC",
            help='Initial query for the algorithm'
        )

        parser.add_argument(
            '--parallel',
            type=bool,
            default=False,
            help='Parallel computing for bootstrapping.'
        )

        parser.add_argument(
            '--demo_mode',
            type=bool,
            default=False,
            help='Demo mode'
        )

        parser.add_argument(
            '--revised_graph',
            type=str,
            default='dataset/sachs/base_graph.npy',
            help='Demo mode'
        )

        args = parser.parse_args()
        return args
    with open('report/test/args.pkl', 'rb') as file:
        args = pickle.load(file)
    with open('report/test/global_state.pkl', 'rb') as file:
        global_state = pickle.load(file)

    main(global_state, args)
