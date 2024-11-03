from rich.diagnose import report

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Gradio.base import Interface
import gradio as gr

from preprocess.dataset import knowledge_info
from preprocess.stat_info_functions import stat_info_collection, convert_stat_info_to_text
from algorithm.filter import Filter
from algorithm.program import Programming
from algorithm.rerank import Reranker
from postprocess.judge import Judge
from postprocess.visualization import Visualization
from preprocess.eda_generation import EDA
from postprocess.report_generation import Report_generation
from global_setting.Initialize_state import global_state_initialization, load_data
import shutil
import io

import json
import argparse
import pandas as pd
from main import parse_args
import sys
import time

from Gradio.demo_config import get_demo_config

def upload_file(file):
    global uploaded_file_name
    global target_path

    UPLOAD_FOLDER = "./uploaded_data"
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

    uploaded_file_name = os.path.basename(file.name)

    target_path = os.path.join(UPLOAD_FOLDER, uploaded_file_name)
    shutil.copy(file.name, target_path)

def serve_file():
    # Path of saved report
    file_path = f'output_report/report.pdf'
    return file_path

# Global variables
chat_history = []
REQUIRED_INFO = {
    'data_uploaded': False,
    'initial_query': False
}

def check_requirements():
    return all(REQUIRED_INFO.values())

def get_initial_guidance():
    return """Welcome to the Causal Discovery Assistant! To begin the analysis, please:
1. Upload your dataset (CSV format)
2. Describe what you want to analyze in your data (e.g., 'Find causal relationships between variables X and Y')

What would you like to analyze?"""

def get_missing_requirements():
    if not REQUIRED_INFO['data_uploaded']:
        return "Please upload your dataset first before proceeding."
    if not REQUIRED_INFO['initial_query']:
        return "Please describe what you want to analyze in your data."
    return None

def process_query(user_query):
    print("Starting process_query with user_query:", user_query)
    global chat_history
    global REQUIRED_INFO
    global target_path

    # Add user query to chat history if not empty
    if user_query:
        chat_history.append({"role": "user", "content": user_query})
        print("Added user query to chat history")
        yield chat_history

    # Show initial guidance if this is the first interaction
    if len(chat_history) == 1:
        print("First interaction - sending initial guidance")
        initial_message = get_initial_guidance()
        chat_history.append({"role": "assistant", "content": initial_message})
        yield chat_history
        return

    # Update REQUIRED_INFO based on user input
    if user_query and not REQUIRED_INFO['initial_query']:
        print("Setting initial_query flag to True")
        REQUIRED_INFO['initial_query'] = True

    # Check requirements
    missing_req = get_missing_requirements()
    if missing_req:
        print("Missing requirements:", missing_req)
        chat_history.append({"role": "assistant", "content": missing_req})
        yield chat_history
        return

    # Start analysis pipeline
    try:
        print("Starting analysis pipeline")
        # Initialize config and global state
        config = get_demo_config()
        config.data_file = target_path
        config.initial_query = user_query
        print("Config initialized with data_file:", target_path)
        
        args = type('Args', (), {})()
        for key, value in config.__dict__.items():
            setattr(args, key, value)
        print("Args created from config")    
            
        global_state = global_state_initialization(args)
        print("Global state initialized")

        # Load data
        print("Loading data from:", target_path)
        global_state.user_data.raw_data = pd.read_csv(target_path)
        global_state.user_data.processed_data = global_state.user_data.raw_data
        print("Data loaded successfully")

        # Simulate interactive stages with fake user acknowledgments
        stages = [
            ("Starting statistical analysis...", "Please proceed with the statistical analysis."),
            ("Analyzing statistical properties of your dataset...", None),
            ("Starting exploratory data analysis...", "Please show me the EDA results."),
            ("Generating visualizations...", None),
            ("Selecting appropriate causal discovery algorithm...", "What algorithm was selected?"),
            ("Running causal discovery analysis...", "Please show me the results."),
            ("Generating final report...", "Can I see the report?")
        ]

        print("Starting stages processing")
        for system_msg, user_ack in stages:
            print(f"\nProcessing stage: {system_msg}")
            # System message
            chat_history.append({"role": "assistant", "content": system_msg})
            yield chat_history
            
            # If there's a user acknowledgment, add it
            if user_ack:
                print("Adding user acknowledgment:", user_ack)
                chat_history.append({"role": "user", "content": user_ack})
                yield chat_history

            # Process each stage
            if "statistical analysis" in system_msg:
                print("Running statistical analysis")
                global_state = stat_info_collection(global_state)
                global_state = knowledge_info(args, global_state)
                global_state.statistics.description = convert_stat_info_to_text(global_state.statistics)
                chat_history.append({"role": "assistant", "content": global_state.statistics.description})
                yield chat_history

            elif "visualizations" in system_msg:
                print("Generating EDA visualizations")
                my_eda = EDA(global_state)
                my_eda.generate_eda()
                # Add EDA plots to chat
                print("Adding EDA plots to chat")
                chat_history.append({"role": "assistant", 
                                   "content": f'{global_state.user_data.output_graph_dir}/eda_corr.jpg'})
                chat_history.append({"role": "assistant", 
                                   "content": f'{global_state.user_data.output_graph_dir}/eda_dist.jpg'})
                yield chat_history

            elif "algorithm" in system_msg:
                print("Running algorithm selection")
                filter = Filter(args)
                global_state = filter.forward(global_state)
                reranker = Reranker(args)
                global_state = reranker.forward(global_state)
                print("Selected algorithm:", global_state.algorithm.selected_algorithm)
                chat_history.append({"role": "assistant", 
                                   "content": f"Selected algorithm: {global_state.algorithm.selected_algorithm}"})
                yield chat_history

            elif "causal discovery analysis" in system_msg:
                print("Running causal discovery analysis")
                programmer = Programming(args)
                global_state = programmer.forward(global_state)
                judge = Judge(global_state, args)
                global_state = judge.forward(global_state)
                
                my_visual = Visualization(global_state)
                if global_state.results.raw_result is not None:
                    print("Generating causal graph visualization")
                    graph_path = f'{global_state.user_data.output_graph_dir}/causal_graph.pdf'
                    my_visual.plot_pdag(global_state.results.raw_result, graph_path)
                    chat_history.append({"role": "assistant", "content": graph_path})
                    yield chat_history

            elif "final report" in system_msg:
                print("Generating final report")
                report_gen = Report_generation(global_state, args)
                report = report_gen.generation(debug=True)
                report_gen.save_report(report, save_path=global_state.user_data.output_report_dir)
                print("Report generated and saved")

        # Final message
        print("Analysis complete")
        chat_history.append({"role": "assistant", 
                           "content": "Analysis complete! You can now download the detailed report using the download button."})
        yield chat_history

    except Exception as e:
        error_msg = f"An error occurred during analysis: {str(e)}"
        chat_history.append({"role": "assistant", "content": error_msg})
        yield chat_history

def file_upload_trigger(file):
    global REQUIRED_INFO
    if file is not None:
        REQUIRED_INFO['data_uploaded'] = True
        upload_file(file)
    return None

if __name__ == "__main__":
    interface = Interface()
    
    interface.prepare_interface(
        upload_file=file_upload_trigger,
        serve_file=serve_file,
        process_query=process_query)
