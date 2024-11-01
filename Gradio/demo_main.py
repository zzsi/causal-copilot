from rich.diagnose import report

from Gradio.base import Interface
import gradio as gr

from Run1 import data_path
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
import os
import shutil
import io

import json
import argparse
import pandas as pd
from main import parse_args
import sys
import time

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

# Global chat history
chat_history = []


def process_query(user_query,args):
    global chat_history

    chat_history.append({"role": "user", "content": user_query})

    assistant_reply = "We are processing ..."
    chat_history.append({"role": f"assistant", "content": assistant_reply})
    yield chat_history

    # Initialize Global States
    args = parse_args()
    args.data_file = target_path
    global_state = global_state_initialization(args)

    # Update user query and raw data and path
    global_state.user_data.initial_query = str(user_query)
    global_state.user_data.raw_data = pd.read_csv(os.path.join('./uploaded_data', uploaded_file_name))

    # Statistics Information Collection
    global_state = stat_info_collection(global_state)
    global_state = knowledge_info(args, global_state)

    # Convert statistics to text
    global_state.statistics.description = convert_stat_info_to_text(global_state.statistics)

    # Show intermediate results
    chat_history.append({"role": f"assistant", "content": global_state.statistics.description})
    yield chat_history

    chat_history.append({"role": f"assistant", "content": str(global_state.user_data.knowledge_docs)})
    yield chat_history

    ############ EDA ######################################

    my_eda = EDA(global_state)
    my_eda.generate_eda()


    chat_history.append({"role": f"assistant", "content": f'{global_state.user_data.output_graph_dir}/eda_corr.jpg'})
    yield chat_history

    chat_history.append({"role": f"assistant", "content": f'{global_state.user_data.output_graph_dir}/eda_dist.jpg'})
    yield chat_history

    ############ Algorithm Selection ######################################
    os.chdir("/Users/fangnan/Library/CloudStorage/OneDrive-UCSanDiego/UCSD/ML Research/Causality-Copilot")

    filter = Filter(args)
    global_state = filter.forward(global_state)

    mystdout = io.StringIO()

    old_stdout = sys.stdout
    sys.stdout = mystdout

    try:
        reranker = Reranker(args)
        global_state = reranker.forward(global_state)

        programmer = Programming(args)
        global_state = programmer.forward(global_state)

        judge = Judge(global_state, args)
        global_state = judge.forward(global_state)
    finally:
        sys.stdout = old_stdout

    output = mystdout.getvalue()
    chat_history.append({"role": f"assistant", "content": output})
    yield chat_history


    # #############Visualization for Initial Graph###################
    print(os.getcwd())
    my_visual_initial = Visualization(global_state)

    assistant_reply = "We are processing ..."
    chat_history.append({"role": f"assistant", "content": assistant_reply})
    yield chat_history


    # #############Visualization for Revised Graph###################
    my_visual_revise = Visualization(global_state)

    #############Report Generation###################
    my_report = Report_generation(global_state, args)
    report = my_report.generation()
    my_report.save_report(report, save_path=global_state.user_data.output_report_dir)


if __name__ == "__main__":
    interface = Interface()

    interface.prepare_interface(
        upload_file=upload_file,
        serve_file=serve_file,
        process_query=process_query)
