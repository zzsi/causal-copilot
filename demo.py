import gradio as gr
import pandas as pd
import os
import sys
import json
import numpy as np
from main import parse_args, main as run_main
from global_setting.state import GlobalState

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

def ensure_output_dirs():
    """Create output directories if they don't exist"""
    os.makedirs(os.path.join(DEMO_DIR, 'data/simulation/simulated_data'), exist_ok=True)

def generate_sample_data():
    """Generate sample data if no file is uploaded"""
    n_samples = 1000
    n_variables = 5
    data = np.random.randn(n_samples, n_variables)
    columns = [f'X{i}' for i in range(n_variables)]
    df = pd.DataFrame(data, columns=columns)
    df['X1'] = df['X0'] * 0.5 + np.random.randn(n_samples) * 0.1
    df['X2'] = df['X1'] * 0.3 + df['X0'] * 0.2 + np.random.randn(n_samples) * 0.1
    sample_path = os.path.join(DEMO_DIR, 'gradio_output', 'sample_data.csv')
    df.to_csv(sample_path, index=False)
    return sample_path

def process_message(history, file_obj, message):
    """Process user message and file uploads"""
    try:
        # Change to demo directory
        original_dir = os.getcwd()
        os.chdir(DEMO_DIR)
        ensure_output_dirs()

        # Handle file upload
        if file_obj is not None:
            temp_path = os.path.join(DEMO_DIR, 'gradio_output', 'base_data.csv')
            if hasattr(file_obj, 'name'):
                df = pd.read_csv(file_obj.name)
                df.to_csv(temp_path, index=False)
                file_path = temp_path
            else:
                file_path = file_obj
            bot_response = "I've received your data file. What would you like to analyze?"
            return history + [[message, bot_response]]
        else:
            file_path = generate_sample_data()

        # Create args for main processing
        class Args:
            pass
        
        args = Args()
        args.data_file = file_path
        args.data_mode = "simulated"
        args.simulation_mode = "offline"
        args.initial_query = message
        args.debug = False
        args.parallel = True
        args.organization = "org-5NION61XDUXh0ib0JZpcppqS"
        args.project = "proj_Ry1rvoznXAMj8R2bujIIkhQN"
        args.apikey = "sk-l4ETwy_5kOgNvt5OzHf_YtBevR1pxQyNrlW8NRNPw2T3BlbkFJdKpqpbcDG0IhInYcsS3CXdz_EMHkJO7s1Bo3e4BBcA"

        # Run analysis
        report, global_state = run_main(args)
        

        # Load results
        try:
            with open(os.path.join(DEMO_DIR, 'gradio_output/report', 'report.json'), 'r') as f:
                report = json.load(f)
        except FileNotFoundError:
            report = {
                'summary': 'Analysis completed but no report was generated.',
                'causal_relationships': '',
                'recommendations': ''
            }

        # Create conversation flow with text and images
        conversation = []
        
        # Add user message
        conversation.append([message, None])
        
        # Add initial text response
        text_response = f"""Here's what I found:

**Summary:**
{report.get('summary', 'No summary available')}

**Causal Relationships:**
{report.get('causal_relationships', 'No relationships identified')}

**Recommendations:**
{report.get('recommendations', 'No recommendations available')}"""
        
        conversation.append([None, text_response])

        # Add visualizations as separate messages
        image_paths = {
            "Initial Causal Graph": os.path.join(, 'gradio_output/graph', 'initial_graph.png'),
            "Revised Causal Graph": os.path.join(DEMO_DIR, 'gradio_output/graph', 'revised_graph.png'),
            "Performance Metrics": os.path.join(DEMO_DIR, 'gradio_output/graph', 'metrics_plot.png'),
            "Bootstrap Analysis": os.path.join(DEMO_DIR, 'gradio_output/graph', 'boot_heatmap.png')
        }

        for title, path in image_paths.items():
            if os.path.exists(path):
                conversation.append([None, f"Here's the {title}:"])
                conversation.append([None, (path,)])

        os.chdir(original_dir)
        return history + conversation

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return history + [[message, error_message]]

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Causal Learning Assistant")
    
    chatbot = gr.Chatbot(
        [],
        height=700,
        show_label=False
    )
    
    with gr.Row():
        file_input = gr.File(
            label="Upload Data (CSV)",
            file_types=[".csv"]
        )
        msg = gr.Textbox(
            show_label=False,
            placeholder="Enter your message here...",
            container=False
        )
    
    gr.Examples(
        examples=[
            "What are the main causal relationships in this dataset?",
            "Can you analyze the temporal patterns in the data?",
            "What recommendations do you have based on the causal analysis?"
        ],
        inputs=msg
    )

    msg.submit(
        process_message,
        [chatbot, file_input, msg],
        [chatbot]
    )

if __name__ == "__main__":
    ensure_output_dirs()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
