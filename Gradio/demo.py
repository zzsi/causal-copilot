import gradio as gr
import pandas as pd
import io
import os
import shutil
from datetime import datetime
import sys
from queue import Queue
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Gradio.demo_config import get_demo_config
from global_setting.Initialize_state import global_state_initialization
from preprocess.stat_info_functions import stat_info_collection, convert_stat_info_to_text
from preprocess.dataset import knowledge_info
from preprocess.eda_generation import EDA
from algorithm.filter import Filter
from algorithm.program import Programming
from algorithm.rerank import Reranker
from postprocess.judge import Judge
from postprocess.visualization import Visualization
from postprocess.report_generation import Report_generation

# Global variables
UPLOAD_FOLDER = "./demo_data"
chat_history = []
target_path = None
output_dir = None
REQUIRED_INFO = {
    'data_uploaded': False,
    'initial_query': False
}
MAX_CONCURRENT_REQUESTS = 5

# Demo dataset configs
DEMO_DATASETS = {
    "Abalone": {
        "name": "🐚 Abalone",
        "path": "dataset/Abalone/Abalone.csv",
        "query": "use PC, Find causal relationships between physical measurements and age of abalone",
    },
    "Sachs": {
        "name": "🧬 Sachs",
        "path": "dataset/sachs/sachs.csv", 
        "query": "use PC, Discover causal relationships between protein signaling molecules"
    },
    "CCS Data": {
        "name": "📊 CCS Data",
        "path": "dataset/CCS_Data/CCS_Data.csv",
        "query": "use PC, Analyze causal relationships in CCS dataset variables"
    },
    "Ozone": {
        "name": "🌫️ Ozone",
        "path": "dataset/Ozone/Ozone.csv",
        "query": "use PC, Investigate causal factors affecting ozone levels"
    }
}

def upload_file(file):
    # TODO: add more complicated file unique ID handling
    global target_path, output_dir
    
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(file.name).replace('.csv', '')), exist_ok=True)

    target_path = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(file.name).replace('.csv', ''), os.path.basename(file.name))
    output_dir = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(file.name).replace('.csv', ''))
    shutil.copy(file.name, target_path)
    return target_path

def handle_file_upload(file, chat_history, file_upload_btn, download_btn):
    try:
        global REQUIRED_INFO
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
            upload_file(file)
            REQUIRED_INFO['data_uploaded'] = True
            bot_message = f"✅ Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns. What would you like to analyze?"
        else:
            bot_message = "❌ Please upload a CSV file."
            
        chat_history.append((None, bot_message))

        return chat_history, file_upload_btn, download_btn
    except Exception as e:
        error_message = f"❌ Error loading file: {str(e)}"
        chat_history.append((None, error_message))
        return chat_history, file_upload_btn, download_btn

def process_initial_query(message):
    global REQUIRED_INFO
    # TODO: check if the initial query is valid or satisfies the requirements
    REQUIRED_INFO['initial_query'] = True
    if not REQUIRED_INFO['initial_query']:
        chat_history.append((None, "Please enter your initial query first before proceeding. It would be helpful to provide some information about the background/context/prior/statistical information about the dataset."))

def process_message(message, chat_history, download_btn):
    global target_path, REQUIRED_INFO

    if not REQUIRED_INFO['data_uploaded']:
        chat_history.append((message, "Please upload your dataset first before proceeding."))
        return chat_history, download_btn
    
    if not REQUIRED_INFO['initial_query']:
        return chat_history, download_btn

    try:
        # Initialize config and global state
        config = get_demo_config()
        config.data_file = target_path
        config.initial_query = message
        
        args = type('Args', (), {})()
        for key, value in config.__dict__.items():
            setattr(args, key, value)

        # Add user message
        # chat_history.append((message, None))
        # chat_history.append(("🔄 Initializing analysis pipeline...", None))
        global_state = global_state_initialization(args)

        # Load data
        # chat_history.append((None, "📊 Loading and preprocessing data..."))
        global_state.user_data.raw_data = pd.read_csv(target_path)
        global_state.user_data.processed_data = global_state.user_data.raw_data
        # chat_history.append((None, "✅ Data loaded successfully"))
        yield chat_history, download_btn

        # Statistical Analysis
        chat_history.append(("📈 Run statistical analysis...", None))
        yield chat_history, download_btn
        global_state = stat_info_collection(global_state)
        global_state = knowledge_info(args, global_state)
        global_state.statistics.description = convert_stat_info_to_text(global_state.statistics)
        chat_history.append((None, global_state.statistics.description))
        yield chat_history, download_btn

        # EDA Generation
        chat_history.append(("🔍 Generate exploratory data analysis...", None))
        yield chat_history, download_btn
        my_eda = EDA(global_state)
        my_eda.generate_eda()
        chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/eda_corr.jpg',)))
        chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/eda_dist.jpg',)))
        yield chat_history, download_btn

        # Algorithm Selection
        chat_history.append(("🤖 Select optimal causal discovery algorithm...", None))
        yield chat_history, download_btn
        filter = Filter(args)
        global_state = filter.forward(global_state)
        reranker = Reranker(args)
        global_state = reranker.forward(global_state)
        chat_history.append((None, f"✅ Selected algorithm: {global_state.algorithm.selected_algorithm}"))
        yield chat_history, download_btn

        # Causal Discovery
        chat_history.append(("🔄 Run causal discovery analysis...", None))
        yield chat_history, download_btn
        programmer = Programming(args)
        global_state = programmer.forward(global_state)
        judge = Judge(global_state, args)
        global_state = judge.forward(global_state)
        chat_history.append((None, "✅ Causal discovery analysis completed"))
        yield chat_history, download_btn
        
        # Visualization
        chat_history.append(("📊 Generate causal graph visualization...", None))
        yield chat_history, download_btn
        my_visual = Visualization(global_state)
        if global_state.results.raw_result is not None:
            my_visual.plot_pdag(global_state.results.raw_result, 'initial_graph.jpg')
            chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/initial_graph.jpg',)))
        yield chat_history, download_btn

        # Report Generation
        chat_history.append(("📝 Generate comprehensive report...", None))
        yield chat_history, download_btn
        report_gen = Report_generation(global_state, args)
        report = report_gen.generation(debug=True)
        report_gen.save_report(report, save_path=global_state.user_data.output_report_dir)
        
        # Final steps
        chat_history.append((None, "🎉 Analysis complete!"))
        chat_history.append((None, "📥 You can now download your detailed report using the download button below."))

        download_btn = gr.DownloadButton(
            "📥 Download Exclusive Report", 
            size="sm",
            elem_classes=["icon-button"],
            scale=1,
            value=os.path.join(output_dir, 'output_report', 'report.pdf'),
            interactive=True
        )
        yield chat_history, download_btn
        
        chat_history.append((None, ""))
        return chat_history, download_btn

    except Exception as e:
        chat_history.append((None, f"❌ An error occurred during analysis: {str(e)}"))
        return chat_history, download_btn

def clear_chat():
    global target_path, REQUIRED_INFO, output_dir, chat_history
    # Reset global variables
    target_path = None
    output_dir = None
    chat_history = []
    
    # Reset required info flags
    REQUIRED_INFO['data_uploaded'] = False
    REQUIRED_INFO['initial_query'] = False
    
    # Return initial welcome message
    return [(None, "👋 Hello! I'm your causal discovery assistant. Want to discover some causal relationships today?")]

def load_demo_dataset(dataset_name, chatbot, demo_btn, download_btn):
    global target_path, REQUIRED_INFO, output_dir
    dataset = DEMO_DATASETS[dataset_name]
    source_path = dataset["path"]
    
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', '')), exist_ok=True)

    target_path = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', ''), os.path.basename(source_path))
    output_dir = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', ''))
    shutil.copy(source_path, target_path)

    REQUIRED_INFO['data_uploaded'] = True
    REQUIRED_INFO['initial_query'] = True
    
    df = pd.read_csv(target_path)
    bot_message = f"✅ Loaded demo dataset '{dataset_name}' with {len(df)} rows and {len(df.columns)} columns."
    chatbot = chatbot.copy()
    chatbot.append((None, bot_message))
    return chatbot, demo_btn, download_btn, dataset['query']

js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'Welcome to Causal Copilot!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""

with gr.Blocks(js=js, theme=gr.themes.Soft(), css="""
    .input-buttons { 
        position: absolute !important; 
        right: 10px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        display: flex !important;
        gap: 5px !important;
    }
    .icon-button { 
        padding: 0 !important;
        width: 32px !important;
        height: 32px !important;
        border-radius: 16px !important;
        background: transparent !important;
    }
    .icon-button:hover { 
        background: #f0f0f0 !important;
    }
    .icon {
        width: 20px;
        height: 20px;
        margin: 6px;
        display: inline-block;
        vertical-align: middle;
    }
    .message-wrap {
        display: flex !important;
        align-items: flex-start !important;
        gap: 10px !important;
        padding: 15px !important;
    }
    .avatar {
        width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 20px !important;
    }
    .bot-avatar {
        background: #e3f2fd !important;
        color: #1976d2 !important;
    }
    .user-avatar {
        background: #f5f5f5 !important;
        color: #333 !important;
    }
    .message {
        padding: 12px 16px !important;
        border-radius: 12px !important;
        max-width: 80% !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    }
    .bot-message {
        background: #e3f2fd !important;
        margin-right: auto !important;
    }
    .user-message {
        background: #f5f5f5 !important;
        margin-left: auto !important;
    }
""") as demo:
    chatbot = gr.Chatbot(
        value=[(None, "👋 Hello! I'm your causal discovery assistant. Want to discover some causal relationships today?")],
        height=700,
        show_label=False,
        show_share_button=False,
        avatar_images=["https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f600.png", "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f916.png"],
        bubble_full_width=False,
        elem_classes=["message-wrap"],
        render_markdown=True
    )

    with gr.Row():
        with gr.Column(scale=24):
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Enter text here", 
                    elem_classes="input-box", 
                    show_label=False, 
                    container=False,
                    scale=12
                )
                file_upload = gr.UploadButton(
                    "📎 Upload Your Data (.csv)",
                    file_types=[".csv"],
                    size="sm",
                    elem_classes=["icon-button"],
                    scale=5,
                    file_count="single"
                )
                download_btn = gr.DownloadButton(
                    "📥 Download Exclusive Report", 
                    size="sm",
                    elem_classes=["icon-button"],
                    scale=6,
                    interactive=False
                )
                reset_btn = gr.Button("🔄 Reset", scale=1, elem_classes=["icon-button"], size="sm")
    
    # Demo dataset buttons
    with gr.Row():
        for dataset_name in DEMO_DATASETS:
            demo_btn = gr.Button(f"{DEMO_DATASETS[dataset_name]['name']} Demo")
            demo_btn.click(
                fn=load_demo_dataset,
                inputs=[gr.Textbox(value=dataset_name, visible=False), chatbot, demo_btn, download_btn],
                outputs=[chatbot, demo_btn, download_btn, msg],
                queue=True,
                concurrency_limit=MAX_CONCURRENT_REQUESTS
            ).then(
                fn=process_message,
                inputs=[msg, chatbot, download_btn],
                outputs=[chatbot, download_btn],
                queue=True,
                concurrency_limit=MAX_CONCURRENT_REQUESTS
            ).then(
                fn=lambda: "",
                outputs=[msg]
            )

    # Event handlers with queue enabled
    msg.submit(
        fn=process_message,
        inputs=[msg, chatbot, download_btn],
        outputs=[chatbot, download_btn],
        concurrency_limit=MAX_CONCURRENT_REQUESTS,
        queue=True
    ).then(
        fn=lambda: "",
        outputs=[msg]
    )
    
    reset_btn.click(
        fn=clear_chat,
        outputs=[chatbot],
        queue=False  # No need for queue on reset
    )
    
    file_upload.upload(
        fn=handle_file_upload,
        inputs=[file_upload, chatbot, file_upload, download_btn],
        outputs=[chatbot, file_upload, download_btn],
        concurrency_limit=MAX_CONCURRENT_REQUESTS,
        queue=True
    )
    
    # Download report handler with updated visibility
    download_btn.click()

if __name__ == "__main__":
    demo.queue() # Enable queuing at the app level
    demo.launch(share=True)