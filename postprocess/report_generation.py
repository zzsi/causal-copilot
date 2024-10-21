from openai import OpenAI
import os
import argparse
import networkx as nx
from mdprint import mdprint

from postprocess.judge_functions import graph_effect_prompts

class Report_generation(object):
    def __init__(self, global_state, args):
        """
        :param global_state: a dict containing global variables and information
        :param args: arguments for the report generation
        """

        self.client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
        self.data_mode = args.data_mode
        self.statistics_desc = global_state.statistics.description
        self.knowledge_docs = global_state.user_data.knowledge_docs[0]
        # Data info
        self.data = global_state.user_data.raw_data
        # EDA info
        self.eda_result = global_state.results.eda
        # Result graph matrix
        self.graph = global_state.results.converted_graph
        self.bootstrap_probability = global_state.results.bootstrap_probability
        self.original_metrics = global_state.results.metrics
        self.revised_metrics = global_state.results.revised_metrics
        # algo&hp selection prompts
        self.prompt = global_state.logging.select_conversation[0]['response']
        self.hp_prompt = global_state.logging.argument_conversation[0]['response']
        # Path to find the visualization graph
        self.visual_dir = args.output_graph_dir

    def eda_prompt(self):
        dist_input = self.eda_result['dist_analysis']
        corr_input = self.eda_result['corr_analysis']
        prompt_dist = (
            f"""
            Given the following statistics about features in a dataset:\n\n
            {dist_input}\n
            Please provide an analysis of the distributions of these features. 
            Please categorize variables according to their distribution features.
            For example, you can say:
            - Slight left skew distributed variables: Length, ...
            - Slight right skew distributed variables: Whole Weight, ...
            - Symmetric distributed variables: Height, ...
            """
        )
        prompt_corr = (
            f"""
            Given the following correlation statistics about features in a dataset:\n\n
            {corr_input}\n
            Please provide an analysis of the correlations of these features.
            You can seperate your analysis into three categories: Strong correlations, Moderate correlations, and Weak correlations.            
            """
        )

        print("Start to find EDA Description")
        response_dist = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt_dist}
            ]
        )
        response_dist_doc = response_dist.choices[0].message.content

        response_corr = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt_corr}
            ]
        )
        response_corr_doc = response_corr.choices[0].message.content

        return response_dist_doc, response_corr_doc

    def discover_process_prompts(self):
        prompt = f"""
        I want to describe the causal discovery procedure of this project. 
        Firstly, we preprocessed the data and checked statistical characteristics of this dataset.
        
        Then we let the LLM help us to select algorithms and hyper-parameters based on statistical characteristics of this dataset and background knowledge.
        This is the prompt for algorithm selection {self.prompt}, and this is the prompt for hyper-parameter selection {self.hp_prompt},
        you can find some useful information, i.e. the selected algorithms and parameters, and justifications for them in these prompts.
        You should include which algorithm we choose, what hyperparameters we choose, and justifications for them in the report.
        The first step is Data Preprocessing, the second Algorithm Selection assisted with LLM, the third is Hyperparameter Values Proposal assisted with LLM,
        and the fourth is graph tuning with bootstrap and LLM suggestion.
        Please discribe the procedure step by step clearly, and also present the choosen parameters in json format.
        """
        print("Start to find discovery procedure")
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        response_doc = response.choices[0].message.content
        return response_doc

    def graph_effect_prompts(self):
        variables = self.data.columns
        G = nx.from_numpy_array(self.graph.T, parallel_edges=False, create_using=nx.DiGraph)
        relations = [(variables[index[0]],variables[index[1]]) for index in G.edges]

        prompt = f"""
        This list of tuples reflects the causal relationship among variables {relations}.
        For example, if the tuple is (X1, X0), it means that {variables[1]} causes {variables[0]}, that is {variables[1]} -> {variables[0]}.
        Please write a paragraph to describe the causal relationship, and you can add some analysis. 
        Don't mention tuples in the paragraph, 
        Please use variable names {variables[0]}, {variables[1]}, ... in your description.
        For example, you can begin in this way:
        The result graph shows the causal relationship among variables clearly. The {variables[1]} causes the {variables[0]}, ...
        """

        print("Start to find graph effect")
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        response_doc = response.choices[0].message.content
        return response_doc

    def confidence_analysis_prompts(self):

        relation_prob = graph_effect_prompts(self.data,
                                             self.graph,
                                             self.bootstrap_probability)

        variables = '\t'.join(self.data.columns)
        print(relation_prob)
        prompt = f"""
        Now we have a causal relationship about these variables:{variables}, and we want to analize the reliability of it.
        The following describes how much confidence we have on each relationship edge: {relation_prob}.
        For example, if it says X1 -> X0 (the bootstrap probability of such edge is 0.99), it means that we have 99% confidence to believe that X1 causes X0.
        The following is the background knowledge about these variables: {self.knowledge_docs}
        Based on this statistical confidence result, and background knowledge about these variables,
        Please write a paragraph to analyze the reliability of this causal relationship graph. 
        
        For example, you can write in the following way, and please analyze 1. the reliability and 2. give conclusion 
        base on both bootstrap probability and expert knowledge background.
        Template:
        From the Statistics perspective, we have high confidence to believe that these edges exist:..., and these edges don't exist:...
        However, based on the expert knowledge, we know that these edges exist:...., and these edges don't exist:... 
        Therefore, the result of this causal graph is reliable/not reliable.
        """

        print("Start to analyze graph reliability")
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        response_doc = response.choices[0].message.content
        return response_doc

    def load_context(self, filepath):
        with open(filepath, "r") as f:
            return f.read()

    def generation(self):
            '''
            generate and save the report
            :return: Str: A technique report explaining all the results for readers
            '''
            # Data info
            data_preview = self.data.head().to_string()
            data_prop = self.statistics_desc
            # Background info
            background_info = self.knowledge_docs
            # EDA info
            dist_info, corr_info = self.eda_prompt()
            # Graph effect info
            graph_prompt = self.graph_effect_prompts()
            discover_process = self.discover_process_prompts()
            # Graph Reliability info
            reliability_prompt = self.confidence_analysis_prompts()
            # Graph paths
            graph_path0 = f'{self.visual_dir}/True_Graph.jpg'
            graph_path1 = f'{self.visual_dir}/Initial_Graph.jpg'
            graph_path2 = f'{self.visual_dir}/Revised_Graph.jpg'
            graph_path3 = f'{self.visual_dir}/metrics.jpg'
            graph_path4 = f'{self.visual_dir}/confidence_heatmap.jpg'
            # EDA Graph paths
            dist_graph_path = self.eda_result['plot_path_dist']
            scat_graph_path = self.eda_result['plot_path_scat']
            corr_graph_path = self.eda_result['plot_path_corr']

            if self.data_mode == 'simulated':
                # Report prompt
                prompt_template = self.load_context("postprocess/context/report_templete_simulated")
                replacements = {
                    "[BACKGROUND_INFO]": background_info,
                    "[DATA_PREVIEW]": data_preview,
                    "[DATA_PROP]": data_prop,
                    "[DIST_INFO]": dist_info,
                    "[CORR_INFO]": corr_info,
                    "[DIST_GRAPH]": dist_graph_path,
                    "[SCAT_GRAPH]": scat_graph_path,
                    "[CORR_GRAPH]": corr_graph_path,
                    "[RESULT_ANALYSIS]": graph_prompt,
                    "[DISCOVER_PROCESS]": discover_process,
                    "[RELIABILITY_ANALYSIS]": reliability_prompt,
                    "[RESULT_GRAPH0]": graph_path0,
                    "[RESULT_GRAPH1]": graph_path1,
                    "[RESULT_GRAPH2]": graph_path2,
                    "[RESULT_GRAPH3]": graph_path3,
                    "[RESULT_GRAPH4]": graph_path4,
                    "[RESULT_METRICS1]": str(self.original_metrics),
                    "[RESULT_METRICS2]": str(self.revised_metrics)
                }
            else:
                # Report prompt
                prompt_template = self.load_context("postprocess/context/report_templete_real")
                replacements = {
                    "[BACKGROUND_INFO]": background_info,
                    "[DATA_PREVIEW]": data_preview,
                    "[DATA_PROP]": data_prop,
                    "[DIST_INFO]": dist_info,
                    "[CORR_INFO]": corr_info,
                    "[DIST_GRAPH]": dist_graph_path,
                    "[SCAT_GRAPH]": scat_graph_path,
                    "[CORR_GRAPH]": corr_graph_path,
                    "[RESULT_ANALYSIS]": graph_prompt,
                    "[DISCOVER_PROCESS]": discover_process,
                    "[RELIABILITY_ANALYSIS]": reliability_prompt,
                    "[RESULT_GRAPH1]": graph_path1,
                    "[RESULT_GRAPH2]": graph_path2,
                    "[RESULT_GRAPH4]": graph_path4
                }

            for placeholder, value in replacements.items():
                prompt_template = prompt_template.replace(placeholder, value)


            print("Start to generate the report")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a causal discovery expert. Provide your response in Markdown format."},
                    {"role": "user", "content": prompt_template}
                ]
            )

            output = response.choices[0].message.content
            return output

    def save_report(self, report, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(f'{save_path}/report.txt', 'w') as file:
            # Write some text to the file
            file.write(report)
        with open(f'{save_path}/report.md', 'w') as f:
            mdprint(report, file=f)

def parse_args():
    parser = argparse.ArgumentParser(description='Args setup for Report Generation')

    # Input data file
    parser.add_argument(
        '--data-file',
        type=str,
        default="postprocess/test_data/20241007_184921_base_nodes8_samples1500/base_data.csv",
        help='Path to the input dataset file (e.g., CSV format)'
    )

    # Input data file of Result Graph
    parser.add_argument(
        '--result-graph',
        type=str,
        default="postprocess/test_data/20241007_184921_base_nodes8_samples1500/revised_graph.npy",
        help='Path to the result graph matrix (e.g., npy format)'
    )

    # Directory of visual graph
    parser.add_argument(
        '--visual-dir',
        type=str,
        default="postprocess/test_data/20241007_184921_base_nodes8_samples1500/output_graph",
        help='Directory of the visualization graph'
    )

    # Target variable
    parser.add_argument(
        '--target-variable',
        type=str,
        help='Name of the target variable in the dataset'
    )

    # Output file for results
    parser.add_argument(
        '--output-file',
        type=str,
        default='results.txt',
        help='File path to save the analysis results'
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

def main():
    args_setup = parse_args()
    statistics_dict = {
                "Missingness": False,
                "Data Type": "Categorical",
                "Linearity": False,
                "Gaussian Error": True,
                "Stationary": "non time-series"
            }

    my_report = Report_generation(args_setup, statistics_dict)
    report = my_report.generation()
    save_path = 'postprocess/test_data/20241007_184921_base_nodes8_samples1500/output_report'
    my_report.save_report(report, save_path)

if __name__ == '__main__':
    main()
