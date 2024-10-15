from openai import OpenAI
import os
import argparse
import networkx as nx
from mdprint import mdprint

class Report_generation(object):
    def __init__(self, args_setup, data, statistics_desc, knowledge_docs,
                 prompt, hp_prompt, result_graph, visual_dir):
        """
        :param args_setup: arguments for the report generation
        :param statistics_desc: statistics description of the dataset
        """
        self.client = OpenAI(organization=args_setup.organization, project=args_setup.project, api_key=args_setup.apikey)
        self.statistics_desc = statistics_desc
        self.knowledge_docs = knowledge_docs
        # Data info
        self.data = data
        # Result graph matrix
        self.graph = result_graph
        self.visual_dir = visual_dir
        # algo&hp selection prompts
        self.prompt = prompt
        self.hp_prompt = hp_prompt

    def background_info_prompts(self):
        data_columns = '\t'.join(self.data.columns)
        prompt = f"""
        In this report we want to explore the relationship among these variables: {data_columns}, and here are background information: {self.knowledge_docs}.
        please write a paragraph to describe the purpose of this report and background knowledge of these variables, 
        like meanings of variable names and potential relationships.
        """

        print("Start to find background information")
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        response_doc = response.choices[0].message.content
        return response_doc

    def data_prop_prompt(self):
        statics_dict = self.statics_dict
        # Data property prompt
        if statics_dict.get("Stationary") == "non time-series":
            missing = "has missing values," if statics_dict.get("Missingness") else "does not have missing values,"
            data_type = f"is {statics_dict.get('Data Type')} data,"
            linear = "satisfies the linearity assumption," if statics_dict.get(
                "Linearity") else "violates the linearity assumption,"
            gaussian = ",and satisfies the Gaussian error assumption" if statics_dict.get(
                "Gaussian Error") else ",and violates the Gaussian error assumption"
            data_prop_prompt = "This dataset " + missing + data_type + linear + gaussian
        else:
            data_prop_prompt = f"This dataset is {'stationary' if statics_dict.get('Stationary') else 'non-stationary'} time-series data"

        return data_prop_prompt

    def discover_process_prompts(self):
        prompt = f"""
        I want to describe the causal discovery procedure of this project. 
        Firstly, we preprocessed the data and checked statistical characteristics of this dataset.
        Then we let the LLM help us to select algorithms and hyper-parameters based on statistical characteristics of this dataset and background knowledge.
        This is the prompt for algorithm selection {self.prompt}, and this is the prompt for hyper-parameter selection {self.hp_prompt},
        you can find some useful information, i.e. the selected algorithms and parameters, in these prompts.
        Please discribe the procedure step by step clearly.
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
            background_info = self.background_info_prompts()
            # Graph effect info
            graph_prompt = self.graph_effect_prompts()
            discover_process = self.discover_process_prompts()
            # Report prompt
            prompt_template = self.load_context("postprocess/context/report_templete")
            # Graph paths
            graph_path1 = f'{self.visual_dir}/Initial_Graph.jpg'
            graph_path2 = f'{self.visual_dir}/Revised_Graph.jpg'

            replacements = {
                "[BACKGROUND_INFO]": background_info,
                "[DATA_PREVIEW]": data_preview,
                "[DATA_PROP]": data_prop,
                "[RESULT_ANALYSIS]": graph_prompt,
                "[DISCOVER_PROCESS]": discover_process,
                "[RESULT_GRAPH1]": graph_path1,
                "[RESULT_GRAPH2]": graph_path2
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
