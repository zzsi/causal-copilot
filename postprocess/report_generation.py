from openai import OpenAI
import os
import re
import numpy as np 
import shutil
import argparse
import networkx as nx
from postprocess.visualization import Visualization
import subprocess

from postprocess.judge_functions import graph_effect_prompts

class Report_generation(object):
    def __init__(self, global_state, args):
        """
        :param global_state: a dict containing global variables and information
        :param args: arguments for the report generation
        """

        self.client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
        self.data_mode = args.data_mode
        self.data_file = args.data_file
        self.global_state = global_state
        self.args = args 
        self.statistics_desc = global_state.statistics.description
        self.knowledge_docs = global_state.user_data.knowledge_docs[0]
        # Data info
        self.data = global_state.user_data.raw_data
        self.statistics = global_state.statistics
        # EDA info
        self.eda_result = global_state.results.eda
        # Result graph matrix
        self.graph = global_state.results.converted_graph
        self.bootstrap_probability = global_state.results.bootstrap_probability
        self.original_metrics = global_state.results.metrics
        self.revised_metrics = global_state.results.revised_metrics
        # algo&hp selection prompts
        self.algo = global_state.algorithm.selected_algorithm
        self.algo_can = global_state.algorithm.algorithm_candidates
        self.algo_param = global_state.algorithm.algorithm_arguments_json
        self.prompt = global_state.logging.select_conversation[0]['response']
        self.hp_prompt = global_state.logging.argument_conversation[0]['response']
        # Path to find the visualization graph
        self.visual_dir = args.output_graph_dir

    def get_title(self):
        for file in os.listdir(self.data_file):
            if file.endswith(".csv"):
                data_path = file
                filename = os.path.splitext(os.path.basename(data_path))[0]
                filename = filename.capitalize()
                title = f'Causal Discovery Report on {filename}'
                break
            else:
                title = 'Causal Discovery Report on Given Dataset'
        return title, filename
    
    def intro_prompt(self):
        prompt = f"""
        I want to conduct a causal discovery on a dataset and write a report. There are some background knowledge about this dataset.
        Please write a brief introduction paragraph. I only need the paragraph, don't include any title.
        
        Background about this dataset: {self.knowledge_docs}
        """
    
        print("Start to find Introduction")
        response_dist = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        response_intro = response_dist.choices[0].message.content
        return response_intro
    
    def background_prompt(self):
        prompt = f"""
        I want to conduct a causal discovery on a dataset and write a report. There are some background knowledge about this dataset.
        There are three sections:
        ### 1. Detailed Explanation about the Variables
        ### 2. Possible Causal Relations among These Variables
        ### 3. Other Background Domain Knowledge that may be Helpful for Experts
        Please give me text in the first section ### 1. Detailed Explanation about the Variables
        I only need the text, do not include title
        Background about this dataset: {self.knowledge_docs}
        """
    
        print("Start to find Background")
        response_background = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        section1 = response_background.choices[0].message.content

        prompt = f"""
        I want to conduct a causal discovery on a dataset and write a report. There are some background knowledge about this dataset.
        There are three sections:
        ### 1. Detailed Explanation about the Variables
        ### 2. Possible Causal Relations among These Variables
        ### 3. Other Background Domain Knowledge that may be Helpful for Experts
        Please give me text in the second section ### 2. Possible Causal Relations among These Variables
        In this part, all relationships should be listed in this format: **A -> B**: explanation. 
        For example: 
        - **'Raf' -> 'Mek'**: Raf activates Mek through phosphorylation, initiating the MAPK signaling cascade.
        I only need the text, do not include title
        Background about this dataset: {self.knowledge_docs}
        """

        response_background = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        section2 = response_background.choices[0].message.content
        variables = self.data.columns
        pattern = r'\*\*(.*?)\s*(→|↔|->|<->)\s*(.*?)\*\*'
        relations = []
        # Find all matches
        matches = re.findall(pattern, section2)
        print(matches)
        for match in matches:
            print('match: ', match)
            left_part = match[0]
            #arrow = match[1]
            right_part = match[2]

            elements = [left_part]
            split_elements = re.split(r'\s*(→|↔|->|<->)\s*', right_part)
            # Iterate over the split elements
            for i in range(0, len(split_elements), 2):
                element = split_elements[i]
                if element:  # Avoid empty strings
                    elements.append(element)
            # Create pairs for the elements
            for i in range(len(elements) - 1):
                if '/' in elements[i + 1]:
                    targets = [t for t in elements[i + 1].split('/')]
                    for target in targets:
                        relations.append((elements[i], target))
                else:
                    relations.append((elements[i], elements[i + 1]))

        zero_matrix = np.zeros((len(variables), len(variables)))
        for tuple in relations:
            if tuple[0] in variables and tuple[1] in variables:
                ind1 = variables.get_loc(tuple[0])
                ind2 = variables.get_loc(tuple[1])
                zero_matrix[ind2, ind1] = 1
        my_visual = Visualization(self.global_state, self.args)
        pos_potential = my_visual.plot_pdag(zero_matrix, 'potential_relation.png')
        return section1, section2, zero_matrix

    def data_prop_prompt(self):
        n, m = self.data.shape
        shape = f'({n}, {m})'
        missingness = 'True' if self.statistics.missingness else 'False'
        data_type = self.statistics.data_type
        linearity = 'True' if self.statistics.linearity else 'False'
        gaussian_error = 'True' if self.statistics.gaussian_error else 'False'
        stationary = 'True' if self.statistics.data_type == 'Time-series' else 'False'
        heterogeneous = 'True' if self.statistics.heterogeneous else 'False'
        prop_table = f"""
    \begin{{tabular}}{{rrrrrrr}}
        \toprule
        Shape ($n$ x $d$) & Data Type & Missing Value & Linearity & Gaussian Errors & Time-Series & Heterogeneity \\
        \midrule
        {shape}   & {data_type} & {missingness} & {linearity} & {gaussian_error} & {stationary} & {heterogeneous} \\
        \bottomrule
    \end{{tabular}}
        """
        return prop_table
        
    def eda_prompt(self):
        dist_input = self.eda_result['dist_analysis']
        corr_input = self.eda_result['corr_analysis']
        prompt_dist = f"""
            Given the following statistics about features in a dataset:\n\n
            {dist_input}\n
            1. Please categorize variables according to their distribution features, do not list out all Distribution values.
            2. Please list variables in one category in one line like:
                \item Slight left skew distributed variables: Length, Shell Weight, Diameter, Whole Weight
            3. If a category has no variable, please fill with None, like:
                \item Symmetric distributed variables: None
            4. Only give me a latex format item list.
            5. Please follow this templete, don't add any other things like subtitle, analysis, etc.
            
            Templete:
            '\begin{{itemize}}
            \item Slight left skew distributed variables: Length, Shell Weight, Diameter, Whole Weight
            \item Slight right skew distributed variables: Whole Weight, Age
            \item Symmetric distributed variables: Color
            \end{{itemize}}'
            
            """

        prompt_corr = f"""
            Given the following correlation statistics about features in a dataset:\n\n
            {corr_input}\n
            1. Please categorize variable pairs according to their correlation, do not list out all correlation values.
            2. Please list variable pairs in one category in one line like:
                \item Slight left skew distributed variables: Shell weight and Length, Age and Height
            3. If a category has no variable, please fill with None, like:
                \item Symmetric distributed variables: None
            4. Please follow this templete, don't add any other things like subtitle, analysis, etc.

            Templete:
            In this analysis, we will categorize the correlation statistics of features in the dataset into three distinct categories: Strong correlations ($r>0.8$), Moderate correlations ($0.5<r<0.8$), and Weak correlations ($r<0.5>$).
            
            \begin{{itemize}}
            \item Strong Correlated Variables: Shell weight and Length, Age and Height
            \item Moderate Correlated Variables: Length and Age
            \item Weak Correlated Variables: Age and Weight, Weight and Sex
            \end{{itemize}}         
            """


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

    def algo_selection_prompt(self):
        algo_candidates = self.algo_can
        response = """
        \begin{itemize}

        """
 
        for algo in algo_candidates:
            sub_block = f"""
                        \item \textbf{algo}:
                        \begin{{itemize}}
                            \item \textbf{{Description}}: {algo_candidates[algo]['description']}
                            \item \textbf{{Justification}}: {algo_candidates[algo]['justification']}
                        \end{{itemize}}

                         """
            response += sub_block

        response += """
                    \end{itemize}
                    """
        return response
    
    def param_selection_prompt(self):
        params = self.algo_param['hyperparameters']
        response = """
        \begin{itemize}

        """
 
        for param in params:
            sub_block = f"""
                        \item \textbf{param}:
                        \begin{{itemize}}
                            \item \textbf{{Value}}: {params[param]['value']}
                            \item \textbf{{Explanation}}: {params[param]['explanation']}
                        \end{{itemize}}

                         """
            response += sub_block

        response += """
                    \end{itemize}
                    """
        return response

    def procedure_prompt(self):
        algo_list = self.algo_selection_prompt()
        param_list = self.param_selection_prompt()
        llm_direction_reason = self.llm_direction_prompts()

        repsonse = f"""
        In this section, we provide a detailed description of the causal discovery process implemented by Causal Copilot. 
        We also provide the chosen algorithms and hyperparameters, along with the justifications for these selections.

        \subsection{{Data Preprocessing}}
        In this initial step, we preprocessed the data and examined its statistical characteristics. 
        This involved cleaning the data, handling missing values, and performing exploratory data analysis to understand distributions and relationships between variables.
                
        \subsection{{Algorithm Selection assisted with LLM}}
        Following data preprocessing, we employed a large language model (LLM) to assist in 
        selecting appropriate algorithms for causal discovery based on the statistical characteristics of the dataset and relevant background knowledge. 
        The top three chosen algorithms, listed in order of suitability, are as follows:   
        {algo_list}

        \subsection{{Hyperparameter Values Proposal assisted with LLM}}
        Once the algorithms were selected, the LLM aided in proposing hyperparameters 
        for the [ALGO] algorithm, which are specified below:
        {param_list}

        \subsection{{Graph Tuning with LLM Suggestion}}
        In the final step, we performed graph tuning with suggestions provided by the LLM.
        We utilize LLM to help us determine the direction of undirected edges according to its knowledge repository.
        By integrating insights from the LLM to refine the causal graph, we can achieve improvements in graph's accuracy and robustness.
        {llm_direction_reason}

        This structured approach ensures a comprehensive and methodical analysis of the causal relationships within the dataset.
        """
        return repsonse
    
    def graph_effect_prompts(self):
        variables = self.data.columns
        G = nx.from_numpy_array(self.graph.T, parallel_edges=False, create_using=nx.DiGraph)
        relations = [(variables[index[0]],variables[index[1]]) for index in G.edges]

        prompt = f"""
        This list of tuples reflects the causal relationship among variables {relations}.
        For example, if the tuple is (X1, X0), it means that {variables[1]} causes {variables[0]}, that is {variables[1]} -> {variables[0]}.
        Please write a paragraph to describe the causal relationship, and you can add some analysis.
        If you want to list something or add subtitles, make sure they are in latex format. 
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

    def llm_direction_prompts(self):
        import ast 
        reason_json = self.global_state.results.llm_directions
        prompts = ''
        for key in reason_json:
            tuple = ast.literal_eval(key)
            direction = reason_json[key]['direction']
            reason = reason_json[key]['justification']
            if direction == 'right':
                pair = f'{tuple[0]} \rightarrow {tuple[1]}'
            elif direction == 'left':
                pair = f'{tuple[1]} \rightarrow {tuple[0]}'
            else:
                continue
            block = f"""
            \item \textbf{pair}: {reason}

            """
            prompts += block
        return prompts
    
    def confidence_analysis_prompts(self):

        relation_prob = graph_effect_prompts(self.data,
                                             self.graph,
                                             self.bootstrap_probability)

        variables = '\t'.join(self.data.columns)
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

    def abstract_prompt(self):
        response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": 
                     f"""
                     You are an expert in the causal discovery field and helpful assistant.                    
                     """},
                    {"role": "user", "content": 
                     f"""
                     Help me to write a short abstract according to the given information. 
                     You should cover what data is analyzed (find it in title and introduction), what methodology we used, what result we got, and what is our contribution.
                     Only include your abstract text content in the response. Don't include any other things like title, etc.
                     0. Title: {self.title}
                     1. Introduction: {self.intro_info}
                     2. Discovery Procedure: {self.discover_process},
                     3. Graph Result Analysis: {self.graph_prompt},
                     4. Reliability Analysis: {self.reliability_prompt}
                     """}
                ]
            )

        response_doc = response.choices[0].message.content
        return response_doc
    
    def keyword_prompt(self):
        response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": 
                     """
                     You are an expert in the causal discovery field and helpful assistant.                    
                     """},
                    {"role": "user", "content": 
                     f"""
                     Give me some keywords according to the given information, and these keywords are for an academic report.
                    You should seperate each keywords with a comma, like 'keyword1, keyword2, keyword3'.
                     Only include your keywords text in the response. Don't include any other things.
                     Only include 5 most important key words.
                     0. Title: {self.title}
                     1. Abstract: {self.abstract}
                     1. Introduction: {self.intro_info}
                     """}
                ]
            )

        response_doc = response.choices[0].message.content
        return response_doc
    
    def load_context(self, filepath):
        with open(filepath, "r") as f:
            return f.read()

    def latex_convert(self, text):
        prompt = f"""
        Please convert this markdown format text into latex format.
        For example, 
        1. for subheadings '## heading', convert it into '\subsection{{heading}}; for subsubheadings '### heading', convert it into '\subsubsection{{heading}}'
        2. for list of items '-item1 -item2 -item3', convert them into 
        '\begin{{itemize}}
        \item item1
        \item item2
        \item item3
        \end{{itemize}}'
        3. For bolding notation '**text**', convert it into \\textbf{{text}}.
        This is the text you need to convert: {text}
        Only response converted text, please do not include other things like 'Here is the converted text in LaTeX format:'
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        response_doc = response.choices[0].message.content
        return response_doc

    def generation(self):
            '''
            generate and save the report
            :return: Str: A technique report explaining all the results for readers
            '''
            # Data info
            data_preview = self.data.head().to_latex(index=False)
            data_prop_table = self.data_prop_prompt()
            # Intro info
            self.title, dataset = self.get_title()
            self.intro_info = self.intro_prompt()
            # Background info
            background_info1, background_info2, relation_mat = self.background_prompt()
            self.background_info1 = self.latex_convert(background_info1)
            self.background_info2 = self.latex_convert(background_info2)
            # EDA info
            dist_info, corr_info = self.eda_prompt()
            dist_info = self.latex_convert(dist_info)
            corr_info = self.latex_convert(corr_info)
            # ALGO info
            #algo_list = self.algo_selection_prompt()
            #param_list = self.param_selection_prompt()
            self.discover_process = self.procedure_prompt()
            # Graph effect info
            self.graph_prompt = self.latex_convert(self.graph_effect_prompts())
            # Graph Reliability info
            self.reliability_prompt = self.confidence_analysis_prompts()
            self.abstract = self.abstract_prompt()
            self.keywords = self.keyword_prompt()
            # Graph paths
            graph_path0 = f'{self.visual_dir}/true_graph.png'
            graph_path1 = f'{self.visual_dir}/initial_graph.png'
            graph_path2 = f'{self.visual_dir}/revised_graph.png'
            graph_path3 = f'{self.visual_dir}/metrics.jpg'
            graph_path4 = f'{self.visual_dir}/confidence_heatmap.jpg'
            graph_relation_path = f'{self.visual_dir}/potential_relation.png'
            # EDA Graph paths
            dist_graph_path = self.eda_result['plot_path_dist']
            scat_graph_path = self.eda_result['plot_path_scat']
            corr_graph_path = self.eda_result['plot_path_corr']

            if self.data_mode == 'simulated':
                # Report prompt
                prompt_template = self.load_context("postprocess/context/template.tex")
                replacements = {
                    "[TITLE]": self.title,
                    "[DATASET]": dataset,
                    "[ABSTRACT]": self.abstract,
                    "[INTRO_INFO]": self.intro_info,
                    "[BACKGROUND_INFO1]": self.background_info1,
                    "[BACKGROUND_INFO2]": self.background_info2,
                    "[POTENTIAL_GRAPH]": graph_relation_path,
                    "[DATA_PREVIEW]": data_preview,
                    "[DATA_PROP_TABLE]": data_prop_table,
                    "[DIST_INFO]": dist_info,
                    "[CORR_INFO]": corr_info,
                    "[DIST_GRAPH]": dist_graph_path,
                    "[CORR_GRAPH]": corr_graph_path,
                    "[RESULT_ANALYSIS]": self.graph_prompt,
                    "[ALGO]": self.algo,
                    #"[ALGO_LIST]": algo_list,
                    #"[PARAM_LIST]": param_list,
                    "[DISCOVER_PROCESS]": self.discover_process,
                    "[RELIABILITY_ANALYSIS]": self.reliability_prompt,
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
                prompt_template = self.load_context("postprocess/context/template.tex")
                replacements = {
                    "[TITLE]": self.title,
                    "[DATASET]": dataset,
                    "[ABSTRACT]": self.abstract,
                    "[INTRO_INFO]": self.intro_info,
                    "[BACKGROUND_INFO1]": self.background_info1,
                    "[BACKGROUND_INFO2]": self.background_info2,
                    "[POTENTIAL_GRAPH]": graph_relation_path,
                    "[DATA_PREVIEW]": data_preview,
                    "[DATA_PROP_TABLE]": data_prop_table,
                    "[DIST_INFO]": dist_info,
                    "[CORR_INFO]": corr_info,
                    "[DIST_GRAPH]": dist_graph_path,
                    "[CORR_GRAPH]": corr_graph_path,
                    "[RESULT_ANALYSIS]": self.graph_prompt,
                    "[ALGO]": self.algo,
                    #"[ALGO_LIST]": algo_list,
                    #"[PARAM_LIST]": param_list,
                    "[DISCOVER_PROCESS]": self.discover_process,
                    "[RELIABILITY_ANALYSIS]": self.reliability_prompt,
                    "[RESULT_GRAPH1]": graph_path1,
                    "[RESULT_GRAPH2]": graph_path2,
                    "[RESULT_GRAPH4]": graph_path4
                }

            for placeholder, value in replacements.items():
                prompt_template = prompt_template.replace(placeholder, value)

            #return prompt_template

            print("Start to generate the report")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": """
                     You are a causal discovery expert, help me to revise this latex report templete. 
                     1. Provide your response in Latex format, and only include Latex content which can be rendered to pdf directly. Don't include other things
                     2. Pay attention to orders of section, subsection, subsubsection... based on the meaning of headings
                     3. If there is number in \subsection{{1. headings}}, please remove the number and only reserve the heading text \subsection{heading}
                     4. If there is step number in \subsection{{Step 1 headings}}, please remove the step number and only reserve the heading text \subsection{heading}
                     5. Replace ↔ with $\leftrightarrow$, -> with $\\rightarrow$
                     6. All **text** should be replaced with \\textbf{text}
                     7. Only include your latex content in the response which can be rendered to pdf directly. Don't include other things like '''latex '''
                     """},
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
        with open(f'{save_path}/report.tex', 'w') as file:
            file.write(report)

        # Compile the .tex file to PDF using pdflatex
        try:
            subprocess.run(['xelatex', '-output-directory', save_path, f'{save_path}/report.tex'], check=False)
            print("PDF generated successfully.")
        except subprocess.CalledProcessError:
            print("An error occurred while generating the PDF.")

