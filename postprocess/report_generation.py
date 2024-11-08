from openai import OpenAI
import os
import re
import numpy as np 
from plumbum.cmd import latexmk
from plumbum import local
import networkx as nx
from postprocess.visualization import Visualization
#from postprocess.judge_functions import graph_effect_prompts
import PyPDF2
import ast

def compile_tex_to_pdf_with_refs(tex_file, output_dir=None, clean=True):
    """
    Silently compile a TeX file to PDF with multiple passes for references
    
    Args:
        tex_file (str): Path to the .tex file
        output_dir (str, optional): Output directory for the PDF
        clean (bool): Whether to clean auxiliary files after compilation
    
    Returns:
        bool: True if compilation successful, False otherwise
    """
    try:
        tex_dir = os.path.dirname(tex_file)
        if output_dir is None:
            output_dir = tex_dir

        # Multiple passes for references 
        try:
            # Build latexmk arguments
            args = [
                '-pdf',                     # Generate PDF output
                '-interaction=nonstopmode', # Don't stop for errors
                '-halt-on-error',           # Stop on errors
                '-f',
                '-bibtex',                   # Use bibtex for references
                f'-output-directory={output_dir}'
            ]
                
            # Add input file
            args.append(tex_file)
            
            # Run latexmk
            with local.env(TEXINPUTS=":./"):
                latexmk[args]()
        except Exception as e:
            print(f"Error in compilation pass")
            print(str(e))
            return False
        
        print(f"Successfully compiled {tex_file} to {output_dir}")
        return True
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return False

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
        self.visual_dir = global_state.user_data.output_graph_dir

    def get_title(self):
        if os.path.isdir(self.data_file):
            for file in os.listdir(self.data_file):
                if file.endswith(".csv"):
                    data_path = file
                    filename = os.path.splitext(os.path.basename(data_path))[0]
                    filename = filename.capitalize()
                    filename = filename.capitalize().replace("_", r"\_")
                    title = f'Causal Discovery Report on {filename}'
                    break
        elif self.data_file.endswith(".csv"):
            data_path = self.data_file
            filename = os.path.splitext(os.path.basename(data_path))[0]
            filename = filename.capitalize()
            filename = filename.capitalize().replace("_", r"\_")
            title = f'Causal Discovery Report on {filename}'
        else:
            title = 'Causal Discovery Report on Given Dataset'
        return title, filename
    
    def intro_prompt(self):
        prompt = f"""
        I want to conduct a causal discovery on a dataset and write a report. There are some background knowledge about this dataset.
        Please write a brief introduction paragraph. I only need the paragraph, don't include any title.
        Do not include any Greek Letters, Please change any Greek Letter into Math Mode, for example, you should change γ into $\gamma$
        
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
        response_intro = response_intro.replace("_", r"\_")
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

        col_names = '\t'.join(self.data.columns)
        prompt = f"""
I want to conduct a causal discovery on a dataset and write a report. There is some background knowledge about this dataset.
There are three sections:
### 1. Detailed Explanation about the Variables
### 2. Possible Causal Relations among These Variables
### 3. Other Background Domain Knowledge that may be Helpful for Experts

Please give me text in the second section ### 2. Possible Causal Relations among These Variables:
1. In this part, all relationships should be listed in this format: **A -> B**: explanation. 
2. Only one variable can appear at each side of ->; for example, **A -> B** is ok but **A -> B/C** is wrong, and **A -> B and C** is also wrong.
3. All variables should be among {col_names}, please delete contents that include any other variables!
4. Do not include any Greek Letters! Please change any Greek Letter into Math Mode, for example, you should change γ into $\gamma$

For example: 
- **'Raf' -> 'Mek'**: Raf activates Mek through phosphorylation, initiating the MAPK signaling cascade.

I only need the text; do not include the title.
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
        for match in matches:
            left_part = match[0]
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
                # deal with case like 'Length and Diameter'->'Viscera Weight' or 'Length\Diameter'->'Viscera Weight'
                if '/' in elements[i + 1] or 'and' in elements[i + 1]:
                    targets = [t for t in elements[i + 1].split('/|and')]
                    for target in targets:
                        relations.append((elements[i], target))
                else:
                    relations.append((elements[i], elements[i + 1]))
        # deal with case like ('Length, Diameter, Height', 'Viscera Weight')
        result = []
        for relation in relations:
            left = relation[0].split(',')
            right = relation[1].split(',')
            for l in left:
                for r in right:
                    result.append((l.strip(), r.strip()))

        zero_matrix = np.zeros((len(variables), len(variables)))
        for tuple in result:
            if tuple[0].lower() in variables.str.lower() and tuple[1].lower() in variables.str.lower():
                ind1 = variables.str.lower().get_loc(tuple[0].lower())
                ind2 = variables.str.lower().get_loc(tuple[1].lower())
                zero_matrix[ind2, ind1] = 1

        my_visual = Visualization(self.global_state)
        g = nx.from_numpy_array(zero_matrix, create_using=nx.DiGraph)
        # Relabel nodes with variable names from data columns
        mapping = {i: self.data.columns[i] for i in range(len(self.data.columns))}
        g = nx.relabel_nodes(g, mapping)
        pos = nx.spring_layout(g)
        _ = my_visual.plot_pdag(zero_matrix, 'potential_relation.pdf', pos=pos, relation=True)
        relation_path = f'{self.visual_dir}/potential_relation.pdf'

        def get_pdf_page_size(pdf_path):
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page = reader.pages[0]
                width = page.mediabox.width
                height = page.mediabox.height
                return height>width
            
        if sum(zero_matrix.flatten())!=0:
            figure_tall =  get_pdf_page_size(relation_path)
            if figure_tall:
                relation_prompt = f"""
                \begin{{minipage}}[t]{{0.7\linewidth}}
                {section2}
                \vfill
                \end{{minipage}}
                \hspace{{0.05\textwidth}}
                \begin{{minipage}}[t]{{0.3\linewidth}}
                    \begin{{figure}}[H]
                        \centering
                        \resizebox{{\linewidth}}{{!}}{{\includegraphics[height=0.3\textheight]{relation_path}}}
                        \caption{{\label{{fig:relation}}Possible Causal Relation Graph}}
                    \end{{figure}}
                \end{{minipage}}
                """
            else:
                relation_prompt = f"""
                {section2}

                \begin{{figure}}[H]
                \centering
                \includegraphics[width=0.5\linewidth]{relation_path}
                \caption{{\label{{fig:relation}}Possible Causal Relation Graph}}
                \end{{figure}}
                """
        else:
            relation_prompt = f"""
                {section2}
                """
        
        return section1, relation_prompt

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
                        \item \textbf{params[param]['full_name']}:
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
        for the chosen algorithm, which are specified below:
        {param_list}

        """

        if self.args.data_mode == 'real':
            repsonse += f"""
            \subsection{{Graph Tuning with Bootstrap and LLM Suggestion}}
            In the final step, we performed graph tuning with suggestions provided by the Bootstrap and LLM.
            
            Firstly, we use the Bootstrap technique to get how much confidence we have on each edge in the initial graph.
            If the confidence probability of a certain edge is greater than 95% and it is not in the initial graph, we force it.
            Otherwise, if the confidence probability is smaller than 5% and it exists in the initial graph, we change it to the edge type with the highest probability.
            
            After that, We utilize LLM to help us prune edges and determine the direction of undirected edges according to its knowledge repository.
            In this step LLM can use background knowledge to add some edges that are neglected by Statistical Methods.
            Voting techniques are used to enhance the robustness of results given by LLM, and the results given by LLM should not change results given by Bootstrap.

            By integrating insights from both of Bootsratp and LLM to refine the causal graph, we can achieve improvements in graph's accuracy and robustness.
            """
        return repsonse
    
    def graph_effect_prompts(self):
        variables = self.data.columns
        G = nx.from_numpy_array(self.graph.T, parallel_edges=False, create_using=nx.DiGraph)
        relations = [(variables[index[0]],variables[index[1]]) for index in G.edges]

        prompt = f"""
        This list of tuples reflects the causal relationship among variables {relations}.
        For example, if the tuple is (X1, X0), it means that {variables[1]} causes {variables[0]}, that is {variables[1]} -> {variables[0]}.
        1. Please write one paragraph to describe the causal relationship, do not include any lists, subtitles, etc.
        2. Don't mention tuples in the paragraph
        3. If variable names have meanings, please integrate background knowledge of these variables in the causal relationship analysis.
        Please use variable names {variables[0]}, {variables[1]}, ... in your description.
        4. Do not include any Greek Letters, Please change any Greek Letter into Math Mode, for example, you should change γ into $\gamma$
        
        For example:
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

    def graph_revise_prompts(self):
        repsonse = f"""
        By using the method mentioned in the Section 4.4, we provide a revise graph pruned with Bootstrap and LLM suggestion.
        Pruning results are as follows.
        """
        if self.global_state.results.bootstrap_errors != []:
                repsonse += f"""
                The following are force and forbidden results given by Bootstrap:
                
                {', '.join(self.global_state.results.bootstrap_errors)}
                """
        else:
            repsonse += f"""
            Bootstrap doesn't force or forbid any edges.
            """
        llm_evaluation_json = self.global_state.results.llm_errors
        force_record = llm_evaluation_json['force_record']
        forbid_record = llm_evaluation_json['forbid_record']
        if force_record != {}:
            repsonse += f"""
            The following are force results given by LLM:
            
            \begin{{itemize}}
            """
            for k in force_record.keys():
                tuple = ast.literal_eval(k)
                pair = f'{tuple[0]} \rightarrow {tuple[1]}'
                repsonse += f"""
                \item \textbf{pair}: {force_record[k]}
                """
            repsonse += f"""
            \end{{itemize}}
            """
        if  forbid_record != {}:
            repsonse += f"""
            The following relationships are forbidden by LLM:
            
            \begin{{itemize}}
            """
            for k in forbid_record.keys():
                tuple = ast.literal_eval(k)
                pair = f'({tuple[0]}, {tuple[1]})'
                repsonse += f"""
                \item \textbf{k}: {forbid_record[k]}
                """
            repsonse += f"""
            \end{{itemize}}
            """           
        if force_record=={} and forbid_record=={}:
            repsonse += f"""
            LLM doesn't force or forbid any edges.
            """
        llm_direction_reason = self.llm_direction_prompts()
        if llm_direction_reason is not None:
            repsonse += f"""
            The following are directions of remaining undirected edges determined by the LLM:
            {llm_direction_reason}
            """
        
        repsonse += """
        This structured approach ensures a comprehensive and methodical analysis of the causal relationships within the dataset.
        """
        return repsonse

    def llm_direction_prompts(self):
        import ast 
        reason_json = self.global_state.results.llm_directions
        if len(reason_json) == 0:
            return None
        prompts = '\begin{itemize}\n'
        for key in reason_json:
            tuple = ast.literal_eval(key)
            reason = reason_json[key]
            pair = f'{tuple[0]} \rightarrow {tuple[1]}'
            
            block = f"""
            \item \textbf{pair}: {reason}

            """
            prompts += block
    
        return prompts + '\end{itemize}'
    
    def confidence_graph_prompts(self):
        name_map = {'certain_edges': 'Directed Edge', #(->)
                    'uncertain_edges': 'Undirected Edge', #(-)
                    'bi_edges': 'Bi-Directed Edge', #(<->)
                    'half_edges': 'Non-Ancestor Edge', #(o->)
                    'non_edges': 'No D-Seperation Edge', #(o-o)
                    'non_existence':'No Edge'}
        graph_prompt = """
        \begin{figure}[H]
            \centering

        """
        bootstrap_dict = {k: v for k, v in self.bootstrap_probability.items() if v is not None and sum(v.flatten())>0}
        zero_graphs = [k for k, v in self.bootstrap_probability.items() if  v is not None and sum(v.flatten())==0]
        length = round(1/len(bootstrap_dict), 2)-0.01
        for key in bootstrap_dict.keys():
            graph_path = f'{self.visual_dir}/{key}_confidence_heatmap.jpg'
            caption = f'{name_map[key]}'
            graph_prompt += f"""
            \begin{{subfigure}}{{{length}\textwidth}}
                    \centering
                    \includegraphics[width=\linewidth]{graph_path}
                    \vfill
                    \caption{caption}
                \end{{subfigure}}"""
        
        graph_prompt += """
        \caption{Confidence Heatmap of Different Edges}
        \end{figure}        
        """
        text_map = {'certain_edges': 'directed edge ($->$)', #(->)
                    'uncertain_edges': 'undirected edge ($-$)', #(-)
                    'bi_edges': 'edge with hidden confounders ($<->$)', #(<->)
                    'half_edges': 'edge of non-ancestor ($o->$)', #(o->)
                    'non_edges': 'egde of no D-Seperation set', #(o-o)
                    'non_existence':'No Edge'}
        graph_text = "The above heatmaps show the confidence probability we have on different kinds of edges, including "
        for k in bootstrap_dict.keys():
            graph_text += f"{text_map[k]}, "
        graph_text += 'and probability of no edge.'
        for k_zero in zero_graphs:
            k_zero= k_zero.replace("_", "-")
            graph_text += f"The heatmap of {k_zero} is not shown because probabilities of all edges are 0. "
    
        graph_prompt += graph_text
        graph_prompt += """Based on the confidence probability heatmap and background knowledge, we can analyze the reliability of our graph."""
        
        return graph_prompt
        

    def confidence_analysis_prompts(self):
        relation_prob = self.graph_effect_prompts()

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
                     Do not include any Greek Letters, Please change any Greek Letter into Math Mode, for example, you should change γ into $\gamma$
                     0. Title: {self.title}
                     1. Introduction: {self.intro_info}
                     2. Discovery Procedure: {self.discover_process},
                     3. Graph Result Analysis: {self.graph_prompt},
                     4. Reliability Analysis: {self.reliability_prompt}
                     """}
                ]
            )

        response_doc = response.choices[0].message.content
        response_doc = response_doc.replace("_", r"\_")
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
        4. Do not include any Greek Letters, Please change any Greek Letter into Math Mode, for example, you should change γ into $\gamma$
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

    def generation(self, debug=False):
            '''
            generate and save the report
            :param debug: bool, if True, use template directly without LLM
            :return: Str: A technique report explaining all the results for readers
            '''
            if debug:
                # Load appropriate template based on data mode and ground truth
                prompt_template = self.load_context("postprocess/context/template_debug.tex")
                return prompt_template

            # Non-debug path continues with full report generation
            print("Start to generate the report")
            # Data info
            df = self.data.copy()
            if self.data.shape[1] > 20:  
                # choose 10 columns randomly
                random_columns = np.random.choice(df.columns, size=10, replace=False)
                df = df[random_columns]
            data_preview = df.head().to_latex(index=False)
            if len(self.data.columns) >= 9:
                data_preview = f"""
                \resizebox{{\textwidth}}{{!}}{{
                {data_preview}
                }}
                """
            #print('get data prop')
            data_prop_table = self.data_prop_prompt()
            # Intro info
            self.title, dataset = self.get_title()
            self.intro_info = self.intro_prompt()
            # Background info
            if self.data_mode == 'real':
                background_info1, relation_prompt = self.background_prompt()
                self.background_info1 = self.latex_convert(background_info1)
                self.background_info2 = self.latex_convert(relation_prompt)
            else:
                self.background_info1, self.background_info2 = '', ''
            # EDA info
            dist_info, corr_info = self.eda_prompt()
            dist_info = self.latex_convert(dist_info)
            corr_info = self.latex_convert(corr_info)
            # Procedure info
            self.discover_process = self.procedure_prompt()
            # Graph effect info
            self.graph_prompt = self.global_state.logging.graph_conversion['initial_graph_analysis']
            # Graph Revise info
            if self.data_mode == 'real':
                self.revise_process = self.graph_revise_prompts()
            else:
                self.revise_process = '' 
            # Graph Reliability info
            self.reliability_prompt = self.confidence_analysis_prompts()
            self.abstract = self.abstract_prompt()
            self.keywords = self.keyword_prompt()

            if self.data_mode == 'simulated':
                if self.global_state.user_data.ground_truth is not None:
                    prompt_template = self.load_context("postprocess/context/template_simulated.tex")
                else:
                    prompt_template = self.load_context("postprocess/context/template_simulated_notruth.tex")
            else:
                if self.global_state.user_data.ground_truth is not None:
                    prompt_template = self.load_context("postprocess/context/template_real.tex")
                else:
                    prompt_template = self.load_context("postprocess/context/template_real_notruth.tex")

            replacements = {
                "[TITLE]": self.title,
                "[DATASET]": dataset,
                "[ABSTRACT]": self.abstract,
                "[INTRO_INFO]": self.intro_info,
                "[BACKGROUND_INFO1]": self.background_info1,
                "[BACKGROUND_INFO2]": self.background_info2,
                "[POTENTIAL_GRAPH]": f'{self.visual_dir}/potential_relation.pdf',
                "[DATA_PREVIEW]": data_preview,
                "[DATA_PROP_TABLE]": data_prop_table,
                "[DIST_INFO]": dist_info,
                "[CORR_INFO]": corr_info,
                "[DIST_GRAPH]": self.eda_result['plot_path_dist'],
                "[CORR_GRAPH]": self.eda_result['plot_path_corr'],
                "[RESULT_ANALYSIS]": self.graph_prompt,
                "[ALGO]": self.algo,
                "[DISCOVER_PROCESS]": self.discover_process,
                "[REVISE_PROCESS]": self.revise_process,
                "[RELIABILITY_ANALYSIS]": self.reliability_prompt,
                "[RESULT_GRAPH0]": f'{self.visual_dir}/true_graph.pdf',
                "[RESULT_GRAPH1]": f'{self.visual_dir}/initial_graph.pdf',
                "[RESULT_GRAPH2]": f'{self.visual_dir}/revised_graph.pdf',
                "[RESULT_GRAPH3]": f'{self.visual_dir}/metrics.jpg',
                "[CONFIDENCE_GRAPH]": self.confidence_graph_prompts(),
                "[RESULT_METRICS1]": str(self.original_metrics),
                "[RESULT_METRICS2]": str(self.revised_metrics)
            }

            for placeholder, value in replacements.items():
                prompt_template = prompt_template.replace(placeholder, value)

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
                     7. Do not include any Greek Letters, Please change any Greek Letter into Math Mode, for example, you should change γ into $\gamma$
                     8. Do not change any parameters in figure settings
                     9. Only include your latex content in the response which can be rendered to pdf directly. Don't include other things like '''latex '''
                     """},
                    {"role": "user", "content": prompt_template}
                ]
            )

            output = response.choices[0].message.content
            return output
    
    def latex_bug_checking(self, tex_path, num_error_corrections=5):
        save_path = self.global_state.user_data.output_report_dir

        # Iteratively fix any LaTeX bugs
        for i in range(num_error_corrections):
            # Filter trivial bugs in chktex
            check_output = os.popen(f"lacheck {tex_path} -q -n2 -n24 -n13 -n1").read()
            with open(tex_path, 'r', encoding='utf-8') as file:
                tex_content = file.read()
            if check_output:
                response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": f"""
                     You are a helpful debugging assistant, help me to fix bugs in the latex report I will give you. 
                     1. Please fix the LaTeX errors guided by the output of `chktek`:
                        {check_output}.
                     2. Make the minimal fix required and do not remove or change any packages.
                     3. If some text is wrong you can delete the sentence
                     4. Only include your latex content in the response which can be rendered to pdf directly. Don't include other things like '''latex '''
                     """},
                    {"role": "user", "content": tex_content}
                ]
                )
                output = response.choices[0].message.content
                with open(f'{save_path}/report.tex', 'w', encoding='utf-8') as file:
                    file.write(output)
            else:
                break

    def save_report(self, report):
        save_path = self.global_state.user_data.output_report_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(f'{save_path}/report.txt', 'w', encoding='utf-8') as file:
            # Write some text to the file
            file.write(report)
        with open(f'{save_path}/report.tex', 'w', encoding='utf-8') as file:
            file.write(report)
        # fix latex bugs before rendering
        self.latex_bug_checking(f'{save_path}/report.tex')
        # Compile the .tex file to PDF using pdflatex
        compile_tex_to_pdf_with_refs(f'{save_path}/report.tex', save_path)
