import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm, CausalModel
from dowhy import gcm, CausalModel
import shap
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from pydantic import BaseModel
import os 

from causal_analysis.hte.hte_filter import HTE_Filter
from causal_analysis.hte.hte_params import HTE_Param_Selector
from causal_analysis.hte.hte_program import HTE_Programming

import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_setting.Initialize_state import global_state_initialization

def convert_adj_mat(mat):
    # In downstream analysis, we only keep direct edges and ignore all undirected edges
    mat = (mat == 1).astype(int)
    G = mat.T
    return G

class Analysis(object):
    def __init__(self, global_state, args):
        """
        Hardcoded to test the Auto MPG dataset and adjacency separately,
        while preserving references to global_state and args (which can be None).
        """
        self.global_state = global_state
        self.args = args
        self.data = global_state.user_data.processed_data
        #self.data = pd.read_csv('dataset/sachs/sachs.csv')
        #TODO: graph format
        self.graph = convert_adj_mat(global_state.results.revised_graph)
        #self.graph = convert_adj_mat(np.load('postprocess/test_result/sachs/cot_all_relation/3_voting/ind_revised_graph.npy'))
        #self.data = pd.read_csv('dataset/sachs/sachs.csv')
        self.graph = convert_adj_mat(global_state.results.revised_graph)
        #self.graph = convert_adj_mat(np.load('postprocess/test_result/sachs/cot_all_relation/3_voting/ind_revised_graph.npy'))
        self.G = nx.from_numpy_array(self.graph, create_using=nx.DiGraph) # convert adj matrix into DiGraph
        self.G = nx.relabel_nodes(self.G, {i: name for i, name in enumerate(self.data.columns)})
        print(self.G)
        self.dot_graph = self._generate_dowhy_graph()
        # Construct Causal Model via dowhy/gcm
        self.causal_model = gcm.InvertibleStructuralCausalModel(self.G)
        gcm.auto.assign_causal_mechanisms(self.causal_model, self.data)
        gcm.fit(self.causal_model, self.data)

    def _print_data_disclaimer(self):
        """
        Analyze the dataset's columns and print disclaimers about their nature (continuous/discrete/categorical).

        Previously, we had a hard-coded disclaimer. Now we dynamically check column types and counts.
        """
        dtypes = self.data.dtypes
        numeric_cols = dtypes[dtypes != 'object'].index.tolist()
        object_cols = dtypes[dtypes == 'object'].index.tolist()

        continuous_cols = []
        discrete_cols = []

        # Simple heuristic: if a numeric column has more than 10 unique values, consider it continuous; otherwise discrete
        for col in numeric_cols:
            unique_vals = self.data[col].nunique()
            if unique_vals > 10:
                continuous_cols.append(col)
            else:
                discrete_cols.append(col)

        print("\n" + "="*60)
        print("DATASET ANALYSIS:")
        if object_cols:
            print(f"Categorical (non-numeric) columns detected: {object_cols}")
        else:
            print("No categorical columns detected.")

        if continuous_cols:
            print(f"Continuous numeric columns detected: {continuous_cols}")
        else:
            print("No continuous numeric columns detected.")

        if discrete_cols:
            print(f"Discrete/low-cardinality numeric columns detected: {discrete_cols}")
        else:
            print("No discrete/low-cardinality numeric columns detected.")

        print("Please ensure your causal assumptions align with these column types.")
        print("="*60 + "\n")

    def feature_importance(self, target_node, visualize=True):
        print('start feature importance analysis')
        print('start feature importance analysis')
        # parent_relevance, noise_relevance = gcm.parent_relevance(self.causal_model, target_node=target_node)
        # parent_relevance, noise_relevance
        parent_nodes = list(self.G.predecessors(target_node))

        X = self.data.drop(columns=[target_node])
        y = self.data[[target_node]]
        X100 = shap.utils.sample(X, 100)  # background distribution for SHAP

        model_linear = sklearn.linear_model.LinearRegression()
        model_linear.fit(X, y)
        explainer_linear = shap.Explainer(model_linear.predict, X100)
        shap_values_linear = explainer_linear(X)
        
        # Mean absolute SHAP values
        shap_df = pd.DataFrame(np.abs(shap_values_linear.values), columns=X.columns)
        mean_shap_values = shap_df.mean()
        
        if visualize == True:
            figs = []
            figs = []
            # 1st SHAP Plot beeswarm
            ax = shap.plots.beeswarm(shap_values_linear, plot_size=(8,6), show=False)
            plt.savefig(f'{self.global_state.user_data.output_graph_dir}/shap_beeswarm_plot.png', bbox_inches='tight')  # Save as PNG
            #plt.savefig(f'shap_beeswarm_plot.png', bbox_inches='tight') 
            figs.append("shap_beeswarm_plot.png")
            #plt.savefig(f'shap_beeswarm_plot.png', bbox_inches='tight') 
            figs.append("shap_beeswarm_plot.png")
            # plt.show()

            # 2nd SHAP Plot Bar
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = shap.plots.bar(shap_values_linear, ax=ax, show=False)
            plt.savefig(f'{self.global_state.user_data.output_graph_dir}/shap_bar_plot.png', bbox_inches='tight')  # Save as PNG
            #plt.savefig(f'shap_bar_plot.png', bbox_inches='tight') 
            figs.append("shap_bar_plot.png")
            #plt.savefig(f'shap_bar_plot.png', bbox_inches='tight') 
            figs.append("shap_bar_plot.png")
            #plt.show()
            plt.close()
        return parent_nodes, mean_shap_values, figs
    
    def estimate_causal_effect(self, treatment, outcome, control_value=0, treatment_value=1):
        """
        Estimate the causal effect of a treatment on an outcome using DoWhy (backdoor.linear_regression).
        """
        print("\n" + "#"*60)
        print(f"Estimating Causal Effect of Treatment: {treatment} on Outcome: {outcome}")
        print("#"*60)

        print("\nCreating Causal Model...")
        model = CausalModel(
            data=self.data,
            treatment=treatment,
            outcome=outcome,
            graph=self.dot_graph()
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        print("\nIdentified Estimand:")
        print(identified_estimand)

        causal_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            control_value=control_value,
            treatment_value=treatment_value
        )

        print("\nCausal Estimate:")
        print(causal_estimate)
        # Call test_significance with the estimate value
        significance_results = causal_estimate.estimator.test_significance(self.data,causal_estimate.value)
        p_value = significance_results['p_value'][0]
        print("Significance Test Results:", p_value)

        print("\n=== Interpretation Hint ===")
        print("A negative causal estimate indicates that increasing the treatment variable (e.g., horsepower)")
        print("tends to decrease the outcome variable (e.g., mpg), assuming the model and assumptions hold.")
        print("============================\n")

        return causal_estimate, p_value

    def _generate_dowhy_graph(self):
        """
        Generate a causal graph in DOT format for DoWhy.
        """
        edges = nx.edges(self.G)
        dot_format = "digraph { "
        for u, v in edges:
            dot_format += f"{u} -> {v}; "
        dot_format += "}"
        return dot_format
    
    def attribute_anomalies(self, target_node, anomaly_samples, confidence_level=0.95):
        """
        Perform anomaly attribution and save the bar chart with confidence intervals to ./auto_mpg_output.
        """
        print("\n" + "#"*60)
        print(f"Performing Anomaly Attribution for target node: {target_node}")
        print("#"*60)

        # Call the attribute_anomalies function
        attribution_results = gcm.attribute_anomalies(
            causal_model=self.causal_model,  # Your fitted InvertibleStructuralCausalModel
            target_node=target_node,        # The target node for anomaly attribution
            anomaly_samples=anomaly_samples,  # DataFrame of anomalous samples
            anomaly_scorer=None,            # Use default anomaly scorer
            attribute_mean_deviation=False, # Attribute anomaly score (not mean deviation)
            num_distribution_samples=3000,  # Number of samples for marginal distribution
            shapley_config=None             # Use default Shapley config
        )

        # Convert results to a DataFrame
        rows = []
        for node, contributions in attribution_results.items():
            rows.append({
                "Node": node,
                "MeanAttributionScore": np.mean(contributions),
                "LowerCI": np.percentile(contributions, (1 - confidence_level) / 2 * 100),
                "UpperCI": np.percentile(contributions, (1 + confidence_level) / 2 * 100)
            })

        df = pd.DataFrame(rows).sort_values("MeanAttributionScore", ascending=False)

        # Extract info for plotting
        nodes = df["Node"]
        scores = df["MeanAttributionScore"]
        lower_bounds = df["LowerCI"]
        upper_bounds = df["UpperCI"]
        error = np.array([scores - lower_bounds, upper_bounds - scores])

        # Create figure
        plt.figure(figsize=(10, 6))  # Adjusted figure size for better readability
        plt.bar(nodes, scores, yerr=error, align='center', ecolor='black', capsize=5, color='skyblue')
        plt.xlabel("Nodes")
        plt.ylabel("Mean Attribution Score")
        plt.title(f"Anomaly Attribution for {target_node}")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()

        # Save figure
        fig_path = f'{self.global_state.user_data.output_graph_dir}/attribution_plot.png'
        plt.savefig(fig_path, bbox_inches='tight')
        figs = [fig_path]
        # plt.show()  # Show the plot in the notebook or script
        # plt.close()
        return df, figs

    def estimate_hte_effect(self, outcome, treatment, X_col, query):
        print("\nCreating Causal Model...")
        model = CausalModel(
            data=self.data,
            treatment=treatment,
            outcome=outcome,
            graph=self.dot_graph()
        )
        W_col = model.get_backdoor_variables()
        # Algorithm selection and deliberation
        filter = HTE_Filter(args)
        global_state = filter.forward(global_state, query)

        reranker = HTE_Param_Selector(args, y_col=outcome, T_col=treatment, X_col=X_col, W_col=W_col)
        global_state = reranker.forward(global_state)

        programmer = HTE_Programming(args, y_col=outcome, T_col=treatment, X_col=X_col, W_col=W_col)
        hte, hte_lower, hte_upper = programmer.forward(global_state, task='hte')

        plt.figure(figsize=(8, 6))
        sns.histplot(hte, bins=30, kde=True, color='skyblue', alpha=0.7)
        plt.axvline(hte.mean(), color='firebrick', linestyle='--', label='Mean HTE')
        plt.xlabel("Heterogeneous Treatment Effect (HTE)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Heterogeneous Treatment Effects")
        # Save figure
        fig_path = f'{self.global_state.user_data.output_graph_dir}/attribution_plot.png'
        plt.savefig(fig_path)
        figs = [fig_path]

        return hte, hte_lower, hte_upper, figs
    
    def sensityvity_analysis(self,):
        pass

    def call_LLM(self, format, prompt, message):
        client = OpenAI(organization=self.args.organization, project=self.args.project, api_key=self.args.apikey)
        if format:
            completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
            response_format=format,
            )
            parsed_response = completion.choices[0].message.parsed
        else: 
            completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
            )
            parsed_response = completion.choices[0].message.content
        return parsed_response

    def forward(self, task, desc, key_node):
        if task == 'Feature Importance':
            parent_nodes, mean_shap_values, figs = self.feature_importance(key_node, visualize=True)
            prompt = f"""
            I'm doing the feature importance analysis and please help me to write a brief analysis in bullet points.
            Here are some informations:
            **Result Variable we care about**: {key_node}
            **Parent Nodes of the Result Variable**: {parent_nodes}
            **Mean of Shapley Values**: {mean_shap_values}
            ""Description from User**: {desc}
            """
            response = self.call_LLM(response = self.call_LLM(args, None, 'You are an expert in Causal Discovery.', prompt))
            return response, figs 
        
        elif task == 'Treatment Effect Estimation':
            prompt = f"""
            I'm doing the Treatment Effect Estimation analysis, please identify the Treatment Variable in this description:
            {desc}
            The variable name must be among these variables: {self.data.columns}
            Only return me with the variable name, do not include anything else.
            """
            treatment = self.call_LLM(prompt)
            causal_estimate, p_value = self.estimate_causal_effect(treatment=treatment, outcome=key_node, control_value=0, treatment_value=1)
            parent_nodes = list(self.G.predecessors(key_node))
            prompt = f"""
            I'm doing the feature importance analysis and please help me to write a brief analysis in bullet points.
            Here are some informations:
            **Result Variable we care about**: {key_node}
            **Parent Nodes of the Result Variable**: {parent_nodes}
            **Causal Estimate Result**: {causal_estimate}
            **P-value of Significance Test for Causal Estimate**: {p_value}
            **Description from User**: {desc}
            """
            response = self.call_LLM(response = self.call_LLM(args, None, 'You are an expert in Causal Discovery.', prompt))
            return response, None
        
        elif task == 'Heterogeneous Treatment Effect Estimation':
            message = desc
            class VarList(BaseModel):
                treantment: str
                confounders: list[str]
            prompt = f"""You are a helpful assistant, please do the following tasks:
            Firstly, identify the Treatment Variable in user's query and save it in treantment as a string
            Secondly, identify a list of confounders and save it in confounders as a list of string
            The variable name must be among these variables: {self.data.columns}
            The outcome Y is {key_node}
            """
            parsed_response = self.call_LLM(args, VarList, prompt, message)
            treatment = parsed_response['treatment']
            confounders = parsed_response['confounders']
            self.estimate_hte_effect(outcome=key_node, treatment=treatment, X_col=confounders, query=desc)
            parent_nodes = list(self.G.predecessors(key_node))

            # prompt = f"""
            # I'm doing the feature importance analysis and please help me to write a brief analysis in bullet points.
            # Here are some informations:
            # **Result Variable we care about**: {key_node}
            # **Parent Nodes of the Result Variable**: {parent_nodes}
            # **Causal Estimate Result**: {causal_estimate}
            # **P-value of Significance Test for Causal Estimate**: {p_value}
            # **Description from User**: {desc}
            # """
            # response = self.call_LLM(prompt)
            return response, None
        
        elif task == 'Anormaly Attribution':
            df, figs = self.attribute_anomalies(target_node=key_node, anomaly_samples=self.data, confidence_level=0.95)
            parent_nodes = list(self.G.predecessors(key_node))
            prompt = f"""
            I'm doing the Anormaly Attribution analysis and please help me to write a brief analysis in bullet points.
            Here are some informations:
            **Abnormal Variable we care about**: {key_node}
            **Parent Nodes of the Abnormal Variable**: {parent_nodes}
            **Anormaly Attribution Result Table**: 
            {df.to_markdown()}
            **Description from User**: {desc}
            **Methods to calculate Anormaly Attribution Score**
            We estimated the contribution of the ancestors of {key_node}, including {key_node} itself, to the observed anomaly.
            In this method, we use invertible causal mechanisms to reconstruct and modify the noise leading to a certain observation. We then ask, “If the noise value of a specific node was from its ‘normal’ distribution, would we still have observed an anomalous value in the target node?”. The change in the severity of the anomaly in the target node after altering an upstream noise variable’s value, based on its learned distribution, indicates the node’s contribution to the anomaly. The advantage of using the noise value over the actual node value is that we measure only the influence originating from the node and not inherited from its parents.
            """
            response = self.call_LLM(args, None, 'You are an expert in Causal Discovery.', prompt)
            return response, figs


    

##############
def LLM_parse_query(args, format, prompt, message):
    client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
    if format:
        completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        response_format=format,
        )
        parsed_response = completion.choices[0].message.parsed
    else: 
        completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        )
        parsed_response = completion.choices[0].message.content
    return parsed_response

def main(global_state, args):
    """
    Modify the main function to call attribute_anomalies and save the results in ./auto_mpg_output.
    """
    print("Welcome to the Causal Analysis Demo using the Auto MPG dataset.\n")
    
    analysis = Analysis(global_state, args)
    message = "The value of PIP3 is abnormal, help me to find which variables cause this anomaly"
    class InfList(BaseModel):
        indicator: bool
        tasks: list[str]
        descriptions: list[str]
        key_node: list[str]
    prompt = """You are a helpful assistant, please do the following tasks:
            Firstly identify whether user want to conduct causal tasks and save the boolean result in indicator. 
            Secondly if user want to conduct causal tasks, please identify what tasks the user want to do and save them as a list in tasks.
            Please choose among the following causal tasks, if there's no matched task just return an empty list 
            Tasks you can choose: 1. Treatment Effect Estimation; 2. Anormaly Attribution; 3. Feature Importance 4. Heterogeneous Treatment Effect Estimation
            thirdly, save user's description for their tasks as a list in descriptions, the length of description list must be the same with task list
            Fourthly, save the key result variable user care about as a list, each task must have a key result variable and they can be the same, the length of result variable list must be the same with task list
            """
    global_state.logging.downstream_discuss.append({"role": "user", "content": message})
    parsed_response = LLM_parse_query(args, InfList, prompt, message)
    indicator, tasks_list, descs_list, key_node_list = parsed_response.indicator, parsed_response.tasks, parsed_response.descriptions, parsed_response.key_node
    print(indicator, tasks_list, descs_list, key_node_list)
    #tasks_list, descs_list, key_node_list = ['Treatment Effect Estimation'], ['Analyze the treatment effect of PIP2 to PIP3.'], ['PIP3']
    for i, (task, desc, key_node) in enumerate(zip(tasks_list, descs_list, key_node_list)):
        print(task, desc, key_node)
        response, figs = analysis.forward(task, desc, key_node)
        print(response, figs)

if __name__ == '__main__':
    import argparse
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

        args = parser.parse_args()
        return args
    args = parse_args()
    global_state = global_state_initialization(args)
    global_state.user_data.raw_data = pd.read_csv(args.data_file)
    global_state.user_data.processed_data = global_state.user_data.raw_data

    main(global_state, args)