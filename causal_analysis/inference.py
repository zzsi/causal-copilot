import numpy as np
import pandas as pd
import networkx as nx

# REMOVE or comment out the direct import of CausalModel from DoWhy:
# from dowhy import gcm, CausalModel

# Keep gcm if you still want anomaly attribution and interventions:
from dowhy import gcm  
# Import econml classes instead of using DoWhy's linear_regression or DML
from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML

import shap
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from pydantic import BaseModel
import os
import sys
import warnings
from CEM_LinearInf.cem import cem
from CEM_LinearInf.balance import balance
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_analysis.DML.hte_filter import HTE_Filter as DML_HTE_Filter
from causal_analysis.DML.hte_params import HTE_Param_Selector as DML_HTE_Param_Selector
from causal_analysis.DML.hte_program import HTE_Programming as DML_HTE_Programming
from causal_analysis.DRL.hte_filter import HTE_Filter as DRL_HTE_Filter
from causal_analysis.DRL.hte_params import HTE_Param_Selector as DRL_HTE_Param_Selector
from causal_analysis.DRL.hte_program import HTE_Programming as DRL_HTE_Programming
from causal_analysis.DRL.hte_filter import HTE_Filter as IV_HTE_Filter
from causal_analysis.DRL.hte_params import HTE_Param_Selector as IV_HTE_Param_Selector
from causal_analysis.DRL.hte_program import HTE_Programming as IV_HTE_Programming
from causal_analysis.help_functions import *
from causal_analysis.analysis import *
from global_setting.Initialize_state import global_state_initialization

class Analysis(object):
    def __init__(self, global_state, args):
        """
        Hardcoded to test the Auto MPG dataset and adjacency separately,
        while preserving references to global_state and args (which can be None).
        """
        self.global_state = global_state
        self.args = args
        self.data = global_state.user_data.processed_data
        #TODO: graph format
        self.graph = convert_adj_mat(global_state.results.converted_graph)
        self.G = nx.from_numpy_array(self.graph, create_using=nx.DiGraph) # convert adj matrix into DiGraph
        self.G = nx.relabel_nodes(self.G, {i: name for i, name in enumerate(self.data.columns)})
        self.dot_graph = self._generate_dowhy_graph()
        # Construct Causal Model via dowhy/gcm
        self.causal_model = gcm.InvertibleStructuralCausalModel(self.G)
        # gcm.auto.assign_causal_mechanisms(self.causal_model, self.data)
        # gcm.fit(self.causal_model, self.data)

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
        shap_df.to_csv(f'{self.global_state.user_data.output_graph_dir}/shap_df.csv', index=False)
        
        figs = []
        if visualize == True:
            # 1st SHAP Plot beeswarm
            ax = shap.plots.beeswarm(shap_values_linear, plot_size=(8,6), show=False)
            plt.savefig(f'{self.global_state.user_data.output_graph_dir}/shap_beeswarm_plot.png', bbox_inches='tight')  # Save as PNG
            figs.append(f'{self.global_state.user_data.output_graph_dir}/shap_beeswarm_plot.png')

            # 2nd SHAP Plot Bar
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = shap.plots.bar(shap_values_linear, show=False)
            plt.savefig(f'{self.global_state.user_data.output_graph_dir}/shap_bar_plot.png', bbox_inches='tight')  # Save as PNG
            figs.append(f'{self.global_state.user_data.output_graph_dir}/shap_bar_plot.png')
            plt.close()
        return parent_nodes, mean_shap_values, figs

    def _propensity_score_matching(self, confounders, treatment, outcome):
        """
        Perform Propensity Score Matching (PSM) using sklearn.
        :param treatment: The treatment variable.
        :param outcome: The outcome variable.
        :return: Matched data after PSM.
        """
        # Debug: Print treatment variable
        print(f"Treatment variable ({treatment}):\n{self.data[treatment].head()}")
        print(f"Unique values in treatment variable: {self.data[treatment].unique()}")

        # Ensure the treatment variable is binary
        if len(self.data[treatment].unique()) > 2:
            # Binarize the treatment variable using a threshold (e.g., median)
            threshold = self.data[treatment].median()
            self.data[treatment] = (self.data[treatment] > threshold).astype(int)
            print(f"Treatment variable binarized using threshold {threshold}.")

        # Debug: Print binarized treatment variable
        print(f"Binarized treatment variable:\n{self.data[treatment].head()}")

        # Ensure the treatment variable is a 1-dimensional array
        treatment_values = self.data[treatment].values
        print(f"Shape of treatment values: {treatment_values.shape}")
        if treatment_values.ndim != 1:
            raise ValueError(f"Treatment variable must be 1-dimensional. Got shape {treatment_values.shape}.")

        # Step 1: Calculate propensity scores using LogisticRegression
        #confounders = self.data[confounders].drop(columns=[treatment, outcome])  # Features (confounders)
        confounders = self.data[confounders]

        # Debug: Print confounders
        print(f"Confounders:\n{confounders.head()}")
        print(f"Shape of confounders: {confounders.shape}")

        # Ensure confounders are not empty
        if confounders.empty:
            raise ValueError("No confounders found. Ensure the dataset contains valid confounders.")

        # Fit logistic regression model
        propensity_model = LogisticRegression(solver='liblinear', max_iter=1000)
        propensity_model.fit(confounders, treatment_values)
        
        # Add propensity scores to the data
        self.data['propensity_score'] = propensity_model.predict_proba(confounders)[:, 1]

        # Step 2: Separate treated and control groups
        treated = self.data[self.data[treatment] == 1]
        control = self.data[self.data[treatment] == 0]

        # Step 3: Perform nearest neighbor matching using sklearn
        nbrs = NearestNeighbors(n_neighbors=1).fit(control[['propensity_score']])
        distances, indices = nbrs.kneighbors(treated[['propensity_score']])

        # Step 4: Create matched data
        matched_control = control.iloc[indices.flatten()]
        matched_data = pd.concat([treated, matched_control])

        return matched_data
                
    def _coarsened_exact_matching(self,confounders,cont_confounder, treatment, outcome):
        """
        Perform Coarsened Exact Matching (CEM).
        """
        # Initialize CEM
        my_cem = cem(self.data, confounder_cols = confounders, cont_confounder_cols = cont_confounder,  col_t = treatment, col_y = outcome)
        matched_data = my_cem.match()
        return matched_data

    def _identify_confounders(self, treatment, outcome):
        """
        Identify confounders for the treatment and outcome using the causal graph.
        If the graph is not available, use the LLM to suggest confounders.
        """
        # Validate that treatment and outcome are valid nodes in the graph
        if treatment not in self.G.nodes or outcome not in self.G.nodes:
            raise ValueError(f"Invalid treatment or outcome variable in the graph: {treatment}, {outcome}")

        # Use the causal graph to identify confounders
        confounders = list(set(self.G.predecessors(treatment)) & set(self.G.predecessors(outcome)))
        return confounders 
    
    def _check_balance(self, data, matched_data, treatment, outcome, confounders, cont_confounders, title):
        """
        Check balance of confounders between treated and control groups using density, KS scatter, and ECDF plots.
        """
        # Ensure the confounders and continuous confounders are valid columns
        valid_confounders = [col for col in confounders if col in matched_data.columns]
        valid_cont_confounders = [col for col in cont_confounders if col in matched_data.columns]

        if not valid_confounders:
            raise ValueError("No valid confounders found in the matched data.")
        if not valid_cont_confounders:
            print("Warning: No valid continuous confounders found in the matched data.")

        # Generate density plot
        self._generate_density_plot(data, matched_data, treatment, valid_confounders, title)

    def _generate_density_plot(self, data, matched_data, treatment, confounders, title):
        """
        Generate a single density plot with subplots for different confounders.
        Each row corresponds to a confounder, with treated and control groups in separate subplots.
        """
        sns.set_style("darkgrid")  
        num_confounders = len(confounders)
        fig, axes = plt.subplots(nrows=num_confounders, ncols=2, figsize=(20, 6 * num_confounders))
        
        if num_confounders == 1:
            axes = [axes]  

        for i, confounder in enumerate(confounders):
            # Treated group (left subplot)
            ax_treated = axes[i][0]
        # Generate density plot
        self._generate_density_plot(data, matched_data, treatment, valid_confounders, title)

    def _generate_density_plot(self, data, matched_data, treatment, confounders, title):
        """
        Generate a single density plot with subplots for different confounders.
        Each row corresponds to a confounder, with treated and control groups in separate subplots.
        """
        sns.set_style("darkgrid")  
        num_confounders = len(confounders)
        fig, axes = plt.subplots(nrows=num_confounders, ncols=2, figsize=(20, 6 * num_confounders))
        
        if num_confounders == 1:
            axes = [axes]  

        for i, confounder in enumerate(confounders):
            # Treated group (left subplot)
            ax_treated = axes[i][0]
            sns.kdeplot(
                data[data[treatment] == 1][confounder], 
                label='Treated (Unmatched)', 
                color='blue', 
                fill=True, 
                alpha=0.3, 
                ax=ax_treated
            )
            sns.kdeplot(
                matched_data[matched_data[treatment] == 1][confounder], 
                label='Treated (Matched)', 
                color='orange', 
                fill=True,  
                alpha=0.3,  
                ax=ax_treated
            )
            ax_treated.set_title(f'Treated Group: {confounder} ({title})')
            ax_treated.set_xlabel(confounder)
            ax_treated.set_ylabel('Density')
            ax_treated.legend()
            ax_treated.grid(True)

            # Control group (right subplot)
            ax_control = axes[i][1]
            sns.kdeplot(
                data[data[treatment] == 0][confounder], 
                label='Control (Unmatched)', 
                color='blue', 
                fill=True,  
                alpha=0.3,  
                ax=ax_control
            )
            sns.kdeplot(
                matched_data[matched_data[treatment] == 0][confounder], 
                label='Control (Matched)', 
                color='orange', 
                fill=True, 
                alpha=0.3,  
                ax=ax_control
            )
            ax_control.set_title(f'Control Group: {confounder} ({title})')
            ax_control.set_xlabel(confounder)
            ax_control.set_ylabel('Density')
            ax_control.legend()
            ax_control.grid(True)

        plt.tight_layout()

        # Save the density plot
        density_plot_filename = f'density_plot_{title.lower().replace(" ", "_")}.png'
        density_plot_path = os.path.join(self.global_state.user_data.output_graph_dir, density_plot_filename)
        plt.savefig(density_plot_path, bbox_inches='tight')
        plt.close()

    def estimate_causal_effect_matching(self, treatment, outcome, confounders, cont_confounders, method="propensity_score", visualize=True):
        """
        Estimate the causal effect using matching methods.
        
        :param treatment: The treatment variable.
        :param outcome: The outcome variable.
        :param confounders: List of confounder column names.
        :param cont_confounders: List of continuous confounder column names.
        :param method: Matching method ("propensity_score" or "cem").
        :param visualize: Whether to visualize balance before and after matching.
        :return: Causal estimate, matched data, and balance plots.
        """
        print("\n" + "#" * 60)
        print(f"Estimating Causal Effect using Matching: {method}")
        print(f"Treatment: {treatment} | Outcome: {outcome}")
        print("#" * 60)

        # Step 1: Perform Matching
        if method == "propensity_score":
            matched_data = self._propensity_score_matching(confounders, treatment, outcome)
        elif method == "cem":
            # Pass confounders and continuous confounders to CEM
            matched_data = self._coarsened_exact_matching(confounders, cont_confounders, treatment, outcome)
        else:
            raise ValueError(f"Unsupported matching method: {method}")
        
        # Step 2: Coarsen continuous variables for balance checking
        if method == "cem":
            matched_data = coarsen_continuous_variables(matched_data, cont_confounders)

        # Step 3: Balance Checking (After Matching)
        if visualize:
            figs = self._check_balance(
                data=self.data,  # Original dataframe (before matching)
                matched_data=matched_data,  # Matched dataframe (after matching)
                treatment=treatment,
                outcome=outcome,
                confounders=confounders,
                cont_confounders=cont_confounders,
                title="Balance Checking"
            )
        else:
            figs = []

        # Step 4: Estimate Treatment Effect
        treated_mean = matched_data[matched_data[treatment] == 1][outcome].mean()
        control_mean = matched_data[matched_data[treatment] == 0][outcome].mean()
        ate = treated_mean - control_mean

        print(f"\nAverage Treatment Effect (ATE) using {method}: {ate}")
        return ate, matched_data, figs

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
        gcm.auto.assign_causal_mechanisms(self.causal_model, self.data)
        gcm.fit(self.causal_model, self.data)
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
        figs = [f'{self.global_state.user_data.output_graph_dir}/attribution_plot.png']
        # plt.show()  # Show the plot in the notebook or script
        # plt.close()
        return df, figs
    
    def attribute_distributional_changes(self, target_node, data_old, data_new, method='distribution_change', confidence_level=0.95):
        """
        Perform distributional change attribution to identify which causal mechanisms changed between two datasets assuming to get the old data fom the user
        Parameters:
        - target_node: The target node whose distribution change we want to explain.
        - data_old: DataFrame representing the system before the change.
        - data_new: DataFrame representing the system after the change.
        - method: distribution_change' (default) or 'distribution_change_robust'.
        - confidence_level: Confidence level for confidence intervals (default: 0.95).

        Returns:
        - DataFrame with attribution scores for each node.
        - List of file paths to the saved plots.
        """
        print("\n" + "#"*60)
        print(f"Performing Distributional Change Attribution for target node: {target_node}")
        print(f"Method: {method}")
        print("#"*60)

        # Define the path where the plot will be saved
        path = self.global_state.user_data.output_graph_dir

        # Performs distributional change attribution
        # distribution_change explain how the entire distribution of a target variable changed between two datasets.
        if method == 'distribution_change':
            attributions = gcm.distribution_change(
                causal_model=self.causal_model,
                old_data=data_old,
                new_data=data_new,
                target_node=target_node
            )
        # distribution_change_robust explains how the average of a target variable changed between two datasets
        elif method == 'distribution_change_robust':
            attributions = gcm.distribution_change_robust(
                causal_model=self.causal_model,
                data_old=data_old,
                data_new=data_new,
                target_node=target_node
            )
        else:
            raise ValueError("Invalid method. Choose 'distribution_change' or 'distribution_change_robust'.")

        # Convert results to a DataFrame
        rows = []
        for node, score in attributions.items():
            rows.append({
                "Node": node,
                "AttributionScore": score
            })

        df = pd.DataFrame(rows).sort_values("AttributionScore", ascending=False)

        # Extract info for plotting
        nodes = df["Node"]
        scores = df["AttributionScore"]

        # Create figure
        plt.figure(figsize=(10, 6))  
        plt.bar(nodes, scores, align='center', color='skyblue')
        plt.xlabel("Nodes")
        plt.ylabel("Attribution Score")
        plt.title(f"Distributional Change Attribution for {target_node}")
        plt.xticks(rotation=45) 
        plt.tight_layout()

        # Save bar plot
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            plot_filename = 'distribution_change_attribution_bar_plot.png'
            plot_path = os.path.join(path, plot_filename)
            print(f"Saving bar plot to {plot_path}")
            plt.savefig(plot_path)
        except Exception as e:
            print(f"Error saving bar plot: {e}")
        finally:
            plt.close()

        # Create violin plot
        plt.figure(figsize=(10, 6))  
        sns.violinplot(x=nodes, y=scores, inner="quartile", density_norm="width")
        plt.xlabel("Nodes")
        plt.ylabel("Attribution Score")
        plt.title(f"Distributional Change Attribution for {target_node}")
        plt.xticks(rotation=45) 
        plt.tight_layout()

        # Save violin plot
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            plot_filename = 'distribution_change_attribution_violin_plot.png'
            plot_path = os.path.join(path, plot_filename)
            print(f"Saving violin plot to {plot_path}")
            plt.savefig(plot_path)
        except Exception as e:
            print(f"Error saving violin plot: {e}")
        finally:
            plt.close()

        # Define figs as a list containing the file paths of the saved plots
        figs = ['distribution_change_attribution_bar_plot.png', 'distribution_change_attribution_violin_plot.png']

        return df, figs

    def estimate_effect_dml(self, outcome, treatment, T0, T1, X_col, W_col, query):
        if len(W_col) == 0:
            W_col = ['W']
            W = pd.DataFrame(np.zeros((len(self.data), 1)), columns=W_col)
            self.data = pd.concat([self.data, W], axis=1)
            self.global_state.user_data.processed_data = self.data
        # Algorithm selection and deliberation
        filter = DML_HTE_Filter(self.args)
        self.global_state = filter.forward(self.global_state, query)
        reranker = DML_HTE_Param_Selector(self.args, y_col=outcome, T_col=treatment, X_col=X_col, W_col=W_col)
        self.global_state = reranker.forward(self.global_state)
        programmer = DML_HTE_Programming(self.args, y_col=outcome, T_col=treatment, T0=T0, T1=T1, X_col=X_col, W_col=W_col)
        programmer.fit_model(self.global_state)
        # Estimate ate, att, hte
        ate, ate_lower, ate_upper = programmer.forward(self.global_state, task='ate')
        att, att_lower, att_upper = programmer.forward(self.global_state, task='att')
        hte, hte_lower, hte_upper = programmer.forward(self.global_state, task='hte')
        hte = pd.DataFrame({'hte': hte.flatten()})
        hte.to_csv(f'{self.global_state.user_data.output_graph_dir}/hte.csv', index=False)

        result = {'ate': [ate, ate_lower, ate_upper],
                  'att': [att, att_lower, att_upper],
                  'hte': [hte, hte_lower, hte_upper]}
        return result

    def estimate_effect_drl(self, outcome, treatment, T0, T1, X_col, W_col, query):
        if len(W_col) == 0:
            W_col = ['W']
            W = pd.DataFrame(np.zeros((len(self.data), 1)), columns=W_col)
            self.data = pd.concat([self.data, W], axis=1)
            self.global_state.user_data.processed_data = self.data
        # Algorithm selection and deliberation
        filter = DRL_HTE_Filter(self.args)
        self.global_state = filter.forward(self.global_state, query)
        reranker = DRL_HTE_Param_Selector(self.args, y_col=outcome, T_col=treatment, X_col=X_col, W_col=W_col)
        self.global_state = reranker.forward(self.global_state)
        programmer = DRL_HTE_Programming(self.args, y_col=outcome, T_col=treatment, T0=T0, T1=T1, X_col=X_col, W_col=W_col)
        # Estimate ate, att, hte
        ate, ate_lower, ate_upper = programmer.forward(self.global_state, task='ate')
        att, att_lower, att_upper = programmer.forward(self.global_state, task='att')
        hte, hte_lower, hte_upper = programmer.forward(self.global_state, task='hte')
        hte = pd.DataFrame({'hte': hte.flatten()})
        hte.to_csv(f'{self.global_state.user_data.output_graph_dir}/hte.csv', index=False)

        result = {'ate': [ate, ate_lower, ate_upper],
                  'att': [att, att_lower, att_upper],
                  'hte': [hte, hte_lower, hte_upper]}
        return result
    
    # TODO: Add def estimate_effect_iv()
    
    def counterfactual_estimation(self, treatment_name, response_name, observed_val = None, intervened_treatment = None):
        # observed_val should be a df as the processed data
        if observed_val is None:
            min_index = self.data[treatment_name].idxmin()
            observed_val = pd.DataFrame([self.data.loc[min_index].to_dict()])
        else:
            column_type = self.global_state.statistics.data_type_column

            for column in self.data.columns:
                if column_type[column] == "Category":
                    observed_cat = observed_val[column].iloc[0]
                    category_mapping = dict(enumerate(self.data[column].cat.categories))
                    observed_cat_code = {v: k for k, v in category_mapping.items()}.get(observed_cat)
                    observed_val.loc[0, column] = observed_cat_code

        if intervened_treatment is None:
            intervened_treatment = self.data[treatment_name].min() + 1.0

        est_sample = gcm.counterfactual_samples(
            self.causal_model,
            {treatment_name: lambda x: intervened_treatment},
            observed_data=observed_val)

        categories = ['Observed', 'Intervened']
        treatment_values = [observed_val[treatment_name].iloc[0], est_sample[treatment_name].iloc[0]]
        response_values = [observed_val[response_name].iloc[0], est_sample[response_name].iloc[0]]

        bar_width = 0.35

        x = np.arange(len(categories))  # [0, 1]

        plt.bar(x - bar_width / 2, treatment_values, width=bar_width, label= treatment_name, color='blue')
        plt.bar(x + bar_width / 2, response_values, width=bar_width, label=response_name, color='orange')

        plt.ylabel('Values')
        plt.title('Counterfactual Estimation')
        plt.xticks(x, categories)
        plt.legend()

        for i, v in enumerate(treatment_values):
            plt.text(i - bar_width / 2, v, f"{v:.1f}", ha='center')
        for i, v in enumerate(response_values):
            plt.text(i + bar_width / 2, v, f"{v:.1f}", ha='center')

        path = self.global_state.user_data.output_graph_dir

        if not os.path.exists(path):
            os.makedirs(path)
        print(f"Saving counterfactual estimation plot to {os.path.join(path, 'counterfactual_est_fig.jpg')}")
        plt.savefig(os.path.join(path, 'counterfactual_est_fig.jpg'))

    def simulate_intervention(self, treatment_name, response_name,
                              shift_intervention_val = None):
    # def simulate_intervention(self, treatment_name, response_name,
    #                             atomic_intervention_org_val=None, atomic_intervention_new_val=None,
    #                             shift_intervention_val=None):
        '''
            shift_intervention: a numeric value indicating how much shift of interventions have on original observations.
            atomic_intervention_org: the initial observed value of treatment for atomic intervention
            atomic_intervention_new: the value of new intervention on the treatment for atomic intervention
        '''
        # Atomic Intervention
        # if atomic_intervention_org_val is None:
        #     atomic_intervention_org_val = self.data[treatment_name].min()
        #
        # if atomic_intervention_new_val is None:
        #     atomic_intervention_new_val = atomic_intervention_org_val + 1
        #
        # atomic_samples_org = gcm.interventional_samples(self.causal_model,
        #                                             {treatment_name: lambda x: atomic_intervention_org_val},
        #                                             num_samples_to_draw=1000)
        # atomic_samples_new = gcm.interventional_samples(self.causal_model,
        #                                             {treatment_name: lambda x: atomic_intervention_new_val},
        #                                             num_samples_to_draw=1000)
        #
        # plt.figure(figsize=(12, 6))
        #
        # # Histogram for atomic_samples_org
        # plt.subplot(1, 2, 1)
        # plt.hist(atomic_samples_org[response_name], bins=30, color='blue', alpha=0.7)
        # plt.title(f'Original Atomic Samples: {response_name}')
        # plt.xlabel(response_name)
        # plt.ylabel('Frequency')
        #
        # # Histogram for atomic_samples_new
        # plt.subplot(1, 2, 2)
        # plt.hist(atomic_samples_new[response_name], bins=30, color='orange', alpha=0.7)
        # plt.title(f'New Atomic Samples: {response_name}')
        # plt.xlabel(response_name)
        # plt.ylabel('Frequency')
        #
        # plt.tight_layout()
        # # plt.show()
        #
        # path = self.global_state.user_data.output_graph_dir
        # if not os.path.exists(path):
        #     os.makedirs(path)
        #
        # print(f"Saving atomic intervention comparison plot to {os.path.join(path, 'atomic_intervention.jpg')}")
        # plt.savefig(os.path.join(path, 'atomic_intervention.jpg'))

        # Shift Intervention
        if shift_intervention_val is None:
            shift_intervention_val = 1
        shift_samples = gcm.interventional_samples(self.causal_model,
                                                    {treatment_name: lambda x: x + shift_intervention_val},
                                                    num_samples_to_draw=1000)

        plt.figure(figsize=(12, 6))

        # Histogram for atomic_samples_org
        plt.subplot(1, 2, 1)
        plt.hist(self.data[response_name], bins=30, color='blue', alpha=0.7)
        plt.title(f'Observed Data: {response_name}')
        plt.xlabel(response_name)
        plt.ylabel('Frequency')

        # Histogram for atomic_samples_new
        plt.subplot(1, 2, 2)
        plt.hist(shift_samples[response_name], bins=30, color='orange', alpha=0.7)
        plt.title(f'Shift Intervention: {response_name}')
        plt.xlabel(response_name)
        plt.ylabel('Frequency')

        plt.tight_layout()
        # plt.show()

        path = self.global_state.user_data.output_graph_dir
        if not os.path.exists(path):
            os.makedirs(path)

        print(f"Saving shift intervention comparison plot to {os.path.join(path, 'shift_intervention.jpg')}")
        plt.savefig(os.path.join(path, 'shift_intervention.jpg'))

        # Save dataset
        # print(f"Saving simulated dataset {os.path.join(path, 'simulated_atomic_intervention_org.csv')}")
        # atomic_samples_org.to_csv(os.path.join(path, 'simulated_atomic_intervention_org.csv'), index=False)
        #
        # print(f"Saving simulated dataset {os.path.join(path, 'simulated_atomic_intervention_new.csv')}")
        # atomic_samples_new.to_csv(os.path.join(path, 'simulated_atomic_intervention_new.csv'), index=False)

        print(f"Saving simulated dataset {os.path.join(path, 'simulated_shift_intervention.csv')}")
        shift_samples.to_csv(os.path.join(path, 'simulated_shift_intervention.csv'), index=False)

    def evaluate_treatment_effect_metrics(self, true_value, estimations, cis):
        """
        Evaluate metrics for Treatment Effect Estimation: Bias, MSE, and CI Coverage Probability.

        Parameters:
        - true_value (float): The true treatment effect.
        - estimations (np.ndarray): A 1D array of estimated treatment effects.
        - cis (list of tuples): A list of confidence intervals, where each element is a tuple (lower_bound, upper_bound).

        Returns:
        - dict: A dictionary containing Bias, MSE, and Coverage Probability.
        """
        # Convert inputs to numpy arrays for vectorized operations
        estimations = np.array(estimations)
        cis = np.array(cis)

        # Bias: Mean difference between estimation and true value
        bias = np.mean(estimations - true_value)

        # MSE: Mean squared error
        mse = np.mean((estimations - true_value) ** 2)

        # Coverage Probability: Proportion of CIs that contain the true value
        lower_bounds = cis[:, 0]
        upper_bounds = cis[:, 1]
        coverage_count = np.sum((true_value >= lower_bounds) & (true_value <= upper_bounds))
        coverage_probability = coverage_count / len(cis)

        # Return metrics as a dictionary
        return {
            "Bias": bias,
            "MSE": mse,
            "Coverage Probability": coverage_probability
        }

    def sensitivity_analysis(self, target_node, model, estimand, estimate, treatment, outcome):
        # if self.global_state.statistics.linearity:
        #      simulation_method = "linear-partial-R2"
        # else:
        #      simulation_method = "non-parametric-partial-R2",

        # Use the most important factor as the benchmark_common_causes
        file_exists = os.path.exists(f'{self.global_state.user_data.output_graph_dir}/shap_df.csv')
        if not file_exists:
            self.feature_importance(target_node, visualize=False)
        shap_df = pd.read_csv(f'{self.global_state.user_data.output_graph_dir}/shap_df.csv')
        max_col = shap_df.mean().idxmax()

        print('model.get_common_causes',model.get_common_causes())
        if model.get_common_causes() == [] or model.get_common_causes() is None:
            refute = model.refute_estimate(estimand, estimate, method_name="data_subset_refuter", subset_fraction=0.9)
            figs = []
        else:
            refute = model.refute_estimate(estimand, estimate,
                                method_name = "add_unobserved_common_cause",
                                simulation_method = "non-parametric-partial-R2",
                                benchmark_common_causes = [max_col],
                                effect_fraction_on_outcome = [1,2,3]
                                )
            plt.savefig(f'{self.global_state.user_data.output_graph_dir}/ate_refutation.png')
            plt.close()
            figs = ['ate_refutation.png']
        print(refute)

        return refute, figs 

    
    def forward(self, task, desc, key_node):
        if task == 'Feature Importance':
            parent_nodes, mean_shap_values, figs = self.feature_importance(key_node, visualize=True)
            response = generate_analysis_feature_importance(self.args, key_node, parent_nodes, mean_shap_values, desc)
            return response, figs 
        
        elif task == 'Treatment Effect Estimation':
            prompt = f"""
            I'm doing the Treatment Effect Estimation analysis, please identify the Treatment Variable in this description:
            {desc}
            The variable name must be among these variables: {self.data.columns}
            Only return me with the variable name, do not include anything else.
            """
            ### Check Treatment
            treatment = LLM_parse_query(self.args, None, 'You are an expert in Causal Discovery.', prompt)
            is_binary, treat, control = check_binary(self.data[treatment])
            while not is_binary:
                treatment_message = input(f"Your treatment column is not binary, please specify another variable name!")
                is_binary, treat, control = check_binary(self.data[treatment_message.strip()])
            treatment_message = input(f"Your treatment column is binary with treatment={treat} and control={control}\n"
                                        "Is there anything you want to correct? ")
            ### Check Confounder
            parent_nodes = list(self.G.predecessors(key_node))
            confounders = self._identify_confounders(treatment, key_node)
            
            remaining_var = list(set(self.data.columns) - set([treatment]) - set([key_node]) - set(confounders))
            #TODO: No Confounder Case and Add LLM variable selection
            add_confounder = input(f"These are Confounders between treatment {treatment} and outcome {key_node}: \n"
                      f"{','.join(confounders)}\n"
                      "Do you want to add any variables as confounders in your dataset? Please choose from the following:\n"
                      f"{','.join(remaining_var)}\n")
            add_confounder = [var.strip() for var in add_confounder.split(',') if var.strip() in remaining_var]
            confounders += add_confounder
            cont_confounders = [col for col in confounders if self.global_state.statistics.data_type_column[col]=='Continuous']
            ### Suggest method based on dataset characteristics
            if len(confounders) > 5:
                method = "dml" # Add drl here
            if len(confounders) - len(cont_confounders) > len(cont_confounders):  # If more than half discrete confounders
                method = "cem"
            else:
                method = "propensity_score"
            print("According to the characteristics of your data, we recommend you to use this Treatment Effect Estimation Method:\n"
                  f"{method}.")
            # TODO: Add logic for IV
            ### Run algorithm
            if method == "dml":
                ### Check Heterogeneous Variable
                hte_variable = input("Is there any heterogeneous variables you care about? If no, we can suggest some variables with LLM.\n")
                #TODO: Add LLM variable selection
                hte_variable = [var.strip() for var in hte_variable.split(',') if var.strip() in self.data.columns]
                result = self.estimate_effect_dml(outcome=key_node, treatment=treatment, T0=control, T1=treat,
                                                        X_col=hte_variable, W_col=confounders, query=desc)
                response, figs = generate_analysis_econml(self.args, self.global_state, key_node, treatment, parent_nodes, hte_variable, confounders, result, desc)

            if method == "drl":
                ### Check Heterogeneous Variable
                hte_variable = input("Is there any heterogeneous variables you care about? If no, we can suggest some variables with LLM.\n")
                #TODO: Add LLM variable selection
                hte_variable = [var.strip() for var in hte_variable.split(',') if var.strip() in self.data.columns]
                result = self.estimate_effect_drl(outcome=key_node, treatment=treatment, T0=control, T1=treat,
                                                        X_col=hte_variable, W_col=confounders, query=desc)
                response, figs = generate_analysis_econml(self.args, self.global_state, key_node, treatment, parent_nodes, hte_variable, confounders, result, desc)

            elif method in ["cem", "propensity_score"]:
                # Perform matching-based estimation
                ate, matched_data, figs = self.estimate_causal_effect_matching(
                    treatment=treatment,
                    outcome=key_node,
                    confounders=confounders,
                    cont_confounders=cont_confounders,
                    method=method,
                    visualize=True
                )
                response = generate_analysis_matching(self.args, treatment, key_node, method, confounders, ate, desc)
            return response, figs
        
        elif task == 'Anormaly Attribution':
            df, figs = self.attribute_anomalies(target_node=key_node, anomaly_samples=self.data, confidence_level=0.95)
            parent_nodes = list(self.G.predecessors(key_node))
            response = generate_analysis_anormaly(self.args, df, key_node, parent_nodes, desc)
            return response, figs
                       
        elif task == 'Distributional Change Attribution':
            # Split the dataset into two subsets (data_old and data_new)
            # For demonstration, we split the dataset into two halves
            data_old = self.data.iloc[:len(self.data)//2]  # First half as "old" data
            data_new = self.data.iloc[len(self.data)//2:]  # Second half as "new" data

            # Perform distributional change attribution
            df, figs = self.attribute_distributional_changes(target_node=key_node, data_old=data_old, data_new=data_new)
            response = generate_analysis_anormaly_dist(self.args, df, key_node, desc)
            return response, figs

        elif task == 'Counterfactual Estimation':
            prompt = f"""
                        I'm doing the Treatment Effect Estimation analysis, please identify the Treatment Variable in this description:
                        {desc}
                        The variable name must be among these variables: {self.data.columns}
                        Only return me with the variable name, do not include anything else.
                        """
            treatment = LLM_parse_query(self.args, None, 'You are an expert in Causal Discovery.', prompt)
            shift_intervention_val = input(f"""
                In this simulation, we are applying a 'shift intervention' to study how changes in the {treatment} 
                impact the {key_node}. A shift intervention involves modifying the value of a variable by a fixed
                amount (the 'shift value') while keeping other variables unchanged.\n
                For example, if we are studying the effect of increasing income on health outcomes, we might apply a
                shift intervention where the income variable is increased by a fixed amount, such as $500, for all individuals.\n
                Please enter the shift value:
            """)
            self.simulate_intervention(treatment_name = treatment, response_name = key_node, shift_intervention_val = shift_intervention_val)
        
        else:
            return None, None


def main(analysis, global_state):    
##################           
    results = analysis.estimate_effect_econml(
        treatment='PIP2',
        outcome='PIP3',
        covariates=[],   # or e.g. ["X1","X2"] if you have them
        controls=[]      # or e.g. ["W1","W2"] if needed
    )
    
    ate_val, ate_ci = results["ATE"]
    att_val, att_ci = results["ATT"]
    cate_vals, cate_cis = results["CATE"]

    print("EconML ATE:", ate_val, "CI:", ate_ci)
    print("EconML ATT:", att_val, "CI:", att_ci)
    if cate_vals is not None:
        print("Mean CATE:", np.mean(cate_vals))

    # Optional plot the distribution of the CATE:
    if cate_vals is not None:
        dist_fig_path = os.path.join(global_state.user_data.output_graph_dir, "cate_dist.png")
        analysis.plot_cate_distribution(cate_vals, dist_fig_path)


    # EXAMPLE: Compare with linear approach, specifying a different target_units
    lin_estimate, lin_p_value, refutaion, figs = analysis.estimate_causal_effect(
        treatment='PIP2',
        outcome='PIP3',
        target_units=lambda df: df[df['PIP2'] > 100].index  # example subgroup
    )
    print("Linear Subgroup Estimate (CATE approach):", lin_estimate.value)
    print("p-value:", lin_p_value)
    # Testing code to check the above functions replace it appropriately
    # Perform Matching-Based Effect Estimation
    response, figs = analysis.forward(
        task='Matching-Based Effect Estimation',
        desc='Analyze the effect of PIP2 on PIP3 using matching.',
        key_node='PIP3'
    )
    print(response)
    
    # Example: Run CEM for treatment 'PIP2' and outcome 'PIP3'
    treatment = 'PIP2'
    outcome = 'PIP3'
    confounders = analysis._identify_confounders(treatment, outcome)
    cont_confounders = [col for col in confounders if analysis.data[col].nunique() > 10]  # Continuous confounders
    
    ate, matched_data = analysis.estimate_causal_effect_matching(
        treatment=treatment,
        outcome=outcome,
        confounders=confounders,
        cont_confounders=cont_confounders,
        method="cem",
        visualize=True
    )
    
    print(f"Average Treatment Effect (ATE) using CEM: {ate}")


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