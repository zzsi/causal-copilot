import numpy as np
import pandas as pd
import networkx as nx

# REMOVE or comment out the direct import of CausalModel from DoWhy:
# from dowhy import gcm, CausalModel

# Keep gcm if you still want anomaly attribution and interventions:
from dowhy import gcm, CausalModel
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
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_analysis.DML.hte_filter import HTE_Filter as DML_HTE_Filter
from causal_analysis.DML.hte_params import HTE_Param_Selector as DML_HTE_Param_Selector
from causal_analysis.DML.hte_program import HTE_Programming as DML_HTE_Programming
from causal_analysis.DRL.hte_filter import HTE_Filter as DRL_HTE_Filter
from causal_analysis.DRL.hte_params import HTE_Param_Selector as DRL_HTE_Param_Selector
from causal_analysis.DRL.hte_program import HTE_Programming as DRL_HTE_Programming
from causal_analysis.IV.hte_filter import HTE_Filter as IV_HTE_Filter
from causal_analysis.IV.hte_params import HTE_Param_Selector as IV_HTE_Param_Selector
from causal_analysis.IV.hte_program import HTE_Programming as IV_HTE_Programming
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

    def feature_importance(self, target_node, linearity, visualize=True):
        print('start feature importance analysis')
        parent_nodes = list(self.G.predecessors(target_node))

        X = self.data.drop(columns=[target_node])
        y = self.data[[target_node]]
        X_background = shap.utils.sample(X, int(len(X)*0.2))  # background distribution for SHAP
        if linearity:
            model_linear = sklearn.linear_model.LinearRegression()
            model_linear.fit(X, y)
            explainer_linear = shap.Explainer(model_linear.predict, X_background)
            shap_values = explainer_linear(X)
        else:
            # Xd = xgboost.DMatrix(X, label=y)
            # model_xgb = xgboost.train({"eta": 1, "max_depth": 3, "base_score": 0, "lambda": 0}, Xd, 1)
            # pred = model_xgb.predict(Xd, output_margin=True)
            model_rf = RandomForestRegressor()
            model_rf.fit(X, y)
            explainer_rf = shap.TreeExplainer(model_rf)
            shap_values = explainer_rf(X)
        # Mean absolute SHAP values
        shap_df = pd.DataFrame(np.abs(shap_values.values), columns=X.columns)
        mean_shap_values = shap_df.mean()
        shap_df.to_csv(f'{self.global_state.user_data.output_graph_dir}/shap_df.csv', index=False)
        
        figs = []
        if visualize == True:
            # 1st SHAP Plot beeswarm
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = shap.plots.beeswarm(shap_values, show=False)
            plt.savefig(f'{self.global_state.user_data.output_graph_dir}/shap_beeswarm_plot.png', bbox_inches='tight')  # Save as PNG
            figs.append(f'{self.global_state.user_data.output_graph_dir}/shap_beeswarm_plot.png')

            # 2nd SHAP Plot Bar
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = shap.plots.bar(shap_values, show=False)
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
        figs = generate_density_plot(self.global_state, data, matched_data, treatment, valid_confounders, title)
        return figs

    def _matched_treatment_effect(self, matched_data, treatment, outcome, treat, control, X_col):
        treated_mean = matched_data[matched_data[treatment] == treat][outcome].mean()
        control_mean = matched_data[matched_data[treatment] == control][outcome].mean()
        ate = treated_mean - control_mean
        cont_X_col = [var for var in X_col if self.global_state.statistics.data_type_column[var]=='Continuous']
        coarsen_data = coarsen_continuous_variables(matched_data, cont_X_col)
        cate_results = {}
        for label in X_col:
            if label in cont_X_col:
                group_col = f'coarsen_{label}'
            else:
                group_col = label
            cate = coarsen_data.groupby(group_col).apply(lambda x: x[x[treatment] == treat][outcome].mean() - x[x[treatment] == 0][outcome].mean())
            cate_results[label] = cate
        
        return ate, cate_results
    
    def estimate_causal_effect_matching(self, treatment, outcome, confounders, X_col, treat, control, cont_confounders, method="propensity_score", visualize=True):
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
        ate, cate_results = self._matched_treatment_effect(matched_data, treatment, outcome, treat, control, X_col)
        cate_plot_path = os.path.join(self.global_state.user_data.output_graph_dir, 'cate_plot.png')
        plot_cate_bars_by_group(cate_results, cate_plot_path)
        figs.append(cate_plot_path)
        print(f"\nAverage Treatment Effect (ATE) using {method}: {ate}")
        return ate, cate_results, matched_data, figs

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
    # TODO: Add def contains_iv() to check where the causal graph contains IV
    # TODO: Add def estimate_effect_iv()
    def estimate_effect_iv(self, outcome, treatment, instrument_variable, T0, T1, X_col, W_col, query):
        if len(W_col) == 0:
            W_col = ['W']
            W = pd.DataFrame(np.zeros((len(self.data), 1)), columns=W_col)
            self.data = pd.concat([self.data, W], axis=1)
            self.global_state.user_data.processed_data = self.data
        # Algorithm selection and deliberation
        filter = IV_HTE_Filter(self.args)
        self.global_state = filter.forward(self.global_state, query)
        reranker = IV_HTE_Param_Selector(self.args, y_col=outcome, T_col=treatment, Z_col=instrument_variable, X_col=X_col, W_col=W_col)
        self.global_state = reranker.forward(self.global_state)
        programmer = IV_HTE_Programming(self.args, y_col=outcome, T_col=treatment, Z_col=instrument_variable, T0=T0, T1=T1, X_col=X_col, W_col=W_col)
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
    
    def estimate_effect_linear(self, 
                               treatment, 
                               outcome, 
                               control_value=0, 
                               treatment_value=1):
        """
        Estimate the causal effect of a treatment on an outcome using
        DoWhy (backdoor.linear_regression).

        :param treatment: str, name of the treatment variable
        :param outcome: str, name of the outcome variable
        :param control_value: value representing 'control'
        :param treatment_value: value representing 'treated'
        :param target_units: 'ate' (default), 'treated', or a callable (lambda df: df[...] for subgroups)

        :return: (causal_estimate, p_value)
        """
        print("\n" + "#"*60)
        print(f"Estimating Causal Effect of Treatment: {treatment} on Outcome: {outcome}")
        print(f"Method: backdoor.linear_regression | target_units='ate'")
        print("#"*60)

        print("\nCreating Causal Model...")
        print(self.dot_graph)
        model = CausalModel(
            data=self.data,
            treatment=treatment,
            outcome=outcome,
            graph=self.dot_graph  # The DOT-format graph
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        # print("\nIdentified Estimand:")
        # print(identified_estimand)
        # Backdoor linear_regression
        causal_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            control_value=control_value,
            treatment_value=treatment_value,
            target_units='ate'
        )
        print("\nCausal Estimate:")
        print(causal_estimate)
        # Significance Test
        significance_results = causal_estimate.estimator.test_significance(
            self.data, 
            causal_estimate.value
        )
        p_value = significance_results['p_value'][0]
        print("Significance Test Results:", p_value)

        return causal_estimate, p_value

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
        gcm.auto.assign_causal_mechanisms(self.causal_model, self.data)
        gcm.fit(self.causal_model, self.data)
        shift_samples = gcm.interventional_samples(self.causal_model,
                                                    {treatment_name: lambda x: x + shift_intervention_val},
                                                    num_samples_to_draw=len(self.data))

        plt.figure(figsize=(8, 6))
        # Histogram for atomic_samples_org
        plt.subplot(1, 2, 1)
        plt.hist(self.data[response_name], bins=30, color='#409fde', alpha=0.7)
        plt.title(f'Observed Data: {response_name}')
        plt.xlabel(response_name)
        plt.ylabel('Frequency')

        # Histogram for atomic_samples_new
        plt.subplot(1, 2, 2)
        plt.hist(shift_samples[response_name], bins=30, color='#f77f3e', alpha=0.7)
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
        figs = [os.path.join(path, 'shift_intervention.jpg')]
        # Save dataset
        # print(f"Saving simulated dataset {os.path.join(path, 'simulated_atomic_intervention_org.csv')}")
        # atomic_samples_org.to_csv(os.path.join(path, 'simulated_atomic_intervention_org.csv'), index=False)
        #
        # print(f"Saving simulated dataset {os.path.join(path, 'simulated_atomic_intervention_new.csv')}")
        # atomic_samples_new.to_csv(os.path.join(path, 'simulated_atomic_intervention_new.csv'), index=False)

        print(f"Saving simulated dataset {os.path.join(path, 'simulated_shift_intervention.csv')}")
        shift_samples.to_csv(os.path.join(path, 'simulated_shift_intervention.csv'), index=False)
        # Boxplot Comparison to summarize distributions
        plt.figure(figsize=(8, 6))
        data_to_plot = [self.data[response_name], shift_samples[response_name]]
        plt.boxplot(data_to_plot, labels=["Observed Data", "Shift Intervention"], patch_artist=True)
        plt.title(f'Shift Intervention Comparison: {response_name}')
        plt.ylabel(response_name)
        print(f"Saving box-plot comparison plot to {os.path.join(path, 'shift_intervention_boxplot.jpg')}")
        plt.savefig(os.path.join(path, 'shift_intervention_boxplot.jpg'))
        figs_boxplot = os.path.join(path, 'shift_intervention_boxplot.jpg')
        figs.append(figs_boxplot)
        return figs, shift_samples

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

    
    def forward(self, task, desc, key_node, chat_history):
        if task == 'Feature Importance':
            linearity = self.global_state.statistics.linearity
            if linearity:
                chat_history.append((None, "💡 We use the linear model to calculate SHAP Value because your dataset is linear."))
            else:
                chat_history.append((None, "💡 We use the non-linear model to calculate SHAP Value because your dataset is non-linear."))
            parent_nodes, mean_shap_values, figs = self.feature_importance(key_node, linearity, visualize=True)
            response = generate_analysis_feature_importance(self.args, key_node, parent_nodes, mean_shap_values, desc)
            for fig in figs:
                chat_history.append((None, (f'{fig}',)))
            chat_history.append((None, response))
            return response, figs, chat_history 
        
        elif task == 'Treatment Effect Estimation':
            task_info = self.global_state.inference.task_info[self.global_state.inference.task_index]
            confounders = task_info['confounders']
            cont_confounders = task_info['cont_confounders']
            treatment = task_info['treatment']
            treat = task_info['treat']
            control = task_info['control']
            hte_variables = task_info['X_col']
            parent_nodes = list(self.G.predecessors(key_node))
            method = task_info['hte_method']
            ### Suggest method based on dataset characteristics
            # TODO: Check for IV in the Causal Graph
            exist_IV = False
            if exist_IV:
                iv_variable = None
                global_state.inference.task_info[self.global_state.inference.task_index]['IV'] = iv_variable
                method = "iv"

            elif len(confounders) <= 5:
                if len(confounders) - len(cont_confounders) > len(cont_confounders):  # If more than half discrete confounders
                    method = "cem"
                else:
                    method = "propensity_score"
            else:
                if len(self.global_state.user_data.processed_data) > 2000:
                    method = "dml"
                else:
                    method = "drl"

            ### Run algorithm
            if method in ["dml", "drl"]:
                if method == "dml":
                    result = self.estimate_effect_dml(outcome=key_node, treatment=treatment, T0=control, T1=treat,
                                                            X_col=hte_variables, W_col=confounders, query=desc)
                else:
                    result = self.estimate_effect_drl(outcome=key_node, treatment=treatment, T0=control, T1=treat,
                                                            X_col=hte_variables, W_col=confounders, query=desc)
                response, figs = generate_analysis_econml(self.args, self.global_state, key_node, treatment, parent_nodes, hte_variables, confounders, result, desc)
                chat_history.append(("📝 Analyze for ATE and ATT...", None))
                chat_history.append((None, response[0]))
                chat_history.append(("📝 Analyze for HTE...", None))
                for fig in figs:
                    chat_history.append((None, (f'{fig}',)))
                chat_history.append((None, response[1]))

            # TODO: Add IV Estimation
            if method == "iv":
                pass

            elif method in ["cem", "propensity_score"]:
                # Perform matching-based estimation
                ate, cate_result, matched_data, figs = self.estimate_causal_effect_matching(
                    treatment=treatment,
                    outcome=key_node,
                    confounders=confounders,
                    X_col=hte_variables,
                    treat=treat,
                    control=control,
                    cont_confounders=cont_confounders,
                    method=method,
                    visualize=True
                )
                response = generate_analysis_matching(self.args, treatment, key_node, method, confounders, ate, cate_result, desc)
                chat_history.append(("📝 Matching Balance Checking...", None))
                chat_history.append((None, (f'{figs[0]}',)))
                chat_history.append((None, response[0]))
                chat_history.append(("📝 Analyze for ATE...", None))
                chat_history.append((None, response[1]))
                chat_history.append(("📝 Analyze for CATE...", None))
                chat_history.append((None, (f'{figs[1]}',)))
                chat_history.append((None, response[2]))

            elif method == 'linear_regression':
                ate, p_value = self.estimate_effect_linear(
                               treatment=treatment,
                               outcome=key_node,
                               control_value=control, 
                               treatment_value=treat)
                response = generate_analysis_linear_regression(args, treatment, key_node, ate, p_value, desc)
            
            return response, figs, chat_history
        
        elif task == 'Anormaly Attribution':
            df, figs = self.attribute_anomalies(target_node=key_node, anomaly_samples=self.data, confidence_level=0.95)
            parent_nodes = list(self.G.predecessors(key_node))
            response = generate_analysis_anormaly(self.args, df, key_node, parent_nodes, desc)
            for fig in figs:
                chat_history.append((None, (f'{fig}',)))
            chat_history.append((None, response))
            return response, figs, chat_history
                       
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
            treatment = self.global_state.inference.task_info[self.global_state.inference.task_index]['treatment']
            shift_intervention_val = self.global_state.inference.task_info[self.global_state.inference.task_index]['shift_value']
            figs, shift_df = self.simulate_intervention(treatment_name = treatment, response_name = key_node, shift_intervention_val = shift_intervention_val)
            response = generate_conterfactual_estimation(self.args, self.global_state, shift_intervention_val, shift_df, treatment, key_node, desc)
            for fig in figs:
                chat_history.append((None, (f'{fig}',)))
            chat_history.append((None, response))
            return response, figs, chat_history
        else:
            return None, None, chat_history


if __name__ == '__main__':
    import argparse
    import pickle
    def parse_args():
        parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

        # Input data file
        parser.add_argument(
            '--data-file',
            type=str,
            default="demo_data/20250121_223113/lalonde/lalonde.csv",
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
        args = parser.parse_args()
        return args
    
    args = parse_args()
    with open('demo_data/20250121_223113/lalonde/output_graph/PC_global_state.pkl', 'rb') as file:
        global_state = pickle.load(file)
    
    my_analysis = Analysis(global_state, args)
    # my_analysis.estimate_effect_dml(outcome='re78', treatment='treat', 
    #                                 T0=0, T1=1, 
    #                                 X_col=['age', 'nodegr'], 
    #                                 W_col=['educ', 'age', 'married', 'nodegr'], 
    #                                 query='What is the treatment effect of treat on re78')
    my_analysis.feature_importance(target_node='re78', linearity=False, visualize=True)
    #my_analysis.simulate_intervention(treatment_name = 'married', response_name = 're78', shift_intervention_val =1)