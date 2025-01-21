from causal_analysis.help_functions import *

def generate_analysis_econml(global_state, key_node, treatment, parent_nodes, X_col, W_col, result, query):
    # Analysis for Effect Estimation
    ate, ate_lower, ate_upper = result['ate']
    att, att_lower, att_upper = result['att']
    hte, hte_lower, hte_upper = result['hte']
    prompt_ate = f"""
    I'm doing the Treatment Effect Estimation analysis and please help me to write a brief analysis in bullet points.
    Here are some informations:
    **Result Variable we care about**: {key_node}
    **Treatment Variable**: {treatment}
    **Parent Nodes of the Result Variable**: {parent_nodes}
    **Confounders**: {W_col}
    
    **Causal Estimate Result**: 
    Average Treatment Effect: {ate}, Confidence Interval: ({ate_lower}, {ate_upper})
    Average Treatment Effect on Treated: {att}, Confidence Interval: ({att_lower}, {att_upper})
    """
    response_ate = LLM_parse_query(None, 'You are an expert in Causal Discovery.', prompt_ate)

    prompt_hte = f"""
    I'm doing the Heterogeneous Treatment Effect Estimation and please help me to write a brief analysis in bullet points.
    Here are some informations:
    **Result Variable we care about**: {key_node}
    **Treatment Variable**: {treatment}
    **Heterogeneous Confounders we coutrol**: {X_col}
    **Description from User**: {query}
    **Information in the plot**: Distribution of the HTE; Violin plot of the CATE grouped by: {X_col}
    """
    response_hte = LLM_parse_query(None, 'You are an expert in Causal Discovery.', prompt_hte)
    
    figs = {'hte_dist': None,
            'cate_dist': None}
    dist_fig_path = f'{global_state.user_data.output_graph_dir}/hte_dist.png'
    plot_hte_dist(hte, dist_fig_path)
    figs['hte_dist'] = dist_fig_path
    cate_fig_path = f'{global_state.user_data.output_graph_dir}/cate_dist.png'
    plot_cate_violin(global_state, hte, X_col, cate_fig_path)
    figs['cate_dist'] = cate_fig_path

    return 

def generate_analysis_matching(treatment, outcome, method, W_col, ate, query):
    # Generate a response using the LLM
    prompt = f"""
    I'm doing the Matching-Based Effect Estimation analysis and please help me to write a brief analysis in bullet points.
    Here are some informations:
    **Treatment Variable**: {treatment}
    **Outcome Variable**: {outcome}
    **Matching Method**: {method}
    **Confounders**: {W_col}
    **Average Treatment Effect (ATE)**: {ate}
    **Description from User**: {query}
    """
    response = LLM_parse_query(None, 'You are an expert in Causal Discovery.', prompt)
    return response
        
    # # Analysis for Refutation Analysis
    # prompt = f"""
    # I'm doing the refutation analysis for my treatment effect estimation, please help me to write a brief analysis in bullet points.
    # **Contents you need to incode**
    # 1. Brief Introduction of the refutation analysis method we use
    # 2. Summary of the refutation analysis result
    # 3. Brief Interpretation of the plot
    # 4. Conclude whether the treatment effect estimation is reliable or not based on the refutation analysis result
    # Here are some informations:
    # **Result Variable we care about**: {key_node}
    # **Causal Estimate Result**: {causal_estimate}
    # """
    # if figs == []:
    #     prompt += f"""
    # **Refutation Result**: 
    # {str(refutation)}
    # **Method We Use**: Use Data Subsampling to answer Does the estimated effect change significantly when we replace the given dataset with a randomly selected subset?
    # """
    # else:
    #     prompt += f"""
    # **Sensitivity Analysis Result**: 
    # {str(refutation)}
    # **Method We Use**:
    # Sensitivity analysis helps us study how robust an estimated effect is when the assumption of no unobserved confounding is violated. That is, how much bias does our estimate have due to omitting an (unobserved) confounder? Known as the omitted variable bias (OVB), it gives us a measure of how the inclusion of an omitted common cause (confounder) would have changed the estimated effect.
    # **Information in the plot**: 
    # a. The x-axis shows hypothetical partial R2 values of unobserved confounder(s) with the treatment. The y-axis shows hypothetical partial R2 of unobserved confounder(s) with the outcome. 
    # b. At <x=0,y=0>, the black diamond shows the original estimate (theta_s) without considering the unobserved confounders.
    # c. The contour levels represent adjusted estimate of the effect, which would be obtained if the unobserved confounder(s) had been included in the estimation model. 
    # d. The red contour line is the critical threshold where the adjusted effect goes to zero. Thus, confounders with such strength or stronger are sufficient to reverse the sign of the estimated effect and invalidate the estimate’s conclusions. 
    # e. The red triangle shows the estimated effect when the unobserved covariate has 1 or 2 or 3 times partial-R^2 of a chosen benchmark observed covariate with the outcome.
    # **Description from User**: {desc}
    # """
    # response2 = LLM_parse_query(None, 'You are an expert in Causal Discovery.', prompt)
    # response = response1 + '\n' + response2

def generate_analysis_feature_importance(key_node, parent_nodes, mean_shap_values, desc):
    prompt = f"""
    I'm doing the feature importance analysis and please help me to write a brief analysis in bullet points.
    Here are some informations:
    **Result Variable we care about**: {key_node}
    **Parent Nodes of the Result Variable**: {parent_nodes}
    ""Description from User**: {desc}
    **Mean of Shapley Values**: {mean_shap_values}
    """
    response = LLM_parse_query(None, 'You are an expert in Causal Discovery.', prompt)

def generate_analysis_anormaly(df, key_node, parent_nodes, desc):
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
    response = LLM_parse_query(None, 'You are an expert in Causal Discovery.', prompt)
    return response

def generate_analysis_anormaly_dist(df, key_node, desc):
    # Generate a response using the LLM
    prompt = f"""
    I'm doing the Distributional Change Attribution analysis and please help me to write a brief analysis in bullet points.
    Here are some informations:
    **Target Variable we care about**: {key_node}
    **Attribution Scores**: 
    {df.to_markdown()}
    **Description from User**: {desc}
    **Methods to calculate Distributional Change Attribution**
    We compared two datasets (old and new) to identify which nodes in the causal graph contributed most to the change in the distribution of the target variable.
    """
    response = LLM_parse_query(None, 'You are an expert in Causal Discovery.', prompt)
    return response