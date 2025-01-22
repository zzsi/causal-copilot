import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from openai import OpenAI

def convert_adj_mat(mat):
    # In downstream analysis, we only keep direct edges and ignore all undirected edges
    mat = np.array(mat)
    mat = (mat == 1).astype(int)
    G = mat.T
    return G

def coarsen_continuous_variables(data, cont_confounders, bins=5):
    """
    Coarsen continuous variables into bins for CEM.
    
    :param data: The dataset.
    :param cont_confounders: List of continuous confounder column names.
    :param bins: Number of bins to create for each continuous variable.
    :return: Dataset with coarsened columns.
    """
    for col in cont_confounders:
        if col in data.columns:
            coarsened_col = f'coarsen_{col}'
            data[coarsened_col] = pd.cut(data[col], bins=bins, labels=False)
    return data

def plot_hte_dist(hte, fig_path):
    plt.figure(figsize=(8, 6))
    sns.histplot(hte['hte'], bins=30, kde=True, color='skyblue', alpha=0.7)
    plt.axvline(hte['hte'].mean(), color='firebrick', linestyle='--', label='Mean HTE')
    plt.xlabel("Heterogeneous Treatment Effect (HTE)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Heterogeneous Treatment Effects")
    # Save figure
    plt.savefig(fig_path)

def plot_cate_violin(global_state, hte, confounders, cont_confounders, fig_path):
    data = global_state.user_data.processed_data
    coarsen_data = coarsen_continuous_variables(data, cont_confounders)
    data = pd.concat([coarsen_data, hte], axis=1)
    num_groups = len(confounders)
    fig, axes = plt.subplots(num_groups, 1, figsize=(10, 6 * num_groups), sharex=False)
    if num_groups == 1:
        axes = [axes]  # Ensure axes is always a list for consistency

    for ax, group_col in zip(axes, confounders):
        if group_col in cont_confounders:
            sns.violinplot(x=f'coarsen_{group_col}', y='hte', data=data, ax=ax, inner="quartile", scale="width")
        else:
            sns.violinplot(x=group_col, y='hte', data=data, ax=ax, inner="quartile", scale="width")
        # Customize the subplot
        ax.set_title(f"CATE Distribution by {group_col.capitalize()}")
        ax.set_xlabel(group_col.capitalize())
        ax.set_ylabel("CATE")
        ax.grid(True)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(fig_path)

def plot_cate_distribution(cate_array, fig_path):
    """
    Plot distribution of the CATE (HTE) array and save to fig_path.
    """
    plt.figure(figsize=(8,6))
    sns.histplot(cate_array, bins=30, kde=True, color='skyblue', alpha=0.7)
    plt.title("Distribution of Estimated CATE")
    plt.xlabel("CATE Value")
    plt.ylabel("Frequency")
    plt.axvline(np.mean(cate_array), color='red', linestyle='--', label='Mean CATE')
    plt.legend()
    plt.savefig(fig_path)
    plt.close()

def plot_cate_bars_by_group(cate_arrays, group_labels, fig_path):
    n_groups = len(group_labels)
    n_rows = (n_groups + 2) // 3  # Calculate number of rows needed (3 plots per row)
    n_cols = min(3, n_groups)
    fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
    for idx, group in enumerate(group_labels):
        plt.subplot(n_rows, n_cols, idx + 1)
        # Create bar plot for each group
        counts, bins, _ = plt.hist(cate_arrays[group], bins=30, 
                                 density=True, alpha=0)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.bar(bin_centers, counts, width=np.diff(bins), 
                color='skyblue', alpha=0.7, 
                align='center', label='CATE')
        plt.title(f"CATE Distribution - {group}")
        plt.xlabel("CATE Value")
        plt.ylabel("Density")
        # Add mean line
        group_mean = np.mean(cate_arrays[group])
        plt.axvline(group_mean, color='red', linestyle='--', 
                   label=f'Mean: {group_mean:.3f}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

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

def check_binary(column):
    unique_values = column.unique()
    treat = max(column)
    control = min(column)
    return len(unique_values) == 2, treat, control
