import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
import random
import math

from algorithm.program import Programming


def bootstrap(data,
              algorithm_setup,
              full_graph,
              ts: bool = False, boot_num: int = 500):
    '''
    :param data: Given Tabular Data in Pandas DataFrame format
    :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
    :param full_graph: Causal graph using the full dataset
    :param ts: indicator of time-series data
    :param boot_num: number of bootstrap iterations
    :return: a dict of obvious errors in causal analysis results based on bootstrap,
             e.g. {"X->Y: "Forced", "Y->Z: "Forbidden", "K <-> Z": Forced};
             bootstrap probability of directed edges and common cause.
    '''

    n, m = data.shape
    errors = {}

    boot_effect_save = np.empty((m, m, boot_num)) # Save graphs based on bootstrapping

    boot_probability = np.empty((m, m)) # Save bootstrap probability of directed edges
    boot_prob_common_cause = np.empty((m, m))  # Save bootstrap probability of common cause

    for boot_time in range(boot_num):

        # Bootstrap samples
        if not ts:
            # General bootstrapping
            boot_index = random.choices(range(n), k=n)
            boot_sample = data.iloc[boot_index, :]
        elif ts:
            # Moving block bootstrapping for time-series
            block_size = 10
            block_num = math.ceil(n / block_size)
            block_start = random.sample(range(n - block_size + 1), block_num)

            blocks = [list(range(start, start + block_size)) for start in block_start]
            subsets = [data.iloc[block] for block in blocks]

            boot_sample = pd.concat(subsets, ignore_index=True).iloc[1:n]

            # Use PC as an example: cg_boot = pc(boot_sample.to_numpy())
            # Use functions in code generation part
            cg_boot = Programming(data, algorithm_setup)
            boot_effect_save[:, :, boot_time] = cg_boot.G.graph

        for i in range(m):
            for j in range(i + 1, m):
                # Consider three cases: i->j, j->i and i<->j

                # boot_probability[i,j] represent the bootstrap probability of the edge i –> j
                boot_probability[i,j] = np.mean(np.logical_and(boot_effect_save[j, i, :] == 1,boot_effect_save[i, j, :] == -1))

                # Indicator of existence of path i->j
                exist_ij = np.logical_and(full_graph[j, i] == 1, full_graph[i, j] == -1)

                # cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicate i –> j or
                # cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j
                # Force the path if the probability is greater than 0.95
                if  boot_probability[i,j] >= 0.95:
                    # Compare with the initial graph: if the path doesn't exist then force it
                    if not exist_ij:
                        errors[data.columns[i] + "->" + data.columns[j]] = "Forced"

                # Forbid the path if the probability is less than 0.05
                elif boot_probability[i,j] <= 0.05:
                    # Compare with the initial graph: if the path exist then forbid it
                    if exist_ij:
                        errors[data.columns[i] + "->" + data.columns[j]] = "Forbidden"

                # j –> i
                # boot_probability[j,i] represent the bootstrap probability of the edge j –> i
                boot_probability[j, i] = np.mean(np.logical_and(boot_effect_save[i, j, :] == 1,boot_effect_save[j, i, :] == -1))

                # Indicator of existence of path i->j
                exist_ji = np.logical_and(full_graph[i, j] == 1, full_graph[j, i] == -1)

                if boot_probability[j, i] >= 0.95:
                    # Compare with the initial graph: if the path doesn't exist then force it
                    if not exist_ji:
                        errors[data.columns[j] + "->" + data.columns[i]] = "Forced"
                elif boot_probability[j, i] <= 0.05:
                    # Compare with the initial graph: if the path exist then forbid it
                    if exist_ji:
                        errors[data.columns[j] + "->" + data.columns[i]] = "Forbidden"

                # i <–> j
                # boot_prob_common_cause[i, j] represent the bootstrap probability of the common cause i <–> j
                boot_prob_common_cause[i, j] = np.mean(np.logical_and(boot_effect_save[i, j, :] == -1,boot_effect_save[j, i, :] == -1))

                # Indicator of common cause i<->j
                exist_common_ij = np.logical_and(full_graph[i, j] == -1, full_graph[j, i] == -1)

                if boot_prob_common_cause[i, j] >= 0.95:
                    # Compare with the initial graph: if the common cause doesn't exist then force it
                    if not exist_common_ij:
                        errors[data.columns[i] + "<->" + data.columns[j]] = "Forced"
                elif boot_prob_common_cause[i, j] <= 0.05:
                    # Compare with the initial graph: if the path exist then forbid it
                    if exist_ji:
                        errors[data.columns[j] + "<->" + data.columns[i]] = "Forbidden"

    return errors, boot_probability, boot_prob_common_cause


def llm_evaluation(data,
                   full_graph,
                   llm_setup,
                   knowledge_docs):
    '''
    :param data: Given Tabular Data in Pandas DataFrame format
    :param full_graph: Causal graph using the full dataset
    :param llm_setup: information of configurations of GPT-4
    :param knowledge_docs: A doc containing all necessary domain knowledge information from GPT-4.
    :return: obvious errors in causal analysis results based on LLM,
             e.g. {"X->Y: "Forced", "Y->Z: "Forbidden"}
    '''

    from openai import OpenAI
    client = OpenAI(organization=llm_setup.organization, project=llm_setup.project, api_key=llm_setup.apikey)

    table_columns = '\t'.join(data.columns)

    prompt = ("Based on this knowledge document: %s \n\n"
              "and these column names %s\n\n, "
              "conclude the causal relationship between each pair of variables among the column names,"
              "and how much confidence you have about such relationship. "
              "You should also report the existence of common cause between variables. \n\n"
              "The output of your response should be a dictionary, for example, if you have 95% confidence that X cause Y,"
              "and only 5% confidence that there is a common cause between Y and Z, "
              "you should output {'X->Y': 95%, 'Y<->Z': 5%}.") % (knowledge_docs,table_columns)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # known_effect is a dict have format: {'X->Y': 95%, 'Y<->Z': 5%}
    known_effect = response.choices[0].message.content

    errors = {}

    for key in known_effect.keys():

        # Consider common cause
        if "<->" in key:
            split_key = key.split("<->")
            i = data.columns.get_loc(split_key[0])
            j = data.columns.get_loc(split_key[1])

            # Indicator of existence of path i<->j
            exist_common_ij = np.logical_and(full_graph[j, i] == -1, full_graph[i, j] == -1)

            # If this path is confirmed by LLM
            if known_effect.get(key) >= 0.95:
                # Compare with the initial graph: if the path doesn't exist then force this path
                if not exist_common_ij:
                    errors[key] = "Forced"

            # The causal effect is rejected by LLM
            if known_effect.get(key) <= 0.05:
                # Compare with the initial graph: if the path does exist then forbid this path
                if exist_common_ij:
                    errors[key] = "Forbidden"

        # Consider directed path
        else:
            # Indicator of existence of path i->j
            exist_ij = np.logical_and(full_graph[j, i] == 1, full_graph[i, j] == -1)

            # If this path is confirmed by LLM
            if known_effect.get(key) >= 0.95:
                # Compare with the initial graph: if the path doesn't exist then force this path
                if not exist_ij:
                    errors[key] = "Forced"

            # The causal effect is rejected by LLM
            if known_effect.get(key) <= 0.05:
                # Compare with the initial graph: if the path does exist then forbid this path
                if exist_ij:
                    errors[key] = "Forbidden"

    return errors



def graph_effect_prompts (column_names,
                          graph,
                          boot_probability,
                          boot_prob_common_cause):
    '''
    :param column_names: Column names.
    :param graph: causal graph
    :param boot_probability: bootstrap probability of directed edges, e.g., i -> j
    :param boot_prob_common_cause: bootstrap probability of common cause, e.g., i <-> j
    :return: A prompt describing relationships in the causal graph and corresponding bootstrap probability.
    '''

    m = graph.shape[0]
    effect_prompt = []

    for i in range(m):
        for j in range(i+1,m):
            if np.logical_and(graph[j, i] == 1, graph[i, j] == -1):
                effect_prompt.append(column_names[i] + "->" + column_names[j] +
                                     "(the bootstrap probability of such edge is" + boot_probability[i,j] + ")")

            if np.logical_and(graph[i, j] == 1, graph[j, i] == -1):
                effect_prompt.append(column_names[j] + "->" + column_names[i]+
                                     "(the bootstrap probability of such edge is" + boot_probability[j,i] + ")")

            if np.logical_and(graph[i, j] == -1, graph[j, i] == -1):
                effect_prompt.append(column_names[j] + "<->" + column_names[i]+
                                     "(the bootstrap probability of there exist a common cause is" + boot_prob_common_cause[i,j] + ")")

    graph_prompt = "All of the edges suggested by the causal discovery are below:\n" + "\n".join(effect_prompt)


    return graph_prompt





