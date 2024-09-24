import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
import random
import math


def bootstrap(data,
              full_graph,
              ts: bool = False, boot_num: int = 500):
    '''
    :param data: Given Tabular Data in Pandas DataFrame format
    :param full_graph: Causal graph using the full dataset
    :param ts: indicator of time-series data
    :param boot_num: number of bootstrap iterations
    :return: a dict of obvious errors in causal analysis results based on bootstrap,
             e.g. {"X->Y: "Forced", "Y->Z: "Forbidden"}
    '''

    n, m = data.shape
    errors = {}

    boot_effect_save = np.empty((m, m, boot_num))
    boot_probability = np.empty((m, m))

    for boot_time in range(boot_num):

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

            # Use PC as an example: will be replaced with functions in code generation part
            cg_boot = pc(boot_sample.to_numpy())
            boot_effect_save[:, :, boot_time] = cg_boot.G.graph

        for i in range(m):
            for j in range(i + 1, m):
                # Only focus on edges whose direction can be determined: i->j or j->i

                # What to do with common cause: i<->j ?

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

    return errors


def llm_evaluation(data, full_graph, known_effect):
    '''
    :param data: Given Tabular Data in Pandas DataFrame format
    :param full_graph: Causal graph using the full dataset
    :param known_effect: A dict containing known causal effects between variables from GPT-4,
                      GPT-4 should give how confidence it has about such effect
                      e.g., {"X->Y": 95%, "Y->Z": 5%}
    :return: obvious errors in causal analysis results based on LLM,
             e.g. {"X->Y: "Forced", "Y->Z: "Forbidden"}
    '''

    errors = {}

    for key in known_effect.keys():
        split_key = key.split("->")
        i = data.columns.get_loc(split_key[0])
        j = data.columns.get_loc(split_key[1])

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


def graph_effect_prompts (column_names, graph):
    '''
    :param column_names: Column names.
    :param graph: causal graph
    :return: A prompt describing edges in the causal graph.
    '''

    m = graph.shape[0]
    effect_prompt = []

    for i in range(m):
        for j in range(i+1,m):
            if graph[i, j] == -1:
                effect_prompt.append(column_names[i] + "->" + column_names[j])

            elif graph[j, i] == -1:
                effect_prompt.append(column_names[j] + "->" + column_names[i])

    graph_prompt = "All of the edges suggested by the causal discovery are below:\n" + "\n".join(effect_prompt)


    return graph_prompt





