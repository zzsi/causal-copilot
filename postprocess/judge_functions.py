def bootstrap(data, full_graph, algorithm, hyperparameters, boot_num, ts):
    '''
    :param data: Given Tabular Data in Pandas DataFrame format
    :param full_graph: An adjacent matrix in Numpy Ndarray format -
                       causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
    :param algorithm: String representing the algorithm name
    :param hyperparameters: Dictionary of hyperparameter names and values
    :param boot_num: Number of bootstrap iterations
    :param ts: An indicator of time-series data
    :return: a dict of obvious errors in causal analysis results based on bootstrap,
             e.g. {"X->Y: "Forced", "Y->Z: "Forbidden"};
             a matrix records bootstrap probability of directed edges, Matrix[i,j] records the
             bootstrap probability of the existence of edge j -> i.
    '''

    import numpy as np
    import pandas as pd
    import random
    import math
    import algorithm.wrappers as wrappers

    n, m = data.shape
    errors = {}

    boot_effect_save = [] # Save graphs based on bootstrapping

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

        # Get the algorithm function from wrappers
        algo_func = getattr(wrappers, algorithm)
        # Execute the algorithm with data and hyperparameters
        boot_graph, info, raw_result = algo_func(hyperparameters).fit(boot_sample)

        boot_effect_save.append(boot_graph)

    boot_effect_save_array = np.array(boot_effect_save)
    boot_probability = np.mean(boot_effect_save_array, axis=0)


    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            else:
                # Only consider directed edge: i->j
                # Indicator of existence of path i->j in full graph
                exist_ij = (full_graph[j, i] == 1)

                # Force the path if the probability is greater than 0.95
                if boot_probability[j, i] >= 0.95:
                    # Compare with the initial graph: if the path doesn't exist then force it
                    if not exist_ij:
                        errors[data.columns[i] + "->" + data.columns[j]] = "Forced"

                # Forbid the path if the probability is less than 0.05
                elif boot_probability[j, i] <= 0.05:
                    # Compare with the initial graph: if the path exist then forbid it
                    if exist_ij:
                        errors[data.columns[i] + "->" + data.columns[j]] = "Forbidden"

    return errors, boot_probability


def llm_evaluation(data, full_graph, args, knowledge_docs):
    '''
    :param data: Given Tabular Data in Pandas DataFrame format
    :param full_graph: An adjacent matrix in Numpy Ndarray format -
                       causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
    :param gpt_setup: Contain information of configurations of GPT-4
    :param knowledge_docs: A doc containing all necessary domain knowledge information from GPT-4.
    :return: obvious errors based on LLM, e.g. {"X->Y: "Forced", "Y->Z: "Forbidden"}
    '''

    import ast
    from openai import OpenAI

    client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)

    table_columns = '\t'.join(data.columns)

    prompt = (f"Based on information provided by the knowledge document: {knowledge_docs} \n\n,"
              f"conclude the causal relationship between each pair of variables as shown among the column names: {table_columns}, "
              "and tell me how much confidence you have about such causal relationship. "
              "The output of your response should be in dict format, which can be detected by python. "
              "For example, if you have 95% confidence to conclude that X cause Y, and only 5% confidence that Y causes Z,"
              "you should output {'X->Y': 0.95, 'Y->Z': 0.05}. "
              "For the format of dict you give, make sure obey the follow rules: \n\n"
              "1. Just give me the output in a dict format, do not provide other information! \n\n"
              "2. The feature name within keys should always be the same as the column names I provided!\n\n"
              "3. Always use directed right arrow like '->' and DO NOT use left arrow '->' in the dict. For example, if you want to express X causes Y, always use 'X->Y' and DO NOT use 'Y<-X.")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # The output from GPT is str
    known_effect = response.choices[0].message.content

    known_effect_cleaned = known_effect.replace('```python', '').replace('```', '').strip()

    # Convert to dict format
    known_effect_dict = ast.literal_eval(known_effect_cleaned)

    errors = {}

    for key in known_effect_dict.keys():
        # Consider directed path
        if "->" in key:
            split_key = key.split("->")
            i = data.columns.get_loc(split_key[0])
            j = data.columns.get_loc(split_key[1])

            # Indicator of existence of path i->j
            exist_ij = (full_graph[j, i] == 1)

            # If this path is confirmed by LLM
            if known_effect_dict.get(key) >= 0.95:
                # Compare with the initial graph: if the path doesn't exist then force this path
                if not exist_ij:
                    errors[key] = "Forced"

            # The causal effect is rejected by LLM
            if known_effect_dict.get(key) <= 0.05:
                # Compare with the initial graph: if the path does exist then forbid this path
                if exist_ij:
                    errors[key] = "Forbidden"

    conversation = {
        "prompt": prompt,
        "response": known_effect
    }

    return conversation, errors



def graph_effect_prompts (data, graph, boot_probability):
    '''
    :param data: Pandas DataFrame format.
    :param graph: An adjacent matrix in Numpy Ndarray format - Matrix[i,j] = 1 indicates j->i
    :param boot_probability: A matrix in Numpy Ndarray format
                             recording bootstrap probability of directed edges,
                             e.g., Matrix[i,j] records probability of existence of edge i -> j.
    :return: A prompt describing relationships in the causal graph and corresponding bootstrap probability.
    '''

    m = data.shape[1]
    column_names = data.columns

    effect_prompt = []

    for i in range(m):
        for j in range(m):
            if graph[j, i] == 1:
                effect_prompt.append(str(column_names[i]) + "->" + str(column_names[j]) +
                                     " (the bootstrap probability of such edge is " + str(boot_probability[i, j]) + ")")

    graph_prompt = "All of the edges suggested by the causal discovery are below:\n" + "\n".join(effect_prompt)

    return graph_prompt
