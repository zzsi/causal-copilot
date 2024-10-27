def bootstrap_iteration(data, ts, algorithm, hyperparameters):
    '''
    :param data: Given Tabular Data in Pandas DataFrame format
    :param ts: Indicator of time-series
    :param algorithm: String representing the algorithm name
    :param hyperparameters: Dictionary of hyperparameter names and values
    :return: Bootstrap result of one iteration
    '''
    import random
    import math
    import pandas as pd
    import algorithm.wrappers as wrappers

    n = data.shape[0]

    # Choose bootstrap method based on the ts flag
    if not ts:
        # General bootstrapping
        boot_index = random.choices(range(n), k=n)
        boot_sample = data.iloc[boot_index, :]
    else:
        # Moving block bootstrapping for time-series
        block_size = 10
        block_num = math.ceil(n / block_size)
        block_start = random.sample(range(n - block_size + 1), block_num)

        blocks = [list(range(start, start + block_size)) for start in block_start]
        subsets = [data.iloc[block] for block in blocks]

        boot_sample = pd.concat(subsets, ignore_index=True).iloc[0:n]

    # Get the algorithm function from wrappers
    algo_func = getattr(wrappers, algorithm)
    # Execute the algorithm with data and hyperparameters
    boot_graph, info, raw_result = algo_func(hyperparameters).fit(boot_sample)
    return boot_graph


def bootstrap(data, full_graph, algorithm, hyperparameters, boot_num, ts, parallel):
    '''
    :param data: Given Tabular Data in Pandas DataFrame format
    :param full_graph: An adjacent matrix in Numpy Ndarray format -
                       causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
    :param algorithm: String representing the algorithm name
    :param hyperparameters: Dictionary of hyperparameter names and values
    :param boot_num: Number of bootstrap iterations
    :param ts: An indicator of time-series data
    :param parallel: indicator of parallel computing
    :return: a dict of obvious errors in causal analysis results based on bootstrap,
             e.g. {"X->Y: "Forced", "Y->Z: "Forbidden"};
             a matrix records bootstrap probability of directed edges, Matrix[i,j] records the
             bootstrap probability of the existence of edge j -> i.
    '''

    import numpy as np
    from multiprocessing import Pool

    m = data.shape[1]
    errors = {}

    boot_effect_save = []  # Save graphs based on bootstrapping

    if not parallel:
        for boot_time in range(boot_num):
            boot_graph = bootstrap_iteration(data, ts, algorithm, hyperparameters)
            boot_effect_save.append(boot_graph)

    if parallel:
        pool = Pool()

        # Prepare arguments for each process
        args = [(data, ts, algorithm, hyperparameters) for _ in range(boot_num)]
        boot_effect_save = pool.starmap(bootstrap_iteration, args)

        pool.close()
        pool.join()

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
            try:
                i = data.columns.get_loc(split_key[0])
                j = data.columns.get_loc(split_key[1])
            except:
                continue

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

def llm_direction(global_state, args):
    '''
    :param data: Given Tabular Data in Pandas DataFrame format
    :param full_graph: An adjacent matrix in Numpy Ndarray format -
                       causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
    :param gpt_setup: Contain information of configurations of GPT-4
    :param knowledge_docs: A doc containing all necessary domain knowledge information from GPT-4.
    :return: obvious errors based on LLM, e.g. {"X->Y: "Forced", "Y->Z: "Forbidden"}
    '''

    import ast
    import json 
    from openai import OpenAI
    from postprocess.visualization import Visualization

    data = global_state.user_data.raw_data
    variables = data.columns
    table_columns = '\t'.join(data.columns)
    full_graph = global_state.results.raw_result

    if global_state.algorithm.selected_algorithm == 'PC':
        revised_graph = global_state.results.raw_result.G.graph.copy()
    elif global_state.algorithm.selected_algorithm == 'CDNOD':
        revised_graph = global_state.results.raw_result.G.graph.copy()
    elif global_state.algorithm.selected_algorithm == 'GES':
        revised_graph = global_state.results.raw_result['G'].graph.copy()
    elif global_state.algorithm.selected_algorithm == 'FCI':
        revised_graph = global_state.results.raw_result[0].graph.copy()
    else:
        return {}, full_graph

    knowledge_docs = global_state.user_data.knowledge_docs

    my_visual = Visualization(global_state, args)
    edges_dict = my_visual.convert_to_edges(full_graph)
    uncertain_edges = edges_dict['uncertain_edges']
    # bi_edges = edges_dict['bi_edges']
    # half_edges = edges_dict['half_edges']
    # none_edges = edges_dict['none_edges']
    # Let GPT determine direction of uncertain edges
    print(uncertain_edges)
    prompt = f"""
    I have a list of tuples where each tuple represents a pair of entities that have a relationship with each other. 
    For each tuple, please determine the causal relationship between the two entities, indicating which entity causes the other. 
    1. Use each tuple as the key of JSON, for example, if the tuple is ('Raf', 'Mek'), then use ('Raf', 'Mek') as a key
    1. If the first entity causes the second, the direction is 'right'. If the second entity causes the first, the direction should be 'left'. The order is very important
    2. The direction can only be 'right' or 'left', do not reply other things
    3. Provide a brief justification for your decision based on the following background knowledge {knowledge_docs}.
    Here is the list of tuples: {uncertain_edges}
    Return me a json following the template below. Do not include anything else.
    JSON format:
    {{
        "tuple": {{
            "direction": 'right' or 'left',
            "justification": 'Your reasoning here.'
        }},
        ...
    }}
    """
    client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    # The output from GPT is str
    directions = response.choices[0].message.content
    directions_cleaned = directions.replace('```json', '').replace('```', '').strip()
    try:
        json_directions = json.loads(directions_cleaned)
    except:
        print('The returned LLM Direction JSON file is wrong')
        return {}, revised_graph
    for key in json_directions.keys():
        tuple = ast.literal_eval(key)
        direction = json_directions[key]['direction']
        try:
            i = variables.get_loc(tuple[0])
            j = variables.get_loc(tuple[1])
            if direction.lower() == 'right':
                revised_graph[j, i] = 1
            elif direction.lower() == 'left':
                revised_graph[i, j] = 1
        except:
            print(tuple)
            continue

    return json_directions, revised_graph
        
        

    
