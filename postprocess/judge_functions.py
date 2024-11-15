import numpy as np
from sympy.stats.rv import probability
import ast
import json 
from openai import OpenAI
from postprocess.visualization import Visualization


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
    converted_graph, info, raw_result = algo_func(hyperparameters).fit(boot_sample)

    if algorithm == 'PC':
        boot_graph = raw_result.G.graph
    elif algorithm == 'FCI':
        boot_graph = raw_result[0].graph
    elif algorithm == "GES":
        boot_graph = raw_result['G'].graph
    elif algorithm == 'CDNOD':
        boot_graph = raw_result.G.graph
    else:
        boot_graph = converted_graph

    return boot_graph

def bootstrap_probability(boot_result, algorithm):

    m = boot_result.shape[1]

    certain_edges_prob = np.zeros((m,m))  # -> and use converted graph
    uncertain_edges_prob = np.zeros((m,m))  # -
    bi_edges_prob = np.zeros((m,m))  # <->
    half_edges_prob = np.zeros((m,m))  # o->
    none_edges_prob = np.zeros((m,m))  # o-o
    none_exist_prob = np.zeros((m,m)) # did not exist edge

    for i in range(m):
        for j in range(m):
            if i == j: continue
            else:
                elements_ij = boot_result[:, i, j]
                elements_ji = boot_result[:, j, i]

                if algorithm in ['PC','GES','CDNOD','FCI']:
                    # j -> i
                    certain_edges_prob[i, j] = np.mean((elements_ij == 1) & (elements_ji == -1))
                    # i - j
                    uncertain_edges_prob[i, j] = np.mean((elements_ij == -1) & (elements_ji == -1))
                    # i <-> j
                    bi_edges_prob[i, j] = np.mean((elements_ij == 1) & (elements_ji == 1))
                else:
                    # j -> i
                    certain_edges_prob[i, j] = np.mean(elements_ij == 1)

                # no existence of edge
                none_exist_prob[i, j] = np.mean(elements_ij == 0)

                if algorithm == 'FCI':
                    # j o-> i
                    half_edges_prob[i, j] = np.mean((elements_ij == 1) & (elements_ji == 2))
                    # i o-o j
                    none_edges_prob[i, j] = np.mean((elements_ij == 2) & (elements_ji == 2))


    edges_prob = np.stack((certain_edges_prob, uncertain_edges_prob, bi_edges_prob, half_edges_prob, none_edges_prob, none_exist_prob), axis=0)

    return edges_prob



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

    from multiprocessing import Pool

    m = data.shape[1]
    errors = {}
    raw_graph = full_graph

    # try:
    #     if algorithm == 'PC':
    #         raw_graph = full_graph.raw_result.G.graph
    #     elif algorithm == 'FCI':
    #         raw_graph = full_graph.raw_result[0].graph
    #     elif algorithm == "GES":
    #         raw_graph = full_graph.raw_result['G'].graph
    #     elif algorithm == 'CDNOD':
    #         raw_graph = full_graph.raw_result.G.graph
    #     else:
    #         raw_graph = full_graph.converted_graph
    # except:
    #     if algorithm == 'PC':
    #         raw_graph = full_graph.G.graph
    #     elif algorithm == 'FCI':
    #         raw_graph = full_graph[0].graph
    #     elif algorithm == "GES":
    #         raw_graph = full_graph['G'].graph
    #     elif algorithm == 'CDNOD':
    #         raw_graph = full_graph.G.graph
    #     else:
    #         raw_graph = full_graph

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

    # Each layer of edges_prob represents:
    # 0. certain_edges_prob: ->
    # 1. uncertain_edges_prob: -
    # 2. bi_edges_prob: <->
    # 3. half_edges_prob: o->
    # 4. none_edges_prob: o-o
    # 5. none_exist_prob: x
    edges_prob = bootstrap_probability(boot_effect_save_array, algorithm)

    recommend = ['->', '-', '<->', 'o->', 'o-o', 'Forbid']

    boot_recommend = {}

    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            else:
                element_ij = raw_graph[i, j]
                element_ji = raw_graph[j, i]
                prob_ij = edges_prob[:, i, j]

                if algorithm in ['PC','GES','CDNOD','FCI']:
                    certain_edge_raw = (element_ij == 1 and element_ji == -1)
                    uncertain_edge_raw = (element_ij == -1 and element_ji == -1)
                    bi_edge_raw = (element_ij == 1 and element_ji == 1)
                    non_exist_raw = (element_ij == 0 or element_ji == 0)

                    cond0 = certain_edge_raw and (edges_prob[0, i, j] < 0.05)  # j -> i
                    cond1 = uncertain_edge_raw and (edges_prob[1, i, j] < 0.05)  # j - i
                    cond2 = bi_edge_raw and (edges_prob[2, i, j] < 0.05)  # j <-> i
                    cond5 = non_exist_raw and (edges_prob[5, i, j] < 0.05)  # j x i

                    if algorithm == 'FCI':
                        half_edge_raw = (element_ij == 1 and element_ji == 2)
                        none_edge_raw = (element_ij == 2 and element_ji == 1)

                        cond3 = half_edge_raw and (edges_prob[3, i, j] < 0.05)  # j o-> i
                        cond4 = none_edge_raw and (edges_prob[4, i, j] < 0.05)  # j o-o i
                else:
                    certain_edge_raw = (element_ij == 1)
                    non_exist_raw = (element_ij == 0)

                    cond0 = certain_edge_raw and (edges_prob[0, i, j] < 0.05)  # j -> i
                    cond5 = non_exist_raw and (edges_prob[5, i, j] < 0.05)  # j x i

                # Bootstrap probability is less than 0.05
                if algorithm in ['PC', 'GES', 'CDNOD']:
                    if cond0 or cond1 or cond2 or cond5:
                        boot_recommend[str(j) + '-' + str(i)] = recommend[np.argmax(prob_ij)] + '(' + str(np.max(prob_ij)) + ')'
                elif algorithm == 'FCI':
                    if cond0 or cond1 or cond2 or cond3 or cond4 or cond5:
                        boot_recommend[str(j) + '-' + str(i)] = recommend[np.argmax(prob_ij)] + '(' + str(np.max(prob_ij)) + ')'
                else:
                    if cond0 or cond5:
                        boot_recommend[str(j) + '-' + str(i)] = recommend[np.argmax(prob_ij)] + '(' + str(np.max(prob_ij)) + ')'

                # Bootstrap probability is greater than 0.95
                if (not certain_edge_raw) and (edges_prob[0, i, j] > 0.95):
                    boot_recommend[str(j) + '-' + str(i)] = '->' + '(' + str(edges_prob[0, i, j]) + ')'
                elif (not non_exist_raw) and (edges_prob[5, i, j] > 0.95):
                    boot_recommend[str(j) + '-' + str(i)] = 'Forbid' + '(' + str(edges_prob[5, i, j]) + ')'

                if algorithm in ['PC', 'GES', 'CDNOD','FCI']:
                    if (not uncertain_edge_raw) and (edges_prob[1, i, j] > 0.95):
                        boot_recommend[str(j) + '-' + str(i)] = '-' + '(' + str(edges_prob[1, i, j]) + ')'
                    elif (not bi_edge_raw) and (edges_prob[2, i, j] > 0.95):
                        boot_recommend[str(j) + '-' + str(i)] = '<->' + '(' + str(edges_prob[2, i, j]) + ')'

                    if algorithm == 'FCI':
                        if (not half_edge_raw) and (edges_prob[3, i, j] > 0.95):
                            boot_recommend[str(j) + '-' + str(i)] = 'o->' + '(' + str(edges_prob[3, i, j]) + ')'
                        elif (not none_edge_raw) and (edges_prob[4, i, j] > 0.95):
                            boot_recommend[str(j) + '-' + str(i)] = 'o-o' + '(' + str(edges_prob[4, i, j]) + ')'

    # Convert edges_prob to a dict
    boot_edges_prob = {'certain_edges': edges_prob[0,:,:],
                       'uncertain_edges': None,
                       'bi_edges': None,
                       'half_edges': None,
                       'non_edges': None,
                       'non_existence':edges_prob[5, :, :]}

    if algorithm in ['PC','GES','CDNOD','FCI']:
        boot_edges_prob['uncertain_edges'] = edges_prob[1,:,:]
        boot_edges_prob['bi_edges'] = edges_prob[2,:,:]
        if algorithm == 'FCI':
            boot_edges_prob['half_edges'] = edges_prob[3,:,:]
            boot_edges_prob['non_edges'] = edges_prob[4,:,:]

    return boot_recommend, boot_edges_prob


def get_json(args, prompt):
        client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
        response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
        jsons = response.choices[0].message.content
        jsons_cleaned = jsons.replace('```json', '').replace('```', '').strip()
        try:
            jsons_cleaned = json.loads(jsons_cleaned)
        except:
            print('The returned LLM JSON file is wrong, try again')
            jsons_cleaned = get_json(args,prompt)
        return jsons_cleaned

def call_llm(args, prompt):
        client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
        response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in Causal Discovery."},
                    {"role": "user", "content": prompt}
                ]
            )
        contents = response.choices[0].message.content
        try:
            result, explanation = contents.split(':')[0].strip().upper(), contents.split(':')[1].strip()
        except:
            print('The returned LLM evaluation response is wrong, try again')
            result, explanation = call_llm(args,prompt)
        return result, explanation

def llm_evaluation(data, args, edges_dict, boot_edges_prob):
    """
    Here we let LLM double check the result of initial graph, and make edition (determine direction & delete edge)
    Provided Info:
    1. Structure of whole graph
    (ancestor decendant ... relationship?)
    2. Bootstrap Probability
    3. Pair relationship in the original graph
    Return: Result with domain knowledge and explanation
    """
    direct_dict = {}
    forbid_dict = {}
    #print('boot_edges_prob',boot_edges_prob)
    relation_text_dict, relation_text = edges_to_relationship(data, edges_dict, boot_edges_prob)
    #print('relation_text_dict:',relation_text_dict)
    #print('edges_dict: ', edges_dict)

    for type in relation_text_dict.keys():
        pair_idx = 0
        for pair_relation in relation_text_dict[type]:
            var_j = edges_dict[type][pair_idx][0]
            var_i = edges_dict[type][pair_idx][1]
            idx_j = data.columns.get_loc(var_j)
            idx_i = data.columns.get_loc(var_i)
            prompt_pruning = f"""
        We want to carry out causal discovery analysis, considering these variables: {data.columns.tolist()}. 
        First, we have conducted the statistical causal discovery algorithm to find the following causal relationships from a statistical perspective:
        {relation_text}
        According to the results shown above, it has been determined that {pair_relation}, but it may not be correct. 
        Then, your task is to double check this result from a domain knowledge perspective and determine whether this statistically suggested hypothesis is plausible in the context of the domain.  

        Firstly, determine the relationship between {var_j} and {var_i}
        If {var_j} causes {var_i}, the "result" is 'A'. If {var_i} causes {var_j}, the "result" should be 'B'. If you are really sure there is no relationship, the "result" is 'C'. 
        Secondly, please provide an explanation of your result, leveraging your expert knowledge on the causal relationship between {var_j} and {var_i}, please use only one to two sentences. 
        Your response should consider the relevant factors and provide a reasoned explanation based on your understanding of the domain.

        Response me following the template below. Do not include anything else. explanations should only include one to two sentences
        A or B or C: explanations 

        Here are some response Examples
        A: Larger abalones, which are indicated by greater shell lengths, generally possess heavier shells. Therefore, there is a direct positive relationship between length and shell weight as a larger size typically results in greater overall mass of the shell.
        """
            #print('prompt_pruning:', prompt_pruning)
            pair_idx += 1
            result, explanation = call_llm(args,prompt_pruning)
            #print((var_j, var_i), result)
            if result == 'A':
                if (var_j, var_i) not in edges_dict['certain_edges']:
                    direct_dict[(idx_j, idx_i)] = ((var_j, var_i), explanation)
            elif result == 'B':
                if (var_i, var_j) not in edges_dict['certain_edges']:
                    direct_dict[(idx_i, idx_j)] = ((var_i, var_j), explanation)
            elif result == 'C':
                forbid_dict[(idx_j, idx_i)] = ((var_j, var_i), explanation)
    return direct_dict, forbid_dict

def edges_to_relationship(data, edges_dict, boot_edges_prob=None):
    '''
    :param data: Pandas DataFrame format.
    :param graph: An adjacent matrix in Numpy Ndarray format - Matrix[i,j] = 1 indicates j->i
    :param boot_probability: A matrix in Numpy Ndarray format
                             recording bootstrap probability of directed edges,
                             e.g., Matrix[i,j] records probability of existence of edge i -> j.
    :param edges_dict: A dict containing lists of all types of relationships
    :param boot_prob: A dict containing probability matrix of all types of edges
    :return: A dictionary of lists describing different edge types' relationships for each node pairs and corresponding bootstrap probability.
    '''
    relation_dict = {
            'certain_edges': 'causes',
            'uncertain_edges': 'has undirected relationship with',
            'bi_edges': 'has hidden confounder with',
            'half_edges': 'is not a descendant of',
            'none_edges': 'has no D-seperation set with'
        }
    result_dict = {
            'certain_edges': [],
            'uncertain_edges': [],
            'bi_edges': [],
            'half_edges': [],
            'none_edges': []
        }
    summary_dict = {
            'certain_edges': 'These variable pairs have certain directed edge between them: \n',
            'uncertain_edges': 'These variable pairs have undirected relationship between them: \n',
            'bi_edges': 'These variable pairs have hidden confounders between them: \n',
            'half_edges': 'These variable pairs have non-descendant relationship between them: \n',
            'none_edges': 'These variable pairs have no D-seperation between them: \n'
        }
    for edge_type in relation_dict.keys():
        edges_list = edges_dict[edge_type]
        for edges in edges_list:
            if boot_edges_prob is not None:
                idx_j = data.columns.get_loc(edges[0])
                idx_i = data.columns.get_loc(edges[1])
                prob = boot_edges_prob[edge_type][idx_i, idx_j]
                result_dict[edge_type].append(f'{edges[0]} {relation_dict[edge_type]} {edges[1]} with bootstrap probability {prob}')
            else:
                result_dict[edge_type].append(f'{edges[0]} {relation_dict[edge_type]} {edges[1]}')
    
    filtered_result_dict = {key: value for key, value in result_dict.items() if value}

    relation_text = ""
    for key in filtered_result_dict:
        relation_text += f"{summary_dict[key]}"
        for pair_relation in filtered_result_dict[key]:
            relation_text += f'{pair_relation}, '
        relation_text += '\n'
    
    return filtered_result_dict, relation_text

def llm_direction(global_state, args, revised_graph, voting_num=10, threshold=0.7):
    '''
    :param data: Given Tabular Data in Pandas DataFrame format
    :param full_graph: An adjacent matrix in Numpy Ndarray format -
                       causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
    :param gpt_setup: Contain information of configurations of GPT-4
    :param knowledge_docs: A doc containing all necessary domain knowledge information from GPT-4.
    :return: obvious errors based on LLM, e.g. {"X->Y: "Forced", "Y->Z: "Forbidden"}
    '''

    data = global_state.user_data.raw_data
    variables = data.columns

    if global_state.algorithm.selected_algorithm not in ['PC', 'CDNOD', 'GES', 'FCI']:
        return {}, revised_graph

    knowledge_docs = global_state.user_data.knowledge_docs

    my_visual = Visualization(global_state)
    edges_dict = my_visual.convert_to_edges(revised_graph)
    uncertain_edges = edges_dict['uncertain_edges']
    # bi_edges = edges_dict['bi_edges']
    # half_edges = edges_dict['half_edges']
    # none_edges = edges_dict['none_edges']
    # Let GPT determine direction of uncertain edges
    #print(uncertain_edges)
    if len(uncertain_edges) == 0:
        print('Empty uncertain_edges')
        return {}, revised_graph
    
    prompt_direction = f"""
    I have a list of tuples where each tuple represents a pair of entities that have a relationship with each other. 
    For each tuple, please determine the causal relationship between the two entities, indicating which entity causes the other. 
    You can reference this background knowledge: {knowledge_docs}
    1. Use each tuple as the key of JSON, for example, if the tuple is ('Raf', 'Mek'), then use ('Raf', 'Mek') as a key
    2. If the first entity causes the second, the value is 'A'. If the second entity causes the first, the value should be 'B'. If cannot decide, the value is 'C'. The order is very important
    3. The direction can only be 'A' or 'B' or 'C', do not reply other things
    Here is the list of tuples: {uncertain_edges}
    Return me a json following the template below. Do not include anything else.
    JSON format:
    {{
        "tuple1": "A" or "B" or "C",    
        "tuple2": "A" or "B" or "C",   
        ......   
    }}
    """
    prob_mat = np.zeros((global_state.statistics.feature_number, global_state.statistics.feature_number))
    for i in range(voting_num):
        json_directions = get_json(args, prompt_direction)
        for key in json_directions.keys():
            try:
                tuple = ast.literal_eval(key)
                direction = json_directions[key]
                i = variables.get_loc(tuple[0])
                j = variables.get_loc(tuple[1])
                if direction == 'A':
                    prob_mat[j, i] += 1
                elif direction == 'B':
                    prob_mat[i, j] += 1
            except:
                print('parse tuple error:', key)
                continue
    
    prob_mat /= voting_num
    #print(prob_mat)
    revise_indice = np.where(prob_mat>threshold)
    edges_list = []
    for i, j in zip(revise_indice[0], revise_indice[1]):
        revised_graph[i, j] = 1
        revised_graph[j, i] = -1
        edges_list.append((variables[j], variables[i]))
    print('edges list:', edges_list)
    prompt_justification = f"""
    I have a list of tuples where each tuple represents a pair of entities that have a relationship with each other. 
    For example, ('Raf', 'Mek') means 'Raf' causes 'Mek'.
    1. Provide a brief justification for each tuple's causal relationship based on the following background knowledge {knowledge_docs}.
    2. Follow this JSON format, you should impute justifications for each tuple as the value
    3. Only Return me a json that can be loaded directly. Do not include anything else.
    JSON format:
    {{
    
    """
    for item in edges_list:
        prompt_justification += f"""
                                "{item}": justification here,

                                """
    prompt_justification += f"""
                             }}
                            """
    
    json_justification = get_json(args, prompt_justification)
    json_justification_filtered = {}
    for pair in edges_list:
        try:
            json_justification_filtered[str(pair)] = json_justification[str(pair)]
        except:
            json_justification_filtered[str(pair)] = 'There is no justification for it.'
    
    return json_justification_filtered, revised_graph
        


def llm_direction_evaluation(global_state):
    from postprocess.visualization import Visualization
    
    true_graph = global_state.user_data.ground_truth
    if global_state.algorithm.selected_algorithm in ['DirectLiNGAM', 'ICALiNGAM', 'NOTEARS']:
        full_graph = global_state.results.converted_graph
    else:
        full_graph =  global_state.results.raw_result
    revised_graph = global_state.results.revised_graph
    my_visual = Visualization(global_state)
    edges_dict = my_visual.convert_to_edges(full_graph)
    uncertain_edges = edges_dict['uncertain_edges']
    variables = global_state.user_data.raw_data.columns

    denom = 0
    nom = 0
    for item in uncertain_edges:
        i = variables.get_loc(item[0])
        j = variables.get_loc(item[1])
        if true_graph[i, j]==1 or true_graph[j, i]==1:
            denom += 1
        if true_graph[i, j]==revised_graph[i, j] or true_graph[j, i]==revised_graph[j, i]:
            nom += 1
    
    all_edges = sum(true_graph.flatten())
    exist_uncertain_edges = denom
    if denom>0:
        correct_rate = nom/denom
    else:
        correct_rate = 'NA'

    print(f"""
        all_edges: {all_edges},
        exist_uncertain_edges: {exist_uncertain_edges},
        correct_rate: {correct_rate}
          """)

    




    
