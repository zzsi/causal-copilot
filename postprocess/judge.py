class Judge(object):
    # Fang Nan Implemented
    def __init__(self, args):
        pass

    def quality_judge(self, data,
                  results,
                  algorithm_setup, known_effect, ts: bool = False):
    '''
    :param data: Given Tabular Data in Pandas DataFrame format
    :param results: A dict containing all algorithm candidates (Need to use bootstrap using all candidate algorithm?)
    :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
    :param known_effect: A dict containing known causal effects between variables from GPT-4,
                      GPT-4 should give how confidence it has about such effect
                      e.g., {"X->Y": 95%, "Y->Z": 5%}
    :param ts: indicator of time-series data
    :return: obvious errors in causal analysis results
    '''

    from judge_functions import *

    # Run algorithm (using algorithm_setup) with the full dataset
    # Use PC as an example: will be replaced with functions in code generation part
    cg = pc(data.to_numpy())
    full_graph = cg.G.graph

    # Statistics Perspective: Bootstrapping to get probability of edges using the selected algorithm.
    errors_stat = bootstrap(data=data, full_graph=full_graph, ts=ts, boot_num=500)

    # LLM perspective: errors based on domain knowledge from GPT-4
    errors_llm = llm_evaluation(data=data, full_graph=full_graph, known_effect=knowledge_docs)

    # Combine error obtained from both statistics and LLM perspectives
    errors = {**errors_stat, **errors_llm}

    return errors

    

    def report_generation(self, llm_setup,
                      data_name, data,
                      graph,
                      results, statics_dict, algorithm_setup, knowledge_docs):
    '''
    generate and save the report
    :param llm_setup: information of configurations of GPT-4
    :param data_name: name of the data file.
    :param data: Given Tabular Data in Pandas DataFrame format
    :param graph: graph generated using causal discovery algorithm
    :param results: A dict containing all algorithm candidates?
    :param statics_dict: A dict containing all necessary statics information
    :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
    :param knowledge_docs: A list containing all necessary domain knowledge information from GPT-4
    :return: Str: A technique report explaining all the results for readers
    '''

    # Generate prompts based on all the information and results
    from openai import OpenAI
    client = OpenAI(organization=llm_setup.organization, project=llm_setup.project, api_key=llm_setup.apikey)

    # Data information
    table_columns = '\t'.join(data.columns)

    # Data property prompt
    if statics_dict.get("Stationary") == "non time-series":
        missing = np.where(statics_dict.get("Missingness") == True, "has missing values,",
                           "does not have missing values,")
        data_type = "is" + statics_dict.get("Data Type") + "data,"
        linear = np.where(statics_dict.get("Linearity") == True, "satisfies the linearity assumption,",
                          "violates the linearity assumption,")
        gaussian = np.where(statics_dict.get("Gaussian Error") == True, ",and satisfies the Gaussian error assumption",
                            ",and violates the Gaussian error assumption")
        data_prop_prompt = missing + data_type + linear + gaussian
    else:
        data_prop_prompt = "is" + np.where(statics_dict.get("Stationary") == True, "stationary",
                                           "non-stationary") + "time-series data"

    # Graph prompt
    graph_prompt = graph_effect_prompts(data.columns, graph)

    prompt = ("I have conducted causal discovery on the Tabular Dataset %s containing the following Columns: \n\n"
              "%s\n\n. "
              "In terms of the property of the dataset, it %s.\n\n"
              "Based on the property of the dataset mentioned above, we think %s is suitable for "
              "running causal discovery on this dataset. \n\n"
              "%s"
              "Based on the context and knowledge %s, and all of the results and information above, "
              "write a comprehensive technical report, the report should be easily readable and understandable to the audience who may not"
              "be familiar with causal discovery.") % (
                 data_name,
                 data_prop_prompt,
                 algorithm_setup,
                 graph_prompt,
                 knowledge_docs)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    report_doc = response.choices[0].message.content

    return report_doc

    def forward(self, data, results, statics_dict, algorithm_setup, knowledge_docs):
        '''

        :param data: Given Tabular Data in Pandas DataFrame format
        :param results: A dict containing all algorithm candidates
        :param statics_dict: A dict containing all necessary statics information
        :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
        :param knowledge_docs: A list containing all necessary domain knowledge information from GPT-4
        :return: A dict containing the revised algorithm and its hyperparameter settings
        '''
        errors = self.quality_judge(data, results, statics_dict, algorithm_setup, knowledge_docs)
        if len(errors) == 0:
            return True, algorithm_setup
        else:
            new_algorithm_setup = {}
            return False, new_algorithm_setup
