class Reranker(object):
    # Kun Zhou Implemented
    def __init__(self, args):
        pass

    def forward(self, data, algo_candidates, statics_dict, knowledge_docs):
        '''

        :param data: Given Tabular Data in Pandas DataFrame format
        :param algo_candidates: A list containing all algorithm candidates
        :param statics_dict: A dict containing all necessary statics information
        :param knowledge_docs: A list containing all necessary domain knowledge information from GPT-4
        :return: A dict containing the selected algorithm and its hyperparameter settings
        '''
        algorithm_setup = {}
        return algorithm_setup