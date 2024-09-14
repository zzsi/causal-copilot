class Judge(object):
    # Fang Nan Implemented
    def __init__(self, args):
        pass

    def quality_judge(self, data, results, statics_dict, algorithm_setup, knowledge_docs):
        '''

        :param data: Given Tabular Data in Pandas DataFrame format
        :param results: A dict containing all algorithm candidates
        :param statics_dict: A dict containing all necessary statics information
        :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
        :param knowledge_docs: A list containing all necessary domain knowledge information from GPT-4
        :return: obvious errors in causal analysis results
        '''
        errors = []
        return errors

    def report_generation(self, data, results, statics_dict, algorithm_setup, knowledge_docs):
        '''
        generate and save the report
        :param data: Given Tabular Data in Pandas DataFrame format
        :param results: A dict containing all algorithm candidates
        :param statics_dict: A dict containing all necessary statics information
        :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
        :param knowledge_docs: A list containing all necessary domain knowledge information from GPT-4
        :return: Str: A technique report explaining all the results for readers
        '''
        report = ''
        save()
        return report

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