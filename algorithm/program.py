class Programming(object):
    # Kun Zhou Implemented
    def __init__(self, args):
        pass

    def code_synthesis(self, data, algorithm_setup):
        '''

        :param data: Given Tabular Data in Pandas DataFrame format
        :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
        :return: executable programs based on Causal-Learn Toolkit
        '''
        algorithm_setup = {}
        return algorithm_setup

    def execute(self, program):
        '''
        :param program: Causal-Learn Toolkit program
        :return: A dict containing Executed Results
        '''
        results = {}
        return results

    def forward(self, data, algorithm_setup):
        '''

        :param data: Given Tabular Data in Pandas DataFrame format
        :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
        :return: executable programs based on Causal-Learn Toolkit
        '''
        program = self.code_synthesis(data, algorithm_setup)
        results = self.execute(program)
        return results