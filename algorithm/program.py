import json


class Programming(object):
    # Kun Zhou Implemented
    def __init__(self, args):
        self.args = args
        self.algo2example_code = json.load(args.algo2example_file)

    def extract(self, output, start_str, end_str):
        if start_str in output and end_str in output:
            try:
                algo = output.split(start_str)[1].split(end_str)[0]
            except:
                algo = ''
            return algo
        else:
            return ''

    def code_synthesis(self, data, algorithm, algorithm_setup):
        '''

        :param data: Given Tabular Data in Pandas DataFrame format
        :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
        :return: executable programs based on Causal-Learn Toolkit
        '''
        from openai import OpenAI
        client = OpenAI(organization=self.args.organization, project=self.args.project, api_key=self.args.apikey)
        algo_example = self.algo2example_code[algorithm]
        prompt = (("Please write the code for using %s algorithm based on the Causal-Learn Toolkit to process the data %s, an example is below:\n\n"
                  "%s\n\nHere, we set up its hyperparameters as below:\n\n"
                  "%s\n\nBased on the examples and hyperparameters, please generate the executable python code in the ending of the output, using the following template <Code>executable code</Code>")
                  % (algorithm, self.args.data_file, algo_example, algorithm_setup))
        code = ''
        while code == '':
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            output = response.choices[0].message.content
            code = self.extract(output, "<Code>", "</Code>")
        return code

    def execute(self, program):
        '''
        :param program: Causal-Learn Toolkit program
        :return: A dict containing Executed Results
        '''
        results = {}
        try:
            global cg
            exec(program)
            # todo what is the executed results?
            results = cg
        except:
            results = {}
        return results

    def forward(self, data, algorithm, algorithm_setup):
        '''

        :param data: Given Tabular Data in Pandas DataFrame format
        :param algorithm: Selected algorithm
        :param algorithm_setup: A dict containing the selected algorithm and its hyperparameter settings
        :return: executable programs based on Causal-Learn Toolkit
        '''
        results = {}
        while results == {}:
            program = self.code_synthesis(data, algorithm, algorithm_setup)
            results = self.execute(program)
        return results