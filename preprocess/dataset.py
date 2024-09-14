def load_data(path):
    # Xinyue Wang Implemented
    '''
    :param path: str for data path or filename
    :return: pandas dataframe
    '''
    data = []
    return data

def statics_info(args, data):
    # Fang Nan Implemented
    '''
    :param args: configurations
    :param data: Given Tabular Data in Pandas DataFrame format
    :return: A dict containing all necessary statics information
    '''
    statics_dict = {}
    return statics_dict


def knowledge_info(args, data):
    # Kun Zhou Implemented
    '''
    :param args: configurations
    :param data: Given Tabular Data in Pandas DataFrame format
    :return: A list containing all necessary domain knowledge information from GPT-4
    '''
    from openai import OpenAI
    client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
    table_name = args.data_file
    table_columns = '\t'.join(data.columns._data)
    prompt = ("I will conduct causal discovery on the Tabular Dataset %s containing the following Columns: \n\n"
              "%s\n\nPlease list the following information with clear format and accurate expression:"
              "\n1.Detailed Explanation about the Variables (columns);"
              "\n2.Possible Causal Relations among these variables;"
              "\n3.Other Background Domain Knowledge that may be helpful for experts to design causal discovery algorithms")%(table_name, table_columns)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    knowledge_doc = response.choices[0].message.content
    knowledge_docs = [knowledge_doc]
    return knowledge_docs