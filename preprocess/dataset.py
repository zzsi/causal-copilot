def load_data(path):
    # Xinyue Wang Implemented
    '''
    :param path: str for data path or filename
    :return: pandas dataframe
    '''
    import pandas as pd
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    file_extension = os.path.splitext(path)[1].lower()

    if file_extension == '.csv':
        data = pd.read_csv(path)
    elif file_extension in ['.xls', '.xlsx']:
        data = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return data

def statics_info(args, data):
    # Fang Nan Implemented
    '''
    :param args: a class containing pre-specified information - an indicator of time-series data,
                 missing ratio for data cleaning, 
                 significance level (default 0.1),
                 maximum number of tests (default 1000).
    :param data: Given Tabular Data in Pandas DataFrame format
    :return: A dict containing all necessary statics information
    '''
    from stat_info_functions import *

    statics_dict = stat_info_collection(args = para, data = data)
    
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
