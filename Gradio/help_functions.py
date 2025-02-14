import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback
from preprocess.stat_info_functions import *
from openai import OpenAI
from pydantic import BaseModel
import torch 

def try_numeric(value):
    """Convert string to int or float if possible, otherwise return string"""
    try:
        # Try converting to float
        return float(value)
    except ValueError:
        # Return original string if both conversions fail
        return value.strip()

def generate_hyperparameter_text(global_state):
    hyperparameter_text = ""
    for param in global_state.algorithm.algorithm_arguments.keys():
        details = global_state.algorithm.algorithm_arguments_json['hyperparameters'][param]
        value = details['value']
        explanation = details['explanation']
        hyperparameter_text += f"  Parameter: {param}\n"
        hyperparameter_text += f"  Value: {value}\n"
        hyperparameter_text += f"  Explanation: {explanation}\n\n"
    return hyperparameter_text, global_state

def LLM_parse_query(format, prompt, message, args):
    client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
    if format:
        completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        response_format=format,
        )
        parsed_response = completion.choices[0].message.parsed
    else: 
        completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        )
        parsed_response = completion.choices[0].message.content
    return parsed_response

# process functions
def sample_size_check(n_row, n_col, chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE):
    ## Few sample case: give warning
    if 1<= n_row/n_col < 5:
        chat_history.append((None, "Sample Size Check Summary: \n"\
                             "⚠️ The dataset provided do not have enough sample size and may result in unreliable analysis. \n"
                                "Please upload a larger dataset if you mind that."))
        CURRENT_STAGE = 'reupload_dataset'
    ## Not enough sample case: must reupload
    elif n_row/n_col < 1:
        chat_history.append((None, "Sample Size Check Summary: \n"\
                             "⚠️ The sample size of dataset provided is less than its feature size. We are not able to conduct further analysis. Please provide more samples. \n"))
        CURRENT_STAGE = 'reupload_dataset'
    ## Enough sample case
    else:
        chat_history.append((None, "Sample Size Check Summary: \n"\
                             "✅ The sample size is enough for the following analysis. \n"))
        CURRENT_STAGE = 'important_feature_selection'
    return chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE

def parse_reupload_query(message, chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE):
    print('reupload query:', message)
    if message == 'continue':
        chat_history.append((message, "📈 Continue the analysis..."))
        CURRENT_STAGE = 'important_feature_selection'
    else:
        #REQUIRED_INFO['data_uploaded'] = False
        CURRENT_STAGE = 'initial_process'
    return chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE


def process_initial_query(message, chat_history, download_btn, args, REQUIRED_INFO, CURRENT_STAGE):
    # TODO: check if the initial query is valid or satisfies the requirements
    print('initial query:', message)
    # algorithm 
    # 
    if 'yes' in message.lower():
        args.data_mode = 'real'
        REQUIRED_INFO['initial_query'] = True
        chat_history.append((message, None))
    elif 'no' in message.lower():
        args.data_mode = 'simulated'
        REQUIRED_INFO['initial_query'] = True
        chat_history.append((message, None))
    else:
        print('not feature indicator')
        chat_history.append((message,
                                """Please enter your initial query first before proceeding. 
                             Please indicate if your dataset has meaningful feature names using 'YES' or 'NO', 
                             and you can also provide some information about the background/context/prior/statistical information about the dataset,
                             which would help us generate appropriate report for you.
                             """))
    return chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE, args    

def parse_mode_query(message, chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE):
    chat_history.append((message, None))
    message = message.strip()
    if message.lower() == 'yes':
        REQUIRED_INFO["interactive_mode"] = True
        CURRENT_STAGE = 'sparsity_check'
        chat_history.append(("✅ Run with Interactive Mode...", None))
    elif message.lower() == 'no' or message == '':
        REQUIRED_INFO["interactive_mode"] = False
        CURRENT_STAGE = 'sparsity_check_2'
        chat_history.append(("✅ Run with Non-Interactive Mode...", None))
    else: 
        chat_history.append((None, "❌ Invalid input, please try again!"))
    return chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE

def parse_var_selection_query(message, chat_history, download_btn, next_step, args, global_state, REQUIRED_INFO, CURRENT_STAGE):
    class VarList(BaseModel):
        variables: list[str]
    prompt = "You are a helpful assistant, please extract variable names as a list. \n"
    "If there is only one variable, also save it in list variables"
    f"Variables must be among this list! {global_state.user_data.raw_data.columns}"
    "variables in the returned list MUST be among the list above, and it's CASE SENSITIVE."
    "If you cannot find variable names, just return an empty list."
    parsed_vars = LLM_parse_query(VarList, prompt, message, args)
    var_list = parsed_vars.variables
    if var_list == []:
        chat_history.append((message, "❌ Your variable selection query cannot be parsed, please make sure variables are among your dataset features and retry. \n"
                             ))
        return var_list, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
    else:
        missing_vars = [var for var in var_list if var not in global_state.user_data.raw_data.columns and var!='']
        if missing_vars != []:
            chat_history.append((message, "❌ Variables " + ", ".join(missing_vars) + " are not in the dataset, please check it and retry.\n"
                                 "Note that it's CASE SENSITIVE."))
            return var_list, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
        elif len(var_list) > 20:
            chat_history.append((message, "❌ Number of chosen Variables should be within 20, please check it and retry."))
            return var_list, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
        else:
            chat_history.append((message, "✅ Successfully parsed your provided variables."))
            CURRENT_STAGE = next_step
            return var_list, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
        
def parse_ts_query(message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE):
    message = message.strip()
    if message.lower() == 'no':
        global_state.statistics.time_series = False
        CURRENT_STAGE = 'ts_check_done'
    elif message == 'continue' or message == '':
        CURRENT_STAGE = 'ts_check_done'
        global_state.statistics.time_series = True
    else:
        try:
            time_lag = int(message)
            global_state.statistics.time_lag = time_lag
            chat_history.append((None, f"✅ We successfully set your time lag to be {time_lag}."))
            CURRENT_STAGE = 'ts_check_done'
            global_state.statistics.time_series = True
        except: 
            chat_history.append((None, f"❌ We cannot parse your query, please follow the template and retry."))
    return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def parse_sparsity_query(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE):
    # Select features based on LLM
    if message.upper() == 'LLM' or message == '':
        try:
            global_state = llm_select_dropped_features(global_state=global_state, args=args)
        except:
            global_state = llm_select_dropped_features(global_state=global_state, args=args)
        if message.upper() == 'LLM':
            chat_history.append((None, "The following sparse variables suggested by LLM will be dropped: \n"
                                            ", ".join(global_state.user_data.llm_drop_features)))
        elif message == '':
            chat_history.append((None, "You do not choose any variables to drop, we will drop the following variables suggested by LLM: \n"
                                            ", ".join(global_state.user_data.llm_drop_features)))
        global_state = drop_greater_miss_between_30_50_feature(global_state)
        CURRENT_STAGE = "sparsity_drop_done"
        #var_list = [var for var in global_state.user_data.llm_drop_features if var in global_state.user_data.raw_data.columns]
    # Select features based on user query
    else:
        class VarList(BaseModel):
            variables: list[str]
        prompt = "You are a helpful assistant, please extract variable names as a list. \n"
        "If there is only one variable, also save it in list variables"
        f"Variables must be among this list! {global_state.user_data.raw_data.columns}"
        "variables in the returned list MUST be among the list above, and it's CASE SENSITIVE."
        "If you cannot find variable names, just return an empty list."
        parsed_vars = LLM_parse_query(VarList, prompt, message, args)
        var_list = parsed_vars.variables
        if var_list == []:
            chat_history.append((None, "⚠️ Your sparse variable dropping query cannot be parsed, Please follow the templete below and retry. \n"
                                            "Templete: PKA, Jnk, PIP2, PIP3, Mek"))
        else:
            missing_vars = [var for var in var_list if var not in global_state.user_data.raw_data.columns]
            if missing_vars != []:
                chat_history.append((None, "❌ Variables " + ", ".join(missing_vars) + " are not in the dataset, please check it and retry."))
            else:
                chat_history.append((None, "✅ Successfully parsed your provided variables. These sparse variables you provided will be dropped."))
                global_state.user_data.user_drop_features = var_list
                global_state = drop_greater_miss_between_30_50_feature(global_state)
                CURRENT_STAGE = "sparsity_drop_done"
    return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def first_stage_sparsity_check(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE):
    class NA_Indicator(BaseModel):
                indicator: bool
                na_indicator: str
    prompt = """You are a helpful assistant, please do the following tasks based on the provided context:
    **Context**
    We ask the user: We do not detect NA values in your dataset, do you have the specific value that represents NA? If so, please provide here. Otherwise please input 'NO'.
    Now we need to parse the user's input.
    **Task**
    Firstly, identify whether user answer 'no' or something like that, and save the boolean result in indicator. 
    If user answers 'no' or something like that, the boolean should be **True!**
    If the user provide the na_indicator, the boolean should be **False!**
    Secondly if user provide the na_indicator, identify the indicator user specified in the query, and save the string result in na_indicator. """
    parsed_response = LLM_parse_query(NA_Indicator, prompt, message, args)
    indicator, na_indicator = parsed_response.indicator, parsed_response.na_indicator
    print(indicator, na_indicator)
    if indicator:
        global_state.user_data.nan_indicator = None
        CURRENT_STAGE = 'sparsity_check_2'
    else:
        global_state.user_data.nan_indicator = na_indicator
        global_state, nan_detect = numeric_str_nan_detect(global_state)
        if nan_detect:
            CURRENT_STAGE = 'sparsity_check_2'
        else:
            chat_history.append((None, "❌ We cannot find the NA value you specified in the dataset, please retry!"))
    return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def parse_algo_query(message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE):
    if message == '' or message.lower()=='no':
        chat_history.append((message, "💬 No algorithm is specified, will go to the next step..."))
        CURRENT_STAGE = 'report_generation_check'      
    elif message not in ['PC', 'FCI', 'CDNOD', 'GES', 'DirectLiNGAM', 'ICALiNGAM', 'NOTEARS', 'FGES', 'XGES', 'AcceleratedDirectLiNGAM']:
        if torch.cuda.is_available():
            chat_history.append((message, "❌ The specified algorithm is not correct, please choose from the following: \n"
                                        "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS\n"
                                        "Fast Version: FGES, XGES, AcceleratedDirectLiNGAM"))   
        else:    
            chat_history.append((message, "❌ The specified algorithm is not correct, please choose from the following: \n"
                                        "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS\n"
                                        "Fast Version: FGES, XGES."))   
    else:  
        global_state.algorithm.selected_algorithm = message
        chat_history.append((message, f"✅ We will rerun the Causal Discovery Procedure with the Selected algorithm: {global_state.algorithm.selected_algorithm}\n"
                                       "Please press 'enter' in the chatbox to start the running..." ))
        CURRENT_STAGE = 'algo_selection'
        #process_message(message, chat_history, download_btn)
    return message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def parse_hyperparameter_query(message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE):
            if message.lower()=='no' or message=='':
                CURRENT_STAGE = 'algo_running'    
                hyperparameter_text, global_state = generate_hyperparameter_text(global_state) 
                chat_history.append((None, f"✅ We will run the Causal Discovery Procedure with the Selected parameters: \n {hyperparameter_text}\n"))
            else:
                try:
                    specified_params = {line.split(':')[0].strip(): line.split(':')[1].strip() for line in message.strip().split('\n')}
                    print('specified_params',specified_params)
                    original_params = global_state.algorithm.algorithm_arguments
                    print('original_params',original_params)
                    common_keys = original_params.keys() & specified_params.keys()
                    if len(common_keys)==0:
                        print(1)
                        chat_history.append((None, "❌ The specified parameters are not correct, please follow the template!"))
                        return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

                    for key in common_keys:
                        global_state.algorithm.algorithm_arguments_json['hyperparameters'][key]['value'] = try_numeric(specified_params[key])
                        global_state.algorithm.algorithm_arguments_json['hyperparameters'][key]['explanation'] = 'User specified'
                        global_state.algorithm.algorithm_arguments[key] = try_numeric(specified_params[key])
                    print(global_state.algorithm.algorithm_arguments)
                    hyperparameter_text, global_state = generate_hyperparameter_text(global_state) 
                    chat_history.append((None, f"✅ We will run the Causal Discovery Procedure with the Specified parameters: \n"
                                         f"{hyperparameter_text}"))
                    CURRENT_STAGE = 'algo_running' 
                except Exception as e:
                    print(e)
                    print(str(e))
                    traceback.print_exc()
                    chat_history.append((None, "❌ The specified parameters are not correct, please follow the template!"))
            return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def parse_user_postprocess(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE):
    edges_dict = {
        "add_edges": [],
        "forbid_edges": [],
        "orient_edges": []
    }
    print('message:', message)
    try:
        if message == '' or not ('Add Edges' in message or 'Forbid Edges' in message or 'Orient Edges' in message):
             CURRENT_STAGE = 'retry_algo'
             chat_history.append((None, "💬 No valid query is provided, will go to the next step."))
             return edges_dict, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
        else:
            class EditList(BaseModel):
                    add_edges: list[str]
                    forbid_edges: list[str]
                    orient_edges: list[str]
            prompt = f""" You are a helpful assistant, please do the following tasks:
            **Tasks**
            1. Extract node relationships in Add Edges, Forbid Edges, and Orient Edges 
            2. For each relationship, save the node pairs as a list of tuples in add_edges, forbid_edges, and orient_edges respectively.
            For example, Add Edges: Age->Height; Age->Shell Weight; should be save as [(Age, Height), (Age, Shell Weight)] in add_edges.
            3. Add Edges, Forbid Edges, and Orient Edges may not all exist. If there's no relationship, just return an empty list.
            4. All node names must be among this list! {global_state.user_data.raw_data.columns}
            **Example**
            Add Edges: Age->Height
            Forbid Edges: Length->Height
            Orient Edges: Age->Diameter
            add_edges = [(Age, Height)]
            forbid_edges = [(Length, Height)]
            orient_edges = [(Age, Diameter)]
            """
            parsed_response = LLM_parse_query(EditList, prompt, message, args)
            add_edges, forbid_edges, orient_edges = parsed_response.add_edges, parsed_response.forbid_edges, parsed_response.orient_edges
            edges_dict["add_edges"] = [(pair.split('->')[0].strip(' '), pair.split('->')[1].strip(' ')) for pair in add_edges]
            edges_dict["forbid_edges"] = [(pair.split('->')[0].strip(' '), pair.split('->')[1].strip(' ')) for pair in forbid_edges]
            edges_dict["orient_edges"] = [(pair.split('->')[0].strip(' '), pair.split('->')[1].strip(' ')) for pair in orient_edges]
            # Check whether all these variables exist
            variables = [item for sublist in edges_dict.values() for pair in sublist for item in pair]
            missing_vars = [var for var in variables if var not in global_state.user_data.raw_data.columns]
            print(edges_dict)
            print(variables)
            print(global_state.user_data.raw_data.columns)
            if missing_vars != []:
                chat_history.append((None, "❌ Variables " + ", ".join(missing_vars) + " are not in the dataset, please check it and retry."))
                return edges_dict, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
            CURRENT_STAGE = 'postprocess_parse_done'
            return edges_dict, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
    except Exception as e:
        chat_history.append((None, "❌ Your query cannot be parsed, please follow the templete and retry"))
        print(str(e))
        traceback.print_exc()
        return edges_dict, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def parse_report_algo_query(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE):
    class AlgoList(BaseModel):
        algorithms: list[str]
    algos = global_state.logging.global_state_logging
    prompt = f"""You are a helpful assistant, please do the following tasks:
    **Tasks**
    1. Extract the algorithm name as a list in algorithms.
    2. If there's no algorithm, just return an empty list.
    3. You can only choose from the following algorithms! {', '.join(algos)}
    """
    parsed_response = LLM_parse_query(AlgoList, prompt, message, args)
    algo_list = parsed_response.algorithms
    if algo_list == []:
        chat_history.append((message, "❌ Your algorithm query cannot be parsed, please choose from the following algorithms!\n"
                             f"{', '.join(algos)}"))
    elif len(algo_list) > 1:
        chat_history.append((message, "⚠️ You can only choose one algorithm at a time, please retry!"))
    else:
        chat_history.append((message, "✅ Successfully parsed your provided algorithm."))
        global_state.results.report_selected_index = algos.index(algo_list[0])
        CURRENT_STAGE = 'report_generation'
    return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def parse_inference_query(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE):
    chat_history.append((message, None))
    message = message.strip()
    if message.lower() == 'no' or message == '':
        chat_history.append((None, "✅ No need for downstream analysis, continue to the next section..."))
        CURRENT_STAGE = 'report_generation_check'
        return None, None, None, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
    else:
        class InfList(BaseModel):
                tasks: list[str]
                reason: str
                descriptions: list[str]
                key_node: list[str]
        columns = global_state.user_data.processed_data.columns
        with open('causal_analysis/context/query_prompt.txt', 'r') as file:
            query_prompt = file.read()
            query_prompt = query_prompt.replace('[COLUMNS]', f",".join(columns))
        
        global_state.logging.downstream_discuss.append({"role": "user", "content": message})
        parsed_response = LLM_parse_query(InfList, query_prompt, message, args)
        reason, tasks_list, descs_list, key_node_list = parsed_response.reason, parsed_response.tasks, parsed_response.descriptions, parsed_response.key_node
        print(tasks_list, descs_list, key_node_list)
        chat_history.append((None, "✅ Successfully parsed your query. We will analyze it in the following perspectives:\n"
                                    f"{', '.join(tasks_list)}\n"))
        return reason, tasks_list, descs_list, key_node_list, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE


def parse_inf_discuss_query(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE):
    chat_history.append((message, None))
    global_state.logging.downstream_discuss.append({"role": "user", "content": message})
    message = message.strip()
    if message.lower() == 'no' or message == '':
        print('go to report_generation')
        chat_history.append((None, "✅ No need for downstream analysis, continue to the next section..."))
        CURRENT_STAGE = 'report_generation_check'
    else:
        class DiscussList(BaseModel):
                    answer: str
        prompt = f"""You are a helpful assistant, here is the previous conversation history for your reference:
                ** Conversation History **
                {global_state.logging.downstream_discuss}
                ** Your Task **
                Answer user's question based on the given history in bullet points.
                Your answer must be based on the given history, DO NOT include any fake information.
                """
        global_state.logging.downstream_discuss.append({"role": "user", "content": message})
        parsed_response = LLM_parse_query(DiscussList, prompt, message, args)
        answer_info = parsed_response.answer 
        global_state.inference.task_info[global_state.inference.task_index]['result']['discussion'][message] = answer_info
        print(answer_info)
    
        chat_history.append((None, answer_info))
        global_state.logging.downstream_discuss.append({"role": "system", "content": answer_info})
        chat_history.append((None, "Do you have questions about this analysis?  Please describe your questions.\n"
                                    "You can also input 'NO' to end this discussion."))
    return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE            
                

def parse_treatment(desc, global_state, args):
    prompt = f"""
    I'm doing the Treatment Effect Estimation analysis, please identify the Treatment Variable in this description:
    {desc}
    The variable name must be among these variables: {global_state.user_data.processed_data.columns}
    Only return me with the variable name, do not include anything else.
    """
    treatment = LLM_parse_query(None, 'You are an expert in Causal Discovery.', prompt, args)
    return treatment

def parse_shift_value(desc, args):
    class ShiftValue(BaseModel):
        shift_value: float
    parsed_response = LLM_parse_query(ShiftValue, "Extract the numerical value from the query and save it in shift_value", desc, args)
    shift_value = parsed_response.shift_value
    return shift_value
