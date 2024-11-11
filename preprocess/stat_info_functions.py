from distutils.command.clean import clean

import numpy as np
import pandas as pd
import os
import random
import json
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.multitest import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import matplotlib.pyplot as plt


def data_preprocess (df: pd.DataFrame, ratio: float = 0.5, ts: bool = False):
    '''
    :param df: Dataset in Panda DataFrame format.
    :param ratio: threshold to remove column.
    :param ts: indicator of time-series data.
    :return: cleaned data, indicator of missingness in cleaned data, overall data type, data type of each feature.
    '''

    # Data clean
    missing_vals = [np.nan]
    missing_mask = df.isin(missing_vals)

    remove_index = []
    for column in missing_mask:
       if missing_mask[column].mean() > ratio:
           remove_index.extend(column)

    clean_df = df.drop(remove_index, axis=1)
    missing_mask_clean = missing_mask.drop(remove_index, axis=1)

    # Judge if missingness exists in the cleaned data
    if missing_mask_clean.sum().sum() > 0:
        miss_res = {"Missingness": True}
    else:
        miss_res = {"Missingness": False}

    # Data Type Classification
    column_type = {}
    overall_type = {}

    for column in clean_df.columns:

        dtype = clean_df[column].dtype

        if pd.api.types.is_numeric_dtype(dtype) and dtype != 'bool':
            column_type[column] = 'Continuous'
        else:
            column_type[column] = 'Category'
        # elif isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype) or dtype == 'bool' or dtype == 'int64':
        #     column_type[column] = 'Category'

    all_type = list(column_type.values())
    unique_type = list(set(all_type))

    if not ts:
        if len(unique_type) == 1:
            if unique_type[0] == "Continuous":
                overall_type["Data Type"] = "Continuous"
            elif unique_type[0] == "Category":
                overall_type["Data Type"] = "Category"
        else:
            overall_type["Data Type"] = "Mixture"

    if ts:
        overall_type["Data Type"] = "Time-series"

    # Convert category data to numeric data
    categorical_features = [key for key, value in column_type.items() if value == "Category"]

    for column in categorical_features:
        clean_df[column] = pd.Categorical(clean_df[column])
        clean_df[column] = clean_df[column].cat.codes.replace(-1, np.nan) # Keep NaN while converting    

    return clean_df, miss_res, column_type, overall_type

# clean_data, miss_res, each_type, dataset_type = data_preprocess(df = df, ratio = 0.5, ts = False)
# print(clean_data)

def imputation (df: pd.DataFrame, column_type: dict, ts: bool = False):
    '''
    :param df: cleaned and converted data in Pandas DataFrame format.
    :param column_type: data type of each column.
    :param ts: indicator of time-series data.
    :return: imputed data.
    '''

    categorical_features = [key for key, value in column_type.items() if value == "Category"]
    continuous_features = [key for key, value in column_type.items() if value == "Continuous"]

    if not ts:
        # Initialize imputer
        imputer_cat = SimpleImputer(strategy='most_frequent')
        imputer_cont = IterativeImputer(random_state=0)

        # Imputation for continuous data
        df[continuous_features] = imputer_cont.fit_transform(df[continuous_features])

        # Imputation for categorical data
        for column in categorical_features:
            df[column] = imputer_cat.fit_transform(df[[column]]).ravel()

    if ts:
        df = df.ffill()

    # Z-score normalization
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return scaled_df

# imputed_data = imputation(df = clean_data, column_type = each_type, ts = False)


def linearity_check (df: pd.DataFrame, num_test: int = 100, alpha: float = 0.1, path = None):
    '''
    :param df: imputed data in Pandas DataFrame format.
    :param num_test: maximum number of tests.
    :param alpha: significance level.
    :return: indicator of linearity, reset testing results for each pair, fitted OLS model.
    '''

    pval = []
    models = []
    m = df.shape[1]

    tot_pairs = m * (m - 1) / 2
    combinations_list = list(combinations(list(range(m)), 2))
    pair_num = min(int(tot_pairs), num_test)
    test_pairs = random.sample(combinations_list, pair_num)

    for i in range(pair_num):
        x = df.iloc[:, test_pairs[i][0]]
        x = sm.add_constant(x).to_numpy()

        y = df.iloc[:, test_pairs[i][1]].to_numpy()

        model = sm.OLS(y, x)
        results = model.fit()
        models.append((results, test_pairs[i]))

        # Ramsey’s RESET - H0: linearity is satisfied
        reset_test = linear_reset(results, power=2)
        pval.append(reset_test.pvalue)

    # Benjamini & Yekutieli procedure: return true for hypothesis that can be rejected for given alpha
    # Return True: reject H0 (p value < alpha) -- linearity is not satisfied
    corrected_result = multipletests(pval, alpha=alpha, method='fdr_by')[0]

    # Once there is one pair of test has been rejected, we conclude non-linearity
    if corrected_result.sum() == 0:
        check_result = {"Linearity": True}
        selected_models = models[:4]
    else:
        check_result = {"Linearity": False}
        # Select one of the non-linear pairs to plot residuals
        non_linear_indices = [i for i, result in enumerate(corrected_result) if result]
        linear_indices = [i for i, result in enumerate(corrected_result) if not result]
        num_nonlinear_pair = len(non_linear_indices)

        if num_nonlinear_pair >= 4:
            selected_models = [models[i] for i in non_linear_indices[:4]]
        else:
            selected_models = [models[i] for i in non_linear_indices[:num_nonlinear_pair]]
            selected_models.extend([models[i] for i in linear_indices[:(4 - num_nonlinear_pair)]])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Residual Plots for Selected Pair of Variables', fontsize=16)

    axs = axs.flatten()
    for idx, (selected_model, selected_pair) in enumerate(selected_models):
        predictions = selected_model.predict()
        residuals = selected_model.resid

        col_x_name = df.columns[selected_pair[0]]
        col_y_name = df.columns[selected_pair[1]]

        axs[idx].scatter(predictions, residuals)
        axs[idx].axhline(y=0, color='r', linestyle='--')
        axs[idx].set_xlabel('Predicted Values')
        axs[idx].set_ylabel('Residuals')
        axs[idx].set_title(f'{col_x_name} vs {col_y_name}')

    # Hide any unused subplots if less than 4 pairs were tested
    for idx in range(len(selected_models), 4):
        fig.delaxes(axs[idx])

    if not os.path.exists(path):
        os.makedirs(path)
    print(f"Saving residuals plot to {os.path.join(path, 'residuals_plot.jpg')}")
    fig.savefig(os.path.join(path, 'residuals_plot.jpg'))

    return check_result

# linearity_res = linearity_check(df = imputed_data, path = '/Users/fangnan/Library/CloudStorage/OneDrive-UCSanDiego/UCSD/ML Research/Causal Copilot/preprocess/stat_figures')
# print(linearity_res)


 # Gaussian error Checking
 #
 # Input: cleaned and transformed data & Linearity testing results
 # Output: testing results
def gaussian_check(df: pd.DataFrame, linearity, num_test: int = 100, alpha: float = 0.1, path=None):
    '''
    :param df: imputed data in Pandas DataFrame format.
    :param linearity: indicator of linearity.
    :param num_test: maximum number of tests.
    :param alpha: significance level.
    :return: indicator of gaussian errors.
    '''

    pval = []
    collect_result = []

    m = df.shape[1]
    tot_pairs = m * (m - 1) / 2
    combinations_list = list(combinations(list(range(m)), 2))
    pair_num = min(int(tot_pairs), num_test)
    test_pairs = random.sample(combinations_list, pair_num)

    for i in range(pair_num):
        if linearity:
            x = df.iloc[:, test_pairs[i][0]]
            x = sm.add_constant(x).to_numpy()

            y = df.iloc[:, test_pairs[i][1]].to_numpy()

            model = sm.OLS(y, x)
            results = model.fit()
            residuals = results.resid
            collect_result.append((residuals, test_pairs[i]))

        elif not linearity:
            x = df.iloc[:, test_pairs[i][0]].to_numpy()
            y = df.iloc[:, test_pairs[i][1]].to_numpy()

            # Fit Lowess
            smoothed = lowess(y, x)
            smoothed_x = smoothed[:, 0]
            smoothed_y = smoothed[:, 1]
            smoothed_values = np.interp(x, smoothed_x, smoothed_y)
            residuals = y - smoothed_values

            collect_result.append((residuals, test_pairs[i]))

        # Shapiro-Wilk test - H0: Gaussian errors
        test = stats.shapiro(residuals)
        pval.append(test.pvalue)

        # Benjamini & Yekutieli procedure - True: reject H0 -- Gaussian error assumption is not satisfied
        corrected_result = multipletests(pval, alpha=alpha, method='fdr_by')[0]

        if corrected_result.sum() == 0:
            check_result = {"Gaussian Error": True}
            selected_results = collect_result[:4]
        else:
            check_result = {"Gaussian Error": False}
            non_gaussain_indices = [i for i, result in enumerate(corrected_result) if result]
            gaussian_indices = [i for i, result in enumerate(corrected_result) if not result]
            num_nongaussian_pair = len(non_gaussain_indices)

            if num_nongaussian_pair >= 4:
                selected_results = [collect_result[i] for i in non_gaussain_indices[:4]]
            else:
                selected_results = [collect_result[i] for i in non_gaussain_indices[:num_nongaussian_pair]]
                selected_results.extend([collect_result[i] for i in gaussian_indices[:(4 - num_nongaussian_pair)]])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Q-Q Plots for Selected Pair of Variables', fontsize=16)
    axs = axs.flatten()
    for idx, (selected_results, selected_pair) in enumerate(selected_results):
        res = selected_results

        col_x_name = df.columns[selected_pair[0]]
        col_y_name = df.columns[selected_pair[1]]

        sm.qqplot(res, line='45', ax=axs[idx])
        axs[idx].set_title(f'{col_x_name} vs {col_y_name}')

    # Hide any unused subplots if less than 4 pairs were tested
    for idx in range(len(selected_results), 4):
        fig.delaxes(axs[idx])

    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path, 'qq_plot.jpg'))

    return check_result

# gaussian_res = gaussian_check(df = imputed_data, linearity = linearity_res, path='/Users/fangnan/Library/CloudStorage/OneDrive-UCSanDiego/UCSD/ML Research/Causal Copilot/preprocess/stat_figures')
#
# print(gaussian_res)

def heterogeneity_check(df: pd.DataFrame, heterogeneity_indicator: str = "domain_index"):
    '''
    :param df: imputed data in Pandas DataFrame format.
    :param test_pairs: maximum number of pairs to be tested.
    :param alpha: significance level.
    :return: indicator of heteroscedasticity.
    '''
    # check if there are multiple domain index
    if heterogeneity_indicator in df.columns:
        if df[heterogeneity_indicator].nunique() > 1:
            return True
    return False


def stationary_check(df: pd.DataFrame, max_test: int = 1000, alpha: float = 0.1):
    '''
    :param df: imputed data in Pandas DataFrame format.
    :param max_test: maximum number of test.
    :param alpha: significance level.
    :return: indicator of stationary.
    '''

    ADF_pval = []
    m = df.shape[1]

    if m > max_test:
        num_test = max_test
        index = random.sample(range(m), num_test)
    else:
        num_test = m
        index = range(num_test)

    for i in range(num_test):
        x = df.iloc[:, index[i]].to_numpy()

        # ADF test - H0: this series is non-stationary
        adf_test = adfuller(x)
        ADF_pval.append(adf_test[1])

        # Bonferroni correction
        # True: reject H0 -- this series is stationary
        # False: accept H0 -- this series is non-stationary
        corrected_ADF = multipletests(ADF_pval, alpha=alpha, method='bonferroni')[0]

        if corrected_ADF.sum() == m:
            check_result = {"Stationary": True}
            continue
        else:
            check_result = {"Stationary": False}
            break

    return check_result


def stat_info_collection(global_state):
    '''
    :param data: given tabular data in pandas dataFrame format.
    :param global_state: GlobalState object to update and use.
    :return: updated GlobalState object.
    '''
    data = global_state.user_data.raw_data
    n, m = data.shape

    # already exacted in the user query function
    # if args.domain_index in data.columns:
    #     m = m - 1

    # Update global state
    global_state.statistics.sample_size = n
    global_state.statistics.feature_number = m

    if global_state.statistics.heterogeneous and global_state.statistics.domain_index is not None:
        # Drop the domain index column from the data
        domain_index = global_state.statistics.domain_index
        col_domain_index = data[domain_index]
        data = data.drop(columns=[domain_index])
    else:
        col_domain_index = None

    # Data pre-processing
    clean_data, miss_res, each_type, dataset_type = data_preprocess(df = data, ratio=global_state.statistics.ratio, ts=False)

    # Update global state
    global_state.statistics.missingness = miss_res['Missingness']
    global_state.statistics.data_type = dataset_type["Data Type"]

    # Imputation
    if global_state.statistics.missingness:
        imputed_data = imputation(df=clean_data, column_type=each_type, ts=False)
    else:
        imputed_data = clean_data

    # Check assumption for continuous data
    if global_state.statistics.data_type == "Continuous":
        if global_state.statistics.linearity is None:
            # Linearity assumption checking
            linearity_res = linearity_check(df=imputed_data, num_test=global_state.statistics.num_test, alpha=global_state.statistics.alpha,
                                            path=global_state.user_data.output_graph_dir)
            # Update global state
            global_state.statistics.linearity = linearity_res["Linearity"]

        if global_state.statistics.gaussian_error is None:
            # Gaussian error checking
            gaussian_res = gaussian_check(df=imputed_data, linearity=global_state.statistics.linearity, num_test=global_state.statistics.num_test, alpha=global_state.statistics.alpha,
                                          path=global_state.user_data.output_graph_dir)
            # Update global state
            global_state.statistics.gaussian_error = gaussian_res["Gaussian Error"]

    else:
        global_state.statistics.linearity = False
        global_state.statistics.gaussian_error = False

    # Assumption checking for time-series data
    # if args.ts:
    #     global_state.statistics.linearity = False
    #     global_state.statistics.gaussian_error = False
    #     stationary_res = stationary_check(df=imputed_data, max_test=args.num_test, alpha=args.alpha)
    #     global_state.statistics.stationary = stationary_res["Stationary"]

    # merge the domain index column back to the data if it exists
    if col_domain_index is not None:
        imputed_data['domain_index'] = col_domain_index

    global_state.user_data.processed_data = imputed_data

    # Convert statistics to JSON for compatibility with existing code
    # stat_info_json = json.dumps(vars(global_state.statistics), indent=4)

    return global_state



def convert_stat_info_to_text(statistics):
    """
    Convert the statistical information from Statistics object to natural language.
    
    :param statistics: Statistics object containing statistical information about the dataset.
    :return: A string describing the dataset characteristics in natural language.
    """
    text = f"The dataset has the following characteristics:\n\n"
    text += f"Data Type: The overall data type is {statistics.data_type}.\n\n"
    text += f"The sample size is {statistics.sample_size} with {statistics.feature_number} features. "
    text += f"This dataset is {'time-series' if statistics.data_type == 'Time-series' else 'not time-series'} data. "
    text += f"Data Quality: {'There are' if statistics.missingness else 'There are no'} missing values in the dataset.\n\n"
    
    text += "Statistical Properties:\n"
    text += f"- Linearity: The relationships between variables {'are' if statistics.linearity else 'are not'} predominantly linear.\n"
    text += f"- Gaussian Errors: The errors in the data {'do' if statistics.gaussian_error else 'do not'} follow a Gaussian distribution.\n"
    text += f"- Heterogeneity: The dataset {'is' if statistics.heterogeneous else 'is not'} heterogeneous. \n\n"

    if statistics.domain_index is not None:
        text += f"If the data is heterogeneous, the column/variable {statistics.domain_index} is the domain index indicating the heterogeneity. "
        text += f"If the data is not heterogeneous, then the existed domain index is constant.\n\n"
    else:
        text += "\n\n"
        
    return text

