import numpy as np
import pandas as pd
import random
import json
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from itertools import combinations
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.multitest import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller


def data_preprocess (df: pd.DataFrame, ratio: float = 0.5, ts: bool = False):
    '''
    :param df: raw data.
    :param ratio: threshold to remove column.
    :param ts: indicator of time-series data.
    :return: cleaned data, missingness indicator of cleaned data, overall data type, data type of each column.
    '''

    """ Data clean """
    missing_vals = [np.nan]
    missing_mask = df.isin(missing_vals)

    remove_index = []
    for column in missing_mask:
       if missing_mask[column].mean() > ratio:
           remove_index.extend(column)

    clean_df = df.drop(remove_index, axis=1)
    missing_mask_clean = missing_mask.drop(remove_index, axis=1)

    """Judge if missingness exist in the cleaned data"""
    if missing_mask_clean.sum().sum() > 0:
        miss_res = {"Missingness": True}
    else:
        miss_res = {"Missingness": False}

    """Data Type Classification"""
    column_type = {}
    overall_type = {"Continuous": False, "Category": False, "Mixture": False, "Time-series": False}

    for column in clean_df.columns:

        dtype = clean_df[column].dtype

        if pd.api.types.is_numeric_dtype(dtype) and dtype != 'bool':
            column_type[column] = 'Continuous'
        elif isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype) or dtype == 'bool':
            column_type[column] = 'Category'

    all_type = list(column_type.values())
    unique_type = list(set(all_type))

    if not ts:
        if len(unique_type) == 1:
            if unique_type[0] == "Continuous":
                overall_type["Continuous"] = True
            elif unique_type[0] == "Category":
                overall_type["Category"] = True
        else:
            overall_type["Mixture"] = True

    if ts:
        overall_type["Time-series"] = True

    """Convert category data to numeric data"""
    categorical_features = [key for key, value in column_type.items() if value == "Category"]

    for column in categorical_features:
        clean_df[column] = pd.Categorical(clean_df[column])
        clean_df[column] = clean_df[column].cat.codes.replace(-1, np.nan) # Keep NaN while converting


    return clean_df, miss_res, column_type, overall_type

clean_data, miss_res, each_type, dataset_type = data_preprocess(df = data, ratio = 0.5, ts = False)


def imputation (df: pd.DataFrame, column_type: dict, ts: bool = False):
    '''
    :param df: cleaned and converted data.
    :param column_type: data type of each column.
    :param ts: indicator of time-series data.
    :return: imputed data.
    '''

    categorical_features = [key for key, value in column_type.items() if value == "Category"]
    continuous_features = [key for key, value in column_type.items() if value == "Continuous"]

    if not ts:
        '''Initialize imputer'''
        imputer_cat = SimpleImputer(strategy='most_frequent')
        imputer_cont = IterativeImputer(random_state=0)

        '''Imputation for continuous data'''
        df[continuous_features] = imputer_cont.fit_transform(df[continuous_features])

        '''Imputation for categorical data'''
        for column in categorical_features:
            df[column] = imputer_cat.fit_transform(df[[column]]).ravel()

    if ts:
        df = df.ffill()

    return df

imputed_data = imputation(df = clean_data, column_type = each_type, ts = False)



def linearity_check (df: pd.DataFrame, test_pairs: int = 1000, alpha: float = 0.1):
    '''
    :param df: imputed data.
    :param test_pairs: maximum number of pairs to be tested.
    :param alpha: significance level.
    :return: indicator of linearity, reset testing results for each pair, fitted OLS model.
    '''


    reset_pval = []
    OLS_model = []

    pair_num = len(test_pairs)

    for i in range(pair_num):
        x = df.iloc[:, test_pairs[i][0]]
        x = sm.add_constant(x).to_numpy()

        y = df.iloc[:, test_pairs[i][1]].to_numpy()

        model = sm.OLS(y, x)
        results = model.fit()

        OLS_model.append(results)

        '''Ramsey’s RESET - H0: linearity is satisfied'''
        reset_test = linear_reset(results)
        reset_pval.append(reset_test.pvalue)

    '''Benjamini & Yekutieli procedure - True: reject H0 -- linearity is not satisfied'''
    corrected_reset = multipletests(reset_pval, alpha=alpha, method='fdr_by')[0]

    if corrected_reset.sum() == 0:
        check_result = {"Linearity": True}
    else:
        check_result = {"Linearity": False}

    return check_result, corrected_reset, OLS_model


linearity_res, all_reset_results, OLS_model = linearity_check(df = imputed_data,
                                                               test_pairs = combinations_select,
                                                               alpha = 0.1)



 # Gaussian error Checking
 #
 # Input: cleaned and transformed data & Linearity testing results
 # Output: testing results
def gaussian_check(df: pd.DataFrame,
                    ols_fit: list,
                    reset_test: list,
                    test_pairs: int = 1000, alpha: float = 0.1):
    '''
    :param df: imputed data.
    :param ols_fit: fitted OLS model for each pair.
    :param reset_test: results of RESET test for each pair.
    :param test_pairs: maximum number of pairs to be tested.
    :param alpha: significance level.
    :return: indicator of gaussian errors.
    '''

    JB_pval = []
    pair_num = len(test_pairs)

    for i in range(pair_num):
        if not reset_test[i]:
            residuals = ols_fit[i].resid
        else:
            x = df.iloc[:, test_pairs[i][0]].to_numpy()
            y = df.iloc[:, test_pairs[i][1]].to_numpy()

            '''Fit Lowess'''
            smoothed = lowess(y, x)
            smoothed_x = smoothed[:, 0]
            smoothed_y = smoothed[:, 1]
            smoothed_values = np.interp(x, smoothed_x, smoothed_y)
            residuals = y - smoothed_values

        '''Jarque–Bera test - H0: residuals are Gaussian'''
        JB_test = jarque_bera(residuals)
        JB_pval.append(JB_test[1])

        '''Benjamini & Yekutieli procedure - True: reject H0 -- Gaussian error assumption is not satisfied'''
        corrected_JB = multipletests(JB_pval, alpha=alpha, method='fdr_by')[0]

        if corrected_JB.sum() == 0:
            check_result = {"Gaussian Error": True}
            continue
        else:
            check_result = {"Gaussian Error": False}

    return check_result

gaussian_res = gaussian_check(df = imputed_data, ols_fit = OLS_model,
                              test_pairs = combinations_select, reset_test = all_reset_results,
                              alpha = 0.1)


def stationary_check(df: pd.DataFrame, max_test: int = 1000, alpha: float = 0.1):
    '''
    :param df: imputed data.
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

        '''ADF test - H0: this series is non-stationary'''
        adf_test = adfuller(x)
        ADF_pval.append(adf_test[1])

        '''Bonferroni correction'''
        '''True: reject H0 -- this series is stationary'''
        '''False: accept H0 -- this series is non-stationary'''
        corrected_ADF = multipletests(ADF_pval, alpha=alpha, method='bonferroni')[0]

        if corrected_ADF.sum() == m:
            check_result = {"Stationary": True}
            continue
        else:
            check_result = {"Stationary": False}
            break

    return check_result


stationary_res = stationary_check(df =  imputed_data, max_test=1000, alpha=0.1)

class ParaStatCollect:
    def __init__(self):
        self.ts = False
        self.ratio = 0.5
        self.alpha = 0.1
        self.num_test = 1000


def stat_info_collection(args, data):
    '''
    :param args: a class contain pre-specified informaiton - indicator of time-series,
                 missing ratio for data cleaning, significance level (default 0.1),
                 maximum number of tests (default 1000).
    :param data: given tabular data in pandas dataFrame format.
    :return: a dict containing all necessary statics information.
    '''

    '''Initialize output'''
    linearity_res = {"Linearity": "time-series"}
    gaussian_res = {"Gaussian Error": "time-series"}
    stationary_res = {"Stationary": "non time-series"}

    '''Data pre-processing'''
    clean_data, miss_res, each_type, dataset_type = data_preprocess(df = data, ratio = args.ratio, ts = args.ts)

    '''Imputation'''
    imputed_data = imputation(df = clean_data, column_type = each_type, ts = args.ts)

    if not args.ts:
        '''Generate combinations of pairs to be tested'''
        m = clean_data.shape[1]
        tot_pairs = m * (m - 1) / 2

        '''Sample pairs without replacement'''
        combinations_list = list(combinations(list(range(m)), 2))

        if tot_pairs > args.num_test:
            num_test = args.num_test
        else:
            num_test = tot_pairs

        num_test = int(num_test)
        combinations_select = random.sample(combinations_list, num_test)

        '''Linearity assumption checking'''
        linearity_res, all_reset_results, OLS_model = linearity_check(df = imputed_data,
                                                                      test_pairs = combinations_select,
                                                                      alpha = args.alpha)

        '''Gaussian error checking'''
        gaussian_res = gaussian_check(df = imputed_data,
                                      ols_fit = OLS_model,
                                      test_pairs = combinations_select,
                                      reset_test = all_reset_results,
                                      alpha = args.alpha)

    '''Assumption checking for time-series data'''
    if args.ts:
        stationary_res = stationary_check(df =  imputed_data, max_test=1000, alpha=0.1)

    stat_info_combine = {**miss_res, **dataset_type, **linearity_res, **gaussian_res, **stationary_res}

    stat_info_combine = json.dumps(stat_info_combine, indent=4)


    return stat_info_combine


'''
Class containing information for statistics information collection:
ts: indicator of time-series.
ratio: missing ratio for data clean.
alpha: significace level.
num_test: maximum number of tests.
'''
class ParaStatCollect:
    def __init__(self):
        self.ts = False
        self.ratio = 0.5
        self.alpha = 0.1
        self.num_test = 1000
