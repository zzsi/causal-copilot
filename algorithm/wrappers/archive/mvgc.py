import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)

from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator
from statsmodels.tsa.api import VAR
from scipy.stats import f as f_distr
from sklearn.preprocessing import StandardScaler
from algorithm.wrappers.utils.ts_utils import generate_stationary_linear

# reference - https://github.com/ckassaad/causal_discovery_for_time_series

class GCModel:
    def __init__(self, x, p=22, scale=True):
        """
        :param x: multivariate time series in the form of a pandas dataframe
        :param p: time stamp for prediction
        :param scale: if True normalize data
        """
        self.p = p
        self.d = x.shape[1]
        self.names = list(x.columns.values)
        self.pa = {self.names[i]: [self.names[i]] for i in range(len(self.names))}
        # scaling data
        if scale:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x.values)
            self.X = pd.DataFrame(x_scaled, columns=self.names)
        else:
            self.X = pd.DataFrame(x, columns=self.names)

    def predict(self, model, x):
        x_hat = []
        for t in range(x.shape[0] - self.p):
            temp = pd.DataFrame(model.forecast(x.values[t:(t + self.p)], 1), columns=list(x.columns.values))
            x_hat.append(temp)
        return pd.concat(x_hat, ignore_index=True) if x_hat else pd.DataFrame(columns=list(x.columns.values))

    def f_test(self, var1, var2, m):
        if var1 > var2:
            f_ = np.divide(var1, var2)
        else:
            f_ = np.divide(var2, var1)
        p_value = 1 - f_distr.cdf(f_, m - 1, m - 1)
        return p_value

    def fit(self, alpha=0.05, criterion=None):
        """
        :param alpha: threshold of F-test
        :return: granger causality denpendencies
        """
        model_full = VAR(self.X)
        model_full_fit = model_full.fit(maxlags=self.p, ic=criterion)
        # make prediction
        x_hat = self.predict(model_full_fit, self.X)
        # compute error
        err_full =np.subtract(x_hat.values, self.X.values[self.p:])
        var_full = list(np.var(err_full, axis=0))

        for j in range(self.d):
            x_temp = self.X.drop(columns=[self.names[j]])
            model_rest = VAR(x_temp)
            model_rest_fit = model_rest.fit(maxlags=self.p, ic=criterion)
            # make prediction
            x_hat = self.predict(model_rest_fit, x_temp)

            # compute error
            err_rest = np.subtract(x_hat.values, x_temp.values[self.p:])
            var_rest = list(np.var(err_rest, axis=0))
            # F test (extremely sensitive to non-normality of X and Y)
            var_full_rest = var_full.copy()
            del var_full_rest[j]
            m = x_hat.shape[0]

            for i in range(len(x_hat.columns.values)):
                # Start Test using F-test
                p_value = self.f_test(var_rest[i], var_full_rest[i], m)
                if p_value < alpha:
                    self.pa[x_hat.columns.values[i]].append(self.names[j])

        res_df = pd.DataFrame(np.ones([self.d, self.d]), columns=self.names, index=self.names)
        summary_matrix = np.zeros([self.d, self.d])
        for e in self.pa.keys():
            for c in self.pa[e]:
                res_df[e].loc[c] = 2
                summary_matrix[int(e)][int(c)] = 1
                if res_df[c].loc[e] == 0:
                    res_df[c].loc[e] = 1
        return res_df, summary_matrix

class MVGC(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'p': int,
            'alpha': 0.1, #significance level for F test
            'scale': True, # normalization
            'criterion': None, # information criterion
        }
        self._params.update(params)

    @property
    def name(self):
        return "MVGC"

    def get_params(self):
        return self._params
    
    def get_primary_params(self):
        self._primary_param_keys = ['p', 'scale']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}
    
    def get_secondary_params(self):
        self._secondary_param_keys = ['alpha', 'criterion']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        node_names = list(data.columns)
        max_lag = self._params['p']
        model = GCModel(data, **self.get_primary_params())
        res_df, summary_matrix = model.fit(**self.get_secondary_params())
        
        info = {
            'lag': max_lag,
            'nodes': node_names,
            'model': model
        }

        return summary_matrix, info, res_df

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples = 1000
        n_nodes = 3
        lag = 2
        
        df, gt_graph_lag, gt_graph_summary, graph_net = generate_stationary_linear(
            n_nodes,
            n_samples,
            lag,
            degree_intra=1,
            degree_inter=2,
        )
        print("Testing MVGC algorithm with pandas DataFrame:")
        print("Ground truth summary graph \n", gt_graph_summary)
        # Initialize lists to store metrics
        f1_scores = []
        precisions = []
        recalls = []
        shds = []
        
        # Run the algorithm
        for _ in range(2):
            prediction, _, _ = self.fit(df)
            evaluator = GraphEvaluator()
            print("Prediction\n", prediction)
            metrics = evaluator._compute_single_metrics(gt_graph_summary, prediction)
            f1_scores.append(metrics['f1'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            shds.append(metrics['shd'])

        # Calculate average and standard deviation
        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        avg_precision = np.mean(precisions)
        std_precision = np.std(precisions)
        avg_recall = np.mean(recalls)
        std_recall = np.std(recalls)
        avg_shd = np.mean(shds)
        std_shd = np.std(shds)

        print("\nAverage Metrics:")
        print(f"F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"Recall: {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"SHD: {avg_shd:.4f} ± {std_shd:.4f}")

if __name__ == "__main__":
    params = {
        'p': 2,
        'alpha': 0.4,
        'scale': True,
        'criterion': None,
    }
    mvgc_algo = MVGC(params)
    mvgc_algo.test_algorithm() 