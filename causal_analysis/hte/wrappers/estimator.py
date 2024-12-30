import numpy as np
import pandas as pd
from typing import Dict, Tuple

from econml.dml import DML

from .base import Estimator

# - DML 
# - LinearDML
# - SparseLinearDML
# - CausalForestDML
# - metalearners

class DML(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None):
        super().__init__(params)
        self._params = {}
        self._params.update(params)
        self.model = DML(**self._params)
        self.y_col = y_col
        self.T_col = T_col
        self.X_col = X_col
        self.W_col = W_col

    @property
    def name(self):
        return "DML"

    def get_params(self):
        return self._params

    # def get_primary_params(self):
    #     self._primary_param_keys = ['alpha', 'indep_test', 'depth']
    #     return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    # def get_secondary_params(self):
    #     self._secondary_param_keys = ['stable', 'uc_rule', 'uc_priority', 'mvpc', 'correction_name',
    #                                   'background_knowledge', 'verbose', 'show_progress']
    #     return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]]
        T = data[[self.T_col]]
        X = data[self.X_col]
        W = data[self.W_col]
        # Run DML algorithm        
        self.model.fit(y, T, X=X, W=W)

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X)
        hte_lower, hte_upper = self.model.effect_interval(X)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass

if __name__ == "__main__":
    pc_algo = DML(y_col='', T_col='', X_col='')
    pc_algo.test_algorithm() 