import numpy as np
import pandas as pd
from typing import Dict, Tuple

from econml.dr import DRLearner as Econ_DRL
from econml.dr import LinearDRLearner as Econ_LinearDRL
from econml.dr import SparseLinearDRLearner as Econ_SparseLinearDRL
from econml.dr import ForestDRLearner as Econ_ForestDRL

from .base import Estimator

# - DRL
# - LinearDRL
# - SparseLinearDRLL
# - ForestDRL

class DRL(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None):
        super().__init__(params, y_col, T_col, X_col, W_col)
        self.model = Econ_DRL(**self._params)

    @property
    def name(self):
        return "DRL"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]]
        T = data[[self.T_col]]
        X = data[self.X_col] if self.X_col else None
        W = data[self.W_col] if self.W_col else None
        # Run DRL algorithm        
        self.model.fit(Y=y, T=T, X=X, W=W)
            
    def ate(self, data: pd.DataFrame):
        ate = self.model.ate(X=None, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=None, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col] if self.X_col else None
        treated_indices = (data[self.T_col] == 1)
        treated_effects = self.model.effect(X[treated_indices], T0=self.T0, T1=self.T1)
        lower_bound, upper_bound = self.model.effect_interval(X[treated_indices], T0=self.T0, T1=self.T1)
        att = np.mean(treated_effects)
        att_lower, att_upper = np.mean(lower_bound), np.mean(upper_bound)
        return att, att_lower, att_upper

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X, T0=self.T0, T1=self.T1)
        hte_lower, hte_upper = self.model.effect_interval(X, T0=self.T0, T1=self.T1)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass


class LinearDRL(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None):
        del params['model_final']
        super().__init__(params, y_col, T_col, X_col, W_col)
        self.model = Econ_LinearDRL(**self._params)

    @property
    def name(self):
        return "LinearDRL"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]]
        T = data[[self.T_col]]
        X = data[self.X_col] if self.X_col else None
        W = data[self.W_col] if self.W_col else None
        # Run DRL algorithm        
        self.model.fit(y, T, X=X, W=W)
        
    def ate(self, data: pd.DataFrame):
        ate = self.model.ate(X=None, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=None, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
        treated_indices = (data[self.T_col] == 1)
        treated_effects = self.model.effect(X[treated_indices], T0=self.T0, T1=self.T1)
        lower_bound, upper_bound = self.model.effect_interval(X[treated_indices], T0=self.T0, T1=self.T1)
        att = np.mean(treated_effects)
        att_lower, att_upper = np.mean(lower_bound), np.mean(upper_bound)
        return att, att_lower, att_upper

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X, T0=self.T0, T1=self.T1)
        hte_lower, hte_upper = self.model.effect_interval(X, T0=self.T0, T1=self.T1)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass

class SparseLinearDRL(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None):
        del params['model_final']
        super().__init__(params, y_col, T_col, X_col, W_col)
        self.model = Econ_SparseLinearDRL(**self._params)

    @property
    def name(self):
        return "SparseLinearDRL"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]]
        T = data[[self.T_col]]
        X = data[self.X_col] if self.X_col else None
        W = data[self.W_col] if self.W_col else None
        # Run DRL algorithm        
        self.model.fit(y, T, X=X, W=W)
           
    def ate(self, data: pd.DataFrame):
        ate = self.model.ate(X=None, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=None, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
        treated_indices = (data[self.T_col] == 1)
        treated_effects = self.model.effect(X[treated_indices], T0=self.T0, T1=self.T1)
        lower_bound, upper_bound = self.model.effect_interval(X[treated_indices], T0=self.T0, T1=self.T1)
        att = np.mean(treated_effects)
        att_lower, att_upper = np.mean(lower_bound), np.mean(upper_bound)
        return att, att_lower, att_upper

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X, T0=self.T0, T1=self.T1)
        hte_lower, hte_upper = self.model.effect_interval(X, T0=self.T0, T1=self.T1)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass

class ForestDRL(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None):
        del params['model_final']
        super().__init__(params, y_col, T_col, X_col, W_col)
        self.model = Econ_ForestDRL(**self._params)

    @property
    def name(self):
        return "CausalForestDRL"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]]
        T = data[[self.T_col]]
        X = data[self.X_col] if self.X_col else None
        W = data[self.W_col] if self.W_col else None
        # Run DRL algorithm        
        self.model.fit(y, T, X=X, W=W)
    
    def ate(self, data: pd.DataFrame):
        ate = self.model.ate(X=None, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=None,T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col] if self.X_col else None
        treated_indices = (data[self.T_col] == 1)
        treated_effects = self.model.effect(X[treated_indices], T0=self.T0, T1=self.T1)
        lower_bound, upper_bound = self.model.effect_interval(X[treated_indices], T0=self.T0, T1=self.T1)
        att = np.mean(treated_effects)
        att_lower, att_upper = np.mean(lower_bound), np.mean(upper_bound)
        return att, att_lower, att_upper

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X, T0=self.T0, T1=self.T1)
        hte_lower, hte_upper = self.model.effect_interval(X, T0=self.T0, T1=self.T1)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass


if __name__ == "__main__":
    pc_algo = DRL(y_col='', T_col='', X_col='')
    pc_algo.test_algorithm() 