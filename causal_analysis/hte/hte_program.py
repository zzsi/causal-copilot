import json

class HTE_Programming(object):
    def __init__(self, args, y_col: str, T_col: str, X_col: list, W_col: list=None):
        self.args = args
        self.y_col = y_col
        self.T_col = T_col
        self.X_col = X_col
        self.W_col = W_col

    def forward(self, global_state, task='hte'):    
        import causal_analysis.hte.wrappers as wrappers
        algo_func = getattr(wrappers, global_state.inference.hte_algo_json['name'])
        model = algo_func(global_state.inference.hte_params, self.y_col, self.T_col, self.X_col, self.W_col)
        model.fit(global_state.user_data.processed_data)
        if task == 'hte':
            hte, hte_lower, hte_upper = model.hte(global_state.user_data.processed_data)
            return hte, hte_lower, hte_upper
