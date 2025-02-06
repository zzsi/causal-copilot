import numpy as np
import pandas as pd

class DataSimulator:
    def __init__(self, num_samples=1000, seed=42):
        self.num_samples = num_samples
        np.random.seed(seed)

    def simulate_no_confounder(self, beta_0=1, beta_T=0.5, noise_std=1.0):
        """Simulate data with no confounder: Y = β_0 + β_T * T + ε"""
        T = np.random.binomial(1, 0.5, self.num_samples)  # Random binary treatment
        epsilon = np.random.normal(0, noise_std, self.num_samples)
        Y = beta_0 + beta_T * T + epsilon
        data = pd.DataFrame({"Treatment": T, "Outcome": Y})
        return data

    def simulate_with_confounder(self, alpha_0=0.2, alpha_X=1,
                                 beta_0=0.5, beta_X=1, beta_T=1,
                                 noise_std=1.0):
        """Simulate data with confounder X: T = sigmoid(α_0 + α_X * X + η), Y = β_0 + β_X * X + β_T * T + ε"""
        X = np.random.normal(0, 1, self.num_samples)  # Confounder
        eta = np.random.normal(0, 1, self.num_samples)
        prob_T = 1 / (1 + np.exp(-(alpha_0 + alpha_X * X + eta)))
        T = np.random.binomial(1, prob_T)
        epsilon = np.random.normal(0, noise_std, self.num_samples)
        Y = beta_0 + beta_X * X + beta_T * T + epsilon
        data = pd.DataFrame({"Confounder": X, "Treatment": T, "Outcome": Y})
        return data

    def simulate_heterogeneous_effect(self, beta_0=0, beta_X=1, noise_std=1.0):
        """Simulate heterogeneous effect: Y = β_0 + β_T(X) * T + β_X * X + ε"""
        X = np.random.normal(0, 1, self.num_samples)  # Confounder
        T = np.random.binomial(1, 0.5, self.num_samples)

        beta_T_X = 0.5 + 2 * np.cos(X)  # Heterogeneous treatment effect depending on X
        epsilon = np.random.normal(0, noise_std, self.num_samples)
        Y = beta_0 + beta_T_X * T + beta_X * X + epsilon
        data = pd.DataFrame({"Confounder": X, "Treatment": T, "Outcome": Y})
        return data

    def simulate_interaction(self, beta_0=0, beta_T=1, beta_X=1,
                             beta_interact=1, noise_std=1.0):
        """Simulate interaction: Y = β_0 + β_T * T + β_X * X + β_interact * (T * X) + ε"""
        X = np.random.normal(0, 1, self.num_samples)  # Confounder
        T = np.random.binomial(1, 0.5, self.num_samples)
        epsilon = np.random.normal(0, noise_std, self.num_samples)
        Y = beta_0 + beta_T * T + beta_X * X + beta_interact * (T * X) + epsilon
        data = pd.DataFrame({"Confounder": X, "Treatment": T, "Outcome": Y})
        return data

    def simulate_non_linear(self, beta_0=0, beta_T=1, noise_std=1.0, function_type="quadratic"):
        """
        Simulate non-linear case: Y = β_0 + β_T * T + f(X) + ε
        Supports:
        - Quadratic: f(X) = X^2
        - Logarithmic: f(X) = log(|X| + 1)
        """
        X = np.random.normal(0, 1, self.num_samples)  # Confounder
        T = np.random.binomial(1, 0.5, self.num_samples)
        epsilon = np.random.normal(0, noise_std, self.num_samples)

        if function_type == "quadratic":
            f_X = X**2
        elif function_type == "logarithmic":
            f_X = np.log(np.abs(X) + 1)
        else:
            raise ValueError("Unsupported function type. Choose 'quadratic' or 'logarithmic'.")

        Y = beta_0 + beta_T * T + f_X + epsilon
        data = pd.DataFrame({"Confounder": X, "Treatment": T, "Outcome": Y})
        return data

simulator = DataSimulator(num_samples=1000)
# data = simulator.simulate_no_confounder(beta_0=1, beta_T=2, noise_std=1.0)
# print(data.head())

# data = simulator.simulate_with_confounder(alpha_0=0.5, alpha_X=2, beta_0=1, beta_X=1, beta_T=2, noise_std=1.0)
# print(data.head())

# data = simulator.simulate_heterogeneous_effect(beta_0=1, beta_X=0.5, noise_std=0.5)
# print(data.head())

# data = simulator.simulate_interaction(beta_0=1, beta_T=1, beta_X=0.5, beta_interact=2, noise_std=0.5)
# print(data.head())

# Quadratic
# data = simulator.simulate_non_linear(beta_0=1, beta_T=1, noise_std=1.0, function_type="quadratic")
# print(data.head())

# Logarithmic
data = simulator.simulate_non_linear(beta_0=1, beta_T=1, noise_std=1.0, function_type="logarithmic")
print(data.head())


def evaluate_treatment_effect_metrics(true_value, estimations, cis):
    """
    Evaluate metrics for Treatment Effect Estimation: Bias, MSE, and CI Coverage Probability.

    Parameters:
    - true_value (float): The true treatment effect.
    - estimations (np.ndarray): A 1D array of estimated treatment effects.
    - cis (list of tuples): A list of confidence intervals, where each element is a tuple (lower_bound, upper_bound).

    Returns:
    - dict: A dictionary containing Bias, MSE, and Coverage Probability.
    """
    # Convert inputs to numpy arrays for vectorized operations
    estimations = np.array(estimations)
    cis = np.array(cis)

    # Bias: Mean difference between estimation and true value
    bias = np.mean(estimations - true_value)

    # MSE: Mean squared error
    mse = np.mean((estimations - true_value) ** 2)

    # Coverage Probability: Proportion of CIs that contain the true value
    lower_bounds = cis[:, 0]
    upper_bounds = cis[:, 1]
    coverage_count = np.sum((true_value >= lower_bounds) & (true_value <= upper_bounds))
    coverage_probability = coverage_count / len(cis)

    # Return metrics as a dictionary
    return {
        "Bias": bias,
        "MSE": mse,
        "Coverage Probability": coverage_probability
    }