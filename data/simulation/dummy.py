import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Callable, Union
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor
import os
import time
import json
from datetime import datetime

class NoiseDistribution:
    @staticmethod
    def gaussian(size, scale=1.0):
        return np.random.normal(0, scale, size)
    
    @staticmethod
    def uniform(size, scale=1.0):
        return np.random.uniform(-scale, scale, size)
    
    @staticmethod
    def laplace(size, scale=1.0):
        return np.random.laplace(0, scale, size)
    
    @staticmethod
    def student_t(size, df=3, scale=1.0):
        return np.random.standard_t(df, size) * scale

class TransformationLibrary:
    @staticmethod
    def linear(X: np.ndarray, noise_func: Callable, noise_scale: float = 0.1) -> np.ndarray:
        W = np.random.randn(X.shape[1])
        return X.dot(W) + noise_func(X.shape[0], noise_scale)

    @staticmethod
    def polynomial(X: np.ndarray, noise_func: Callable, noise_scale: float = 0.1, degree: int = 2) -> np.ndarray:
        W = [np.random.randn(X.shape[1]) for _ in range(degree + 1)]
        y = sum(X ** d @ W[d] for d in range(degree + 1))
        return y + noise_func(X.shape[0], noise_scale)

    @staticmethod
    def gaussian_process(X: np.ndarray, noise_func: Callable, noise_scale: float = 0.1) -> np.ndarray:
        gp = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(1e-3, 1e3)), n_restarts_optimizer=5, random_state=0)
        y = np.random.randn(X.shape[0])
        gp.fit(X, y)
        return gp.predict(X) + noise_func(X.shape[0], noise_scale)

    @staticmethod
    def sigmoid(X: np.ndarray, noise_func: Callable, noise_scale: float = 0.1) -> np.ndarray:
        W = np.random.randn(X.shape[1])
        return np.sum(W * (1 / (1 + np.exp(-X))), axis=1) + noise_func(X.shape[0], noise_scale)

    @staticmethod
    def neural_network(X: np.ndarray, noise_func: Callable, noise_scale: float = 0.1) -> np.ndarray:
        nn = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000)
        nn.fit(X, np.random.randn(X.shape[0]))
        return nn.predict(X) + noise_func(X.shape[0], noise_scale)

class DataSimulator:
    def __init__(self):
        self.data = None
        self.graph = None
        self.ground_truth = {}
        self.transformation_library = TransformationLibrary()
        self.noise_distribution = NoiseDistribution()

    def generate_graph(self, n_nodes: int, edge_probability: float = 0.3) -> None:
        """Generate a random directed acyclic graph (DAG)."""
        self.graph = nx.gnp_random_graph(n=n_nodes, p=edge_probability, directed=True)
        self.graph = nx.DiGraph([(u, v) for (u, v) in self.graph.edges() if u < v])
        self.graph = nx.relabel_nodes(self.graph, {node: i for i, node in enumerate(self.graph.nodes())})
        self.ground_truth['graph'] = self.graph

    def generate_data(self, n_samples: int, noise_scale: float = 0.1, 
                      noise_type: str = 'gaussian', 
                      function_type: Union[str, List[str], Dict[int, str]] = 'linear') -> None:
        """Generate data based on the graph structure with various functional relationships."""
        if self.graph is None:
            raise ValueError("Generate graph first")

        data = {}
        function_types = ['linear', 'polynomial', 'gaussian_process', 'sigmoid', 'neural_network']
        noise_func = getattr(self.noise_distribution, noise_type)

        if isinstance(function_type, str):
            if function_type != 'random' and function_type not in function_types:
                raise ValueError(f"Invalid function type: {function_type}")
        elif isinstance(function_type, list):
            if not all(ft in function_types for ft in function_type):
                raise ValueError(f"Invalid function type in list: {function_type}")
        elif isinstance(function_type, dict):
            if not all(node in self.graph.nodes() for node in function_type.keys()):
                raise ValueError("Invalid node in function_type dictionary")
            if not all(ft in function_types for ft in function_type.values()):
                raise ValueError(f"Invalid function type in dictionary: {function_type}")
        else:
            raise ValueError("function_type must be a string, list, or dictionary")

        for node in nx.topological_sort(self.graph):
            parents = list(self.graph.predecessors(node))
            if not parents:
                data[node] = noise_func(n_samples, noise_scale)
                func_type = 'independent'
            else:
                parent_data = np.column_stack([data[p] for p in parents])
                if isinstance(function_type, str):
                    if function_type == 'random':
                        func_type = np.random.choice(function_types)
                    else:
                        func_type = function_type
                elif isinstance(function_type, list):
                    func_type = np.random.choice(function_type)
                else:  # function_type is a dictionary
                    func_type = function_type.get(node, np.random.choice(function_types))
                
                func = getattr(self.transformation_library, func_type)
                if func_type == 'polynomial':
                    degree = np.random.randint(2, 5)  # Random degree between 2 and 4
                    data[node] = func(parent_data, noise_func, noise_scale, degree=degree)
                else:
                    data[node] = func(parent_data, noise_func, noise_scale)
            
            self.ground_truth.setdefault('node_functions', {})[node] = func_type

        self.data = pd.DataFrame({f'X{node}': data[node] for node in self.graph.nodes()})
        self.ground_truth['noise_type'] = noise_type
        self.ground_truth['function_type'] = function_type

    def add_categorical_variable(self, n_categories: int = 10, column: str = None) -> None:
        """Add a categorical variable, either as a new column or by converting an existing one."""
        if self.data is None:
            raise ValueError("Generate data first")

        if column and column not in self.data.columns:
            raise ValueError(f"Column {column} not found in data")

        if column:
            self.data[column] = pd.qcut(self.data[column], q=n_categories, labels=range(n_categories))
        else:
            new_col = f'Cat_{len(self.data.columns)}'
            self.data[new_col] = np.random.choice(n_categories, size=len(self.data))
        
        self.ground_truth['categorical'] = column or new_col

    def add_measurement_error(self, error_std: float = 0.1, columns: Union[str, List[str]] = None) -> None:
        """Add measurement error to specified columns or all if not specified."""
        if self.data is None:
            raise ValueError("Generate data first")

        columns = columns or self.data.columns
        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                self.data[col] += np.random.randn(len(self.data)) * error_std
        
        self.ground_truth['measurement_error'] = {col: error_std for col in columns}

    def add_selection_bias(self, condition: Callable[[pd.DataFrame], pd.Series]) -> None:
        """Introduce selection bias based on a given condition."""
        if self.data is None:
            raise ValueError("Generate data first")

        self.data = self.data[condition(self.data)].reset_index(drop=True)
        self.ground_truth['selection_bias'] = str(condition)

    def add_confounding(self, affected_columns: List[str], strength: float = 0.5) -> None:
        """Add a confounder that affects specified columns."""
        if self.data is None:
            raise ValueError("Generate data first")

        confounder = np.random.randn(len(self.data))
        for col in affected_columns:
            self.data[col] += strength * confounder
        
        self.ground_truth['confounding'] = {'affected_columns': affected_columns, 'strength': strength}

    def add_missing_values(self, missing_rate: float = 0.1, columns: Union[str, List[str]] = None) -> None:
        """Introduce missing values to specified columns or all if not specified."""
        if self.data is None:
            raise ValueError("Generate data first")

        columns = columns or self.data.columns
        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            mask = np.random.random(len(self.data)) < missing_rate
            self.data.loc[mask, col] = np.nan
        
        self.ground_truth['missing_rate'] = {col: missing_rate for col in columns}

    def generate_dataset(self, n_nodes: int, n_samples: int, edge_probability: float = 0.3,
                         noise_scale: float = 0.1, noise_type: str = 'gaussian',
                         function_type: Union[str, List[str], Dict[int, str]] = 'random',
                         add_categorical: bool = False, add_measurement_error: bool = False,
                         add_selection_bias: bool = False, add_confounding: bool = False,
                         add_missing_values: bool = False) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """
        Generate a complete dataset with various characteristics.
        """
        self.generate_graph(n_nodes, edge_probability)
        self.generate_data(n_samples, noise_scale, noise_type, function_type)
        
        if add_categorical:
            n_categorical = np.random.randint(1, max(2, n_nodes // 3))
            for _ in range(n_categorical):
                self.add_categorical_variable(n_categories=np.random.randint(2, 5))
        
        if add_measurement_error:
            self.add_measurement_error(error_std=np.random.uniform(0.05, 0.2))
        
        if add_selection_bias:
            condition = lambda df: df[np.random.choice(df.columns)] > df[np.random.choice(df.columns)].median()
            self.add_selection_bias(condition)
        
        if add_confounding:
            n_confounded = np.random.randint(2, max(3, n_nodes // 2))
            affected_columns = np.random.choice(self.data.columns, n_confounded, replace=False)
            self.add_confounding(affected_columns.tolist(), strength=np.random.uniform(0.3, 0.7))
        
        if add_missing_values:
            self.add_missing_values(missing_rate=np.random.uniform(0.05, 0.2))
        
        return self.graph, self.data

    def save_simulation(self, output_dir: str = 'simulated_data', prefix: str = 'base') -> None:
        """
        Save the simulated data, graph structure, and simulation settings.
        
        :param output_dir: The directory to save the files
        :param prefix: A prefix for the filenames
        """
        if self.data is None or self.graph is None:
            raise ValueError("No data or graph to save. Generate dataset first.")

        # Create a timestamped folder with specific settings
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_nodes = self.graph.number_of_nodes()
        n_samples = len(self.data)
        folder_name = f"{timestamp}_{prefix}_nodes{n_nodes}_samples{n_samples}"
        save_dir = os.path.join(output_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the data as CSV
        data_filename = os.path.join(save_dir, f'{prefix}_data.csv')
        self.data.to_csv(data_filename, index=False)
        print(f"Data saved to {data_filename}")
        
        # Save the graph structure as NPY
        graph_filename = os.path.join(save_dir, f'{prefix}_graph.npy')
        np.save(graph_filename, nx.to_numpy_array(self.graph))
        print(f"Graph structure saved to {graph_filename}")
        
        # Save the simulation settings as JSON
        config_filename = os.path.join(save_dir, f'{prefix}_config.json')
        config = {
            'n_nodes': n_nodes,
            'n_samples': n_samples,
            'noise_type': self.ground_truth.get('noise_type'),
            'function_type': self.ground_truth.get('function_type'),
            'node_functions': self.ground_truth.get('node_functions'),
            'categorical': self.ground_truth.get('categorical'),
            'measurement_error': self.ground_truth.get('measurement_error'),
            'selection_bias': self.ground_truth.get('selection_bias'),
            'confounding': self.ground_truth.get('confounding'),
            'missing_rate': self.ground_truth.get('missing_rate')
        }
        with open(config_filename, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"Simulation settings saved to {config_filename}")

    def generate_and_save_dataset(self, n_nodes: int, n_samples: int, output_dir: str = 'simulated_data', prefix: str = 'base', **kwargs) -> None:
        """
        Generate a dataset and save the results.
        
        :param n_nodes: Number of nodes in the graph
        :param n_samples: Number of samples in the dataset
        :param output_dir: The directory to save the files
        :param prefix: A prefix for the filenames
        :param kwargs: Additional arguments for the generate_dataset method
        """
        self.generate_dataset(n_nodes, n_samples, **kwargs)
        self.save_simulation(output_dir, prefix)

class DomainSpecificSimulator:
    def __init__(self):
        self.base_simulator = DataSimulator()

    def simulate_gene_regulatory_network(self, n_genes: int, n_samples: int) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """Simulate a gene regulatory network."""
        graph, data = self.base_simulator.generate_dataset(
            n_nodes=n_genes,
            n_samples=n_samples,
            edge_probability=0.1,  # Sparse connections in gene networks
            function_type=['sigmoid', 'polynomial'],  # Gene interactions often involve thresholds and complex relationships
            noise_type='gaussian',
            noise_scale=0.05,  # Gene expression measurements often have low noise
            add_measurement_error=True,
            add_missing_values=True
        )
        
        # Rename columns to gene names
        data.columns = [f'Gene_{i}' for i in range(n_genes)]
        
        # Add time component for gene expression
        time = np.linspace(0, 10, n_samples)
        data['Time'] = time
        
        return graph, data

    def simulate_climate_data(self, n_variables: int, n_samples: int) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """Simulate climate data with various interdependent variables."""
        graph, data = self.base_simulator.generate_dataset(
            n_nodes=n_variables,
            n_samples=n_samples,
            edge_probability=0.3,  # More connections in climate systems
            function_type=['polynomial', 'gaussian_process'],  # Climate relationships can be complex and nonlinear
            noise_type='student_t',  # Climate data often has heavy tails
            noise_scale=0.1,
            add_measurement_error=True,
            add_missing_values=True
        )
        
        # Rename columns to climate variables
        climate_vars = ['Temperature', 'Humidity', 'Pressure', 'WindSpeed', 'Precipitation', 'CO2Level']
        data.columns = climate_vars[:n_variables]
        
        # Add seasonal component
        data['Season'] = np.tile(['Winter', 'Spring', 'Summer', 'Fall'], n_samples // 4 + 1)[:n_samples]
        
        return graph, data

    def simulate_economic_data(self, n_indicators: int, n_samples: int) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """Simulate economic data with various economic indicators."""
        graph, data = self.base_simulator.generate_dataset(
            n_nodes=n_indicators,
            n_samples=n_samples,
            edge_probability=0.4,  # Economic indicators are often interconnected
            function_type=['linear', 'polynomial'],  # Economic relationships can be linear or nonlinear
            noise_type='laplace',  # Economic data can have sharp changes
            noise_scale=0.15,
            add_measurement_error=True,
            add_selection_bias=True,  # Economic data often has selection bias
            add_confounding=True
        )
        
        # Rename columns to economic indicators
        economic_vars = ['GDP', 'Inflation', 'Unemployment', 'InterestRate', 'ExchangeRate', 'StockIndex']
        data.columns = economic_vars[:n_indicators]
        
        # Add time component
        data['Year'] = pd.date_range(start='2000-01-01', periods=n_samples, freq='Q')
        
        return graph, data

    def simulate_social_network(self, n_users: int, n_interactions: int) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """Simulate a social network with user interactions."""
        graph = nx.barabasi_albert_graph(n=n_users, m=3)  # Scale-free network typical for social networks
        graph = nx.DiGraph(graph)  # Convert to directed graph
        
        user_data = pd.DataFrame({
            'Age': np.random.randint(18, 80, n_users),
            'ActivityLevel': np.random.uniform(0, 1, n_users),
            'Influence': np.random.power(0.5, n_users)  # Power law distribution for influence
        })
        
        source_nodes = np.random.choice(n_users, n_interactions, replace=True)
        target_nodes = [list(graph.neighbors(node))[0] if list(graph.neighbors(node)) else np.random.choice(n_users) 
                        for node in source_nodes]
        
        interaction_data = pd.DataFrame({
            'Source': source_nodes,
            'Target': target_nodes,
            'Timestamp': pd.date_range(start='2023-01-01', periods=n_interactions, freq='H'),
            'InteractionType': np.random.choice(['Like', 'Comment', 'Share'], n_interactions)
        })
        
        return graph, (user_data, interaction_data)

    def simulate_healthcare_data(self, n_patients: int, n_variables: int) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """Simulate healthcare data for patients."""
        graph, data = self.base_simulator.generate_dataset(
            n_nodes=n_variables,
            n_samples=n_patients,
            edge_probability=0.2,
            function_type=['sigmoid', 'polynomial', 'linear'],
            noise_type='gaussian',
            noise_scale=0.1,
            add_measurement_error=True,
            add_missing_values=True,
            add_categorical=True
        )
        
        health_vars = ['Age', 'BMI', 'BloodPressure', 'CholesterolLevel', 'GlucoseLevel', 'HeartRate']
        data.columns = health_vars[:n_variables]
        
        data['Gender'] = np.random.choice(['Male', 'Female'], n_patients)
        data['SmokingStatus'] = np.random.choice(['Never', 'Former', 'Current'], n_patients)
        
        data['Diagnosis'] = (data['BloodPressure'] > data['BloodPressure'].mean() + 
                             data['CholesterolLevel'] > data['CholesterolLevel'].mean()).astype(int)
        
        return graph, data

    def simulate_ecological_data(self, n_species: int, n_samples: int) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """Simulate ecological data with species interactions."""
        graph, data = self.base_simulator.generate_dataset(
            n_nodes=n_species,
            n_samples=n_samples,
            edge_probability=0.15,
            function_type=['sigmoid', 'polynomial'],
            noise_type='gaussian',
            noise_scale=0.1,
            add_measurement_error=True,
            add_missing_values=True
        )
        
        data.columns = [f'Species_{i}' for i in range(n_species)]
        
        data['Temperature'] = np.random.normal(15, 5, n_samples)
        data['Rainfall'] = np.random.gamma(2, 2, n_samples)
        data['Season'] = np.tile(['Spring', 'Summer', 'Fall', 'Winter'], n_samples // 4 + 1)[:n_samples]
        
        return graph, data

    def simulate_neuroscience_data(self, n_regions: int, n_timepoints: int) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """Simulate brain connectivity data."""
        graph, data = self.base_simulator.generate_dataset(
            n_nodes=n_regions,
            n_samples=n_timepoints,
            edge_probability=0.2,
            function_type=['sigmoid', 'gaussian_process'],
            noise_type='gaussian',
            noise_scale=0.05,
            add_measurement_error=True
        )
        
        data.columns = [f'Region_{i}' for i in range(n_regions)]
        
        data['Task'] = np.random.choice(['Rest', 'Task1', 'Task2'], n_timepoints)
        data['Subject'] = np.repeat(range(n_timepoints // 100 + 1), 100)[:n_timepoints]
        
        return graph, data

    def simulate_physics_data(self, n_particles: int, n_timepoints: int) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """Simulate particle physics data."""
        graph, data = self.base_simulator.generate_dataset(
            n_nodes=n_particles * 3,  # x, y, z coordinates for each particle
            n_samples=n_timepoints,
            edge_probability=0.1,
            function_type=['polynomial', 'neural_network'],
            noise_type='gaussian',
            noise_scale=0.01,
            add_measurement_error=True
        )
        
        columns = [f'Particle_{i}_{coord}' for i in range(n_particles) for coord in ['x', 'y', 'z']]
        data.columns = columns
        
        data['Time'] = np.linspace(0, 10, n_timepoints)
        data['Energy'] = np.sum(data[columns] ** 2, axis=1) / 2  # Kinetic energy proxy
        
        return graph, data

    def save_simulation(self, graph: nx.DiGraph, data: pd.DataFrame, domain: str, output_dir: str = 'simulated_data', **kwargs) -> None:
        """
        Save the simulated data, graph structure, and simulation settings.
        
        :param graph: The ground truth graph structure
        :param data: The simulated data
        :param domain: The name of the simulated domain (e.g., 'neuroscience', 'climate')
        :param output_dir: The directory to save the files
        :param kwargs: Additional settings to include in the config file
        """
        # Create a timestamped folder with specific settings
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_nodes = graph.number_of_nodes()
        n_samples = len(data)
        settings = '_'.join(f"{k}{v}" for k, v in kwargs.items())
        folder_name = f"{timestamp}_{domain}_nodes{n_nodes}_samples{n_samples}_{settings}"
        save_dir = os.path.join(output_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the data as CSV
        data_filename = os.path.join(save_dir, f'{domain}_data.csv')
        data.to_csv(data_filename, index=False)
        print(f"Data saved to {data_filename}")
        
        # Save the graph structure as NPY
        graph_filename = os.path.join(save_dir, f'{domain}_graph.npy')
        np.save(graph_filename, nx.to_numpy_array(graph))
        print(f"Graph structure saved to {graph_filename}")
        
        # Save the simulation settings as JSON
        config_filename = os.path.join(save_dir, f'{domain}_config.json')
        config = {
            'domain': domain,
            'n_nodes': n_nodes,
            'n_samples': n_samples,
            **kwargs  # Include all additional settings
        }
        with open(config_filename, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"Simulation settings saved to {config_filename}")

    def simulate_and_save(self, domain: str, output_dir: str = 'simulated_data', **kwargs) -> None:
        """
        Simulate data for a specific domain and save the results.
        
        :param domain: The name of the domain to simulate
        :param output_dir: The directory to save the files
        :param kwargs: Additional arguments for the specific simulation function
        """
        simulation_method = getattr(self, f'simulate_{domain}_data', None)
        if simulation_method is None:
            raise ValueError(f"No simulation method found for domain: {domain}")
        
        graph, data = simulation_method(**kwargs)
        self.save_simulation(graph, data, domain, output_dir, **kwargs)


# Generate pure simulated data using base simulator
base_simulator = DataSimulator()

# 1. Linear Gaussian data (simple)
base_simulator.generate_and_save_dataset(function_type='linear', n_nodes=5, n_samples=1000, edge_probability=0.3)

# 2. Non-linear Gaussian data
base_simulator.generate_and_save_dataset(function_type='polynomial', n_nodes=10, n_samples=2000, edge_probability=0.4)

# 3. Linear Non-Gaussian data
base_simulator.generate_and_save_dataset(function_type='linear', noise_type='uniform', n_nodes=15, n_samples=3000, edge_probability=0.3)

# 4. Discrete data
# base_simulator.generate_and_save_dataset(function_type='neural_network', n_nodes=8, n_samples=1500, edge_probability=0.35, 
#                                          add_categorical=True)

# 5. Mixed data (complex)
base_simulator.generate_and_save_dataset(function_type=['linear', 'polynomial', 'neural_network'], n_nodes=20, n_samples=5000, edge_probability=0.25)



# Example usage
# domain_simulator = DomainSpecificSimulator()

# # Simulate and save neuroscience data
# domain_simulator.simulate_and_save('neuroscience', n_regions=20, n_timepoints=1000)

# # Simulate and save climate data
# domain_simulator.simulate_and_save('climate', n_variables=6, n_samples=1000)

# # Simulate and save gene regulatory network data
# domain_simulator.simulate_and_save('gene_regulatory_network', n_genes=50, n_samples=500)