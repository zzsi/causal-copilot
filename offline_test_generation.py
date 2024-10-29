import numpy as np
import os
from data.simulation.dummy import DataSimulator

def generate_datasets():
    base_simulator = DataSimulator()
    
    dataset_configs = [
        # Easy
        {
            "name": "Linear-Gaussian",
            "function_type": "linear",
            "noise_type": "gaussian",
            "heterogeneous": False
        },
        {
            "name": "Linear-Non-gaussian",
            "function_type": "linear",
            "noise_type": "uniform",
            "heterogeneous": False
        },
        # Medium
        {
            "name": "Heterogenous-Linear-Non-gaussian",
            "function_type": "linear",
            "noise_type": "uniform",
            "heterogeneous": True
        },
        {
            "name": "Heterogenous-Linear-Gaussian",
            "function_type": "linear",
            "noise_type": "gaussian",
            "heterogeneous": True
        },
        {
            "name": "Non-Linear-Gaussian",
            "function_type": "mlp",
            "noise_type": "gaussian",
            "heterogeneous": False
        },
        {
            "name": "Heterogenous-Non-Linear-Gaussian",
            "function_type": "mlp",
            "noise_type": "gaussian",
            "heterogeneous": True
        },
    ]

    output_dir = os.path.join(os.getcwd(), "simulated_data")
    os.makedirs(output_dir, exist_ok=True)

    for config in dataset_configs:
        for i in range(10):  # Generate 10 datasets for each configuration
            n_nodes = np.random.choice([5, 10, 25, 50])
            n_samples = np.random.choice([1000, 2500, 5000, 10000])
            edge_probability = 0.3  # You can adjust this if needed

            if config["heterogeneous"]:
                n_domains = np.random.choice([2, 5, 10])
                base_simulator.generate_and_save_dataset(
                    n_nodes=n_nodes,
                    n_samples=n_samples,
                    edge_probability=edge_probability,
                    function_type=config["function_type"],
                    noise_type=config["noise_type"],
                    n_domains=n_domains,
                    prefix=f"{config['name']}_id_{i}",
                    output_dir=output_dir
                )
            else:
                base_simulator.generate_and_save_dataset(
                    n_nodes=n_nodes,
                    n_samples=n_samples,
                    edge_probability=edge_probability,
                    function_type=config["function_type"],
                    noise_type=config["noise_type"],
                    prefix=f"{config['name']}_id_{i}",
                    output_dir=output_dir
                )

if __name__ == "__main__":
    generate_datasets()

