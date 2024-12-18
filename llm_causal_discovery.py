import numpy as np
import os
from data.simulation.dummy import DataSimulator

def generate_datasets():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Default settings
    default_settings = {
        "n_samples": 7680,
        "edge_probability": 0.1,
        "function_type": "linear",
        "noise_type": "gaussian",
        "noise_scale": 1.0
    }

    dataset_configs = []
    
    # Models
    # models = ["Bert-small", "Bert-large", "Llama"]
    
    # Sample sizes
    sample_sizes = [7680, 10240, 40960]
    
    # Graph sizes
    graph_sizes = [5, 10, 15, 25]
    
    # Densities
    densities = [0.1, 0.2, 0.3]
    
    # Functional forms
    functional_forms = ["linear", "mlp"]
    
    # Noise forms
    noise_forms = ["gaussian", "uniform"]
    
    # Generate configurations
    # for model in models:
    for n_samples in sample_sizes:
        for n_nodes in graph_sizes:
            for edge_prob in densities:
                for function_type in functional_forms:
                    for noise_type in noise_forms:
                        config = default_settings.copy()
                        config.update({
                            "name": f"samples_{n_samples}_nodes_{n_nodes}_density_{int(edge_prob*100)}_function_{function_type}_noise_{noise_type}",
                            "n_samples": n_samples,
                            "n_nodes": n_nodes,
                            "edge_probability": edge_prob,
                            "function_type": function_type,
                            "noise_type": noise_type
                        })
                        dataset_configs.append(config)

    output_dir = os.path.join(os.getcwd(), "simulated_data/llm_causal_discovery")
    os.makedirs(output_dir, exist_ok=True)

    # Generate 3 random versions of each configuration
    for config in dataset_configs:
        for i in range(3):
            # Set seed based on configuration to ensure reproducibility
            seed = hash(f"{config['name']}_{i}") % (2**32)
            np.random.seed(seed)

            simulator = DataSimulator()
            simulator.generate_and_save_dataset(
                n_nodes=config["n_nodes"],
                n_samples=config["n_samples"],
                edge_probability=config["edge_probability"],
                function_type=config["function_type"],
                noise_type=config["noise_type"],
                noise_scale=config["noise_scale"],
                discrete_ratio=config.get("discrete_ratio", 0.0),
                add_measurement_error=config.get("add_measurement_error", False),
                add_missing_values=config.get("add_missing_values", False),
                error_std=config.get("error_std", 0.1),
                error_rate=config.get("error_rate", 0.1),
                missing_rate=config.get("missing_rate", 0.0),
                n_domains=config.get("n_domains", 1),
                output_dir=output_dir,
                prefix=f"{config['name']}_seed_{i}"
            )
            

if __name__ == "__main__":
    generate_datasets()
