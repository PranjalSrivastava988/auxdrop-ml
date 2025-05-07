"""
Baseline implementation for evaluating ODL on datasets with different feature configurations
"""

import random
import numpy as np
import torch
from tqdm import tqdm
from ODL import ODL
from dataset import dataset


def run_experiments(config):
    """Run experiments with given configuration and return error statistics"""
    error_list = []

    for ex in range(config["num_experiments"]):
        print(f"Experiment {ex + 1}/{config['num_experiments']}")

        # Set random seed for reproducibility
        seed = random.randint(0, 10000)

        # Load dataset
        n_base_feat, _, X_base, X_aux, _, _, Y, label = dataset(
            config["data_name"],
            type="variable_p",
            aux_feat_prob=0.99,
            use_cuda=config["use_cuda"],
            seed=seed,
        )

        # Configure feature set based on data_type
        if config["data_type"] == "only_base":
            X = X_base
        elif config["data_type"] == "all_feat":
            X = np.concatenate((X_base, X_aux), axis=1)
        else:
            raise ValueError(f"Invalid data type: {config['data_type']}")

        # Initialize model
        model = ODL(
            features_size=X.shape[1],
            max_num_hidden_layers=config["max_hidden_layers"],
            qtd_neuron_per_hidden_layer=config["neurons_per_layer"],
            n_classes=config["n_classes"],
            batch_size=config["batch_size"],
            b=config["b"],
            n=config["learning_rate"],
            s=config["s"],
            use_cuda=config["use_cuda"],
        )

        # Train model
        N = X.shape[0]
        for i in tqdm(range(N), desc="Training"):
            model.partial_fit(X[i].reshape(1, X.shape[1]), Y[i].reshape(1))

        # Calculate error
        prediction = model.prediction
        error = sum(np.array(prediction) != label)
        error_list.append(error)

    # Calculate statistics
    mean_error = np.mean(error_list)
    std_error = np.std(error_list)

    return {"errors": error_list, "mean": mean_error, "std": std_error}


def main():
    """Main function to configure and run experiments"""
    # Configuration
    config = {
        # Dataset parameters
        "data_name": "SUSY",  # Options: "SUSY", "HIGGS"
        "data_type": "all_feat",  # Options: "only_base", "all_feat"
        # Model parameters
        "learning_rate": 0.05,
        "max_hidden_layers": 11,
        "neurons_per_layer": 50,
        "n_classes": 2,
        "batch_size": 1,
        "b": 0.99,
        "s": 0.2,
        # Experiment settings
        "use_cuda": False,
        "num_experiments": 1,
    }

    # Run experiments
    results = run_experiments(config)

    # Print results
    print(
        f"Results for {config['data_name']} dataset ({config['num_experiments']} experiments):"
    )
    print(f"Mean error: {results['mean']}")
    print(f"Standard deviation: {results['std']}")


if __name__ == "__main__":
    main()
