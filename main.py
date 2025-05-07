"""
Optimized implementation of Aux-Drop models for handling data unavailability
"""

import random
import numpy as np
import torch
from tqdm import tqdm

# Import all model variants
from AuxDrop import (
    AuxDrop_ODL,
    AuxDrop_OGD,
    AuxDrop_ODL_AuxLayer1stlayer,
    AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer,
    AuxDrop_ODL_RandomAllLayer,
    AuxDrop_ODL_RandomInAuxLayer,
    AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst,
)
from dataset import dataset


# Configuration
class Config:
    """Configuration class for experiment parameters"""

    def __init__(self):
        # Data parameters
        self.data_name = "magic04"  # Options: "german", "svmguide3", "magic04", "a8a", "ItalyPowerDemand", "SUSY", "HIGGS"
        self.unavailability_type = (
            "variable_p"  # Options: "variable_p", "trapezoidal", "obsolete_sudden"
        )

        # Model selection
        self.model_type = "AuxDrop_ODL"  # Model type to run

        # Model hyperparameters
        self.eta = 0.1  # Learning rate
        self.aux_feat_prob = 0.27  # Probability of auxiliary features
        self.dropout_p = 0.3  # Dropout probability
        self.max_num_hidden_layers = 6
        self.qtd_neuron_per_hidden_layer = 50
        self.n_classes = 2
        self.aux_layer = 3  # Position of auxiliary layer
        self.n_neuron_aux_layer = 100  # Neurons in auxiliary layer
        self.batch_size = 1
        self.b = 0.99  # Parameter for ODL
        self.s = 0.2  # Parameter for ODL
        self.use_cuda = False
        self.num_experiments = 1  # Number of experiments to run


# Helper functions
def validate_config(config):
    """Validate configuration based on dataset and model requirements"""
    # Validate a8a dataset requirements
    if config.data_name == "a8a":
        if (
            config.unavailability_type == "trapezoidal"
            and config.n_neuron_aux_layer < 600
        ):
            raise ValueError(
                "For a8a dataset with trapezoidal type, neurons in aux layer must be ≥ 600"
            )
        elif (
            config.unavailability_type == "variable_p"
            and config.n_neuron_aux_layer < 400
        ):
            raise ValueError(
                "For a8a dataset with variable_p type, neurons in aux layer must be ≥ 400"
            )

    # Validate model compatibility with datasets
    unsupported_models = [
        "AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer",
        "AuxDrop_ODL_RandomAllLayer",
        "AuxDrop_ODL_RandomInAuxLayer",
        "AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst",
    ]

    if config.model_type in unsupported_models and config.data_name not in [
        "german",
        "svmguide3",
        "magic04",
        "a8a",
    ]:
        supported_datasets = ["german", "svmguide3", "magic04", "a8a"]
        raise ValueError(
            f"{config.model_type} only supports these datasets: {supported_datasets}"
        )

    # Check auxiliary layer constraints for OGD
    if config.model_type == "AuxDrop_OGD":
        if config.data_name in ["ItalyPowerDemand", "SUSY", "HIGGS"]:
            raise ValueError(f"AuxDrop_OGD doesn't support {config.data_name} dataset")
        if config.aux_layer == 1:
            raise ValueError("For AuxDrop_OGD, aux layer position must be > 1")


def create_model(config, n_base_feat, n_aux_feat):
    """Create and return the specified model instance"""
    common_params = {
        "features_size": n_base_feat,
        "max_num_hidden_layers": config.max_num_hidden_layers,
        "qtd_neuron_per_hidden_layer": config.qtd_neuron_per_hidden_layer,
        "n_classes": config.n_classes,
        "aux_layer": config.aux_layer,
        "n_neuron_aux_layer": config.n_neuron_aux_layer,
        "batch_size": config.batch_size,
        "n_aux_feat": n_aux_feat,
        "dropout_p": config.dropout_p,
    }

    # Create the appropriate model based on model_type
    if config.model_type == "AuxDrop_ODL":
        if config.aux_layer == 1:
            return AuxDrop_ODL_AuxLayer1stlayer(
                b=config.b,
                n=config.eta,
                s=config.s,
                use_cuda=config.use_cuda,
                **common_params,
            )
        else:
            return AuxDrop_ODL(
                b=config.b,
                n=config.eta,
                s=config.s,
                use_cuda=config.use_cuda,
                **common_params,
            )
    elif config.model_type == "AuxDrop_OGD":
        return AuxDrop_OGD(n=config.eta, **common_params)
    elif config.model_type == "AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer":
        return AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer(
            n=config.eta, **common_params
        )
    elif config.model_type == "AuxDrop_ODL_RandomAllLayer":
        return AuxDrop_ODL_RandomAllLayer(n=config.eta, **common_params)
    elif config.model_type == "AuxDrop_ODL_RandomInAuxLayer":
        return AuxDrop_ODL_RandomInAuxLayer(n=config.eta, **common_params)
    elif config.model_type == "AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst":
        return AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst(
            n=config.eta, **common_params
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def run_experiment(config):
    """Run a single experiment with given configuration"""
    # Set random seed
    seed = random.randint(0, 10000)

    # Load dataset with chosen configuration
    n_base_feat, n_aux_feat, X_base, X_aux, X_aux_new, aux_mask, Y, label = dataset(
        config.data_name,
        type=config.unavailability_type,
        aux_feat_prob=config.aux_feat_prob,
        use_cuda=config.use_cuda,
        seed=seed,
    )

    # Create model
    model = create_model(config, n_base_feat, n_aux_feat)

    # Train model
    N = X_base.shape[0]
    for i in tqdm(range(N), desc="Training"):
        model.partial_fit(
            X_base[i].reshape(1, n_base_feat),
            X_aux_new[i].reshape(1, n_aux_feat),
            aux_mask[i].reshape(1, n_aux_feat),
            Y[i].reshape(1),
        )

    # Calculate metrics
    if config.data_name == "ItalyPowerDemand":
        return np.mean(model.loss_array)
    else:
        prediction = [torch.argmax(i).item() for i in model.prediction]
        error = len(prediction) - sum(prediction == label)
        return error


def main():
    """Main function to run experiments"""
    # Initialize configuration
    config = Config()

    print(
        f"Running {config.model_type} with aux layer {config.aux_layer} on {config.data_name} dataset ({config.unavailability_type})"
    )

    try:
        # Validate configuration
        validate_config(config)

        # Run experiments
        results = []
        for ex in range(config.num_experiments):
            print(f"Experiment {ex+1}/{config.num_experiments}")
            result = run_experiment(config)
            results.append(result)

        # Report results
        metric_name = "loss" if config.data_name == "ItalyPowerDemand" else "error"
        print(
            f"Results for {config.data_name} dataset ({config.num_experiments} experiments):"
        )
        print(f"Mean {metric_name}: {np.mean(results):.4f}")
        print(f"Standard deviation: {np.std(results):.4f}")

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
