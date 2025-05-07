import pandas as pd
import random
import numpy as np
import torch
import os
import pickle


def dataset(
    name="german", type="variable_p", aux_feat_prob=0.5, use_cuda=False, seed=2022
):
    """Load and prepare a dataset with auxiliary features and masking

    Args:
        name: Dataset name
        type: Type of auxiliary feature availability ('variable_p', 'trapezoidal', 'obsolete_sudden')
        aux_feat_prob: Probability of each auxiliary feature being available
        use_cuda: Whether to use CUDA
        seed: Random seed for reproducibility

    Returns:
        Tuple of (n_base_feat, n_aux_feat, X_base, X_aux, X_aux_new, aux_mask, Y, label)
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Dataset configuration
    dataset_config = {
        "german": {"n_feat": 24, "n_aux_feat": 22, "instances": 1000},
        "svmguide3": {"n_feat": 21, "n_aux_feat": 19, "instances": 1243},
        "magic04": {"n_feat": 10, "n_aux_feat": 8, "instances": 19020},
        "a8a": {"n_feat": 123, "n_aux_feat": 121, "instances": 32561},
        "ItalyPowerDemand": {"n_feat": 24, "n_aux_feat": 12, "instances": 1096},
        "SUSY": {"n_feat": 8, "n_aux_feat": 6, "instances": 1000000},
        "HIGGS": {"n_feat": 21, "n_aux_feat": 16, "instances": 1000000},
    }

    if name not in dataset_config:
        raise ValueError(f"Unknown dataset: {name}")

    config = dataset_config[name]
    n_feat = config["n_feat"]
    n_aux_feat = config["n_aux_feat"]
    n_base_feat = n_feat - n_aux_feat
    number_of_instances = config["instances"]

    # Load and prepare data based on dataset name
    if name == "german":
        data_path = os.path.join(
            os.path.dirname(__file__), "Datasets", name, "german.data-numeric"
        )
        data_initial = pd.read_csv(data_path, sep="  ", header=None, engine="python")
        data_initial.iloc[np.array(data_initial[24].isnull()), 24] = 2.0
        data_shuffled = data_initial.sample(frac=1)  # Randomly shuffle
        label = np.array(data_shuffled[24] == 1) * 1
        data = data_shuffled.iloc[:, :24]
        data.insert(0, column="class", value=label)
        for i in range(data.shape[0]):
            data.iloc[i, 3] = int(data.iloc[i, 3].split(" ")[1])

    elif name == "svmguide3":
        data_path = os.path.join(
            os.path.dirname(__file__), "Datasets", name, "svmguide3.txt"
        )
        data_initial = pd.read_csv(data_path, sep=" ", header=None)
        data_initial = data_initial.iloc[:, :22]

        # Extract feature values from libsvm format
        for j in range(1, data_initial.shape[1]):
            for i in range(data_initial.shape[0]):
                data_initial.iloc[i, j] = data_initial.iloc[i, j].split(":")[1]

        # Convert labels to binary
        for i in range(data_initial.shape[0]):
            data_initial.iloc[i, 0] = (data_initial.iloc[i, 0] == -1) * 1

        data = data_initial.sample(frac=1)
        label = np.asarray(data[0])

    elif name == "magic04":
        data_path = os.path.join(
            os.path.dirname(__file__), "Datasets", name, "magic04.data"
        )
        data_initial = pd.read_csv(data_path, sep=",", header=None)
        data_shuffled = data_initial.sample(frac=1)
        label = np.array(data_shuffled[n_feat] == "g") * 1
        data = data_shuffled.iloc[:, :n_feat]
        data.insert(0, column="class", value=label)

    elif name == "a8a":
        data_path = os.path.join(os.path.dirname(__file__), "Datasets", name, "a8a.txt")
        data = pd.DataFrame(
            0, index=range(number_of_instances), columns=list(range(1, n_feat + 1))
        )
        data_initial = pd.read_csv(data_path, sep=" ", header=None)
        data_initial = data_initial.iloc[:, :15]

        # Process sparse format
        for j in range(data_initial.shape[0]):
            l = [
                int(i.split(":")[0]) - 1
                for i in list(data_initial.iloc[j, 1:])
                if not pd.isnull(i)
            ]
            data.iloc[j, l] = 1

        label = np.array(data_initial[0] == -1) * 1
        data.insert(0, column="class", value=label)
        data = data.sample(frac=1)
        label = np.array(data["class"])

    elif name == "ItalyPowerDemand":
        data_path_train = os.path.join(
            os.path.dirname(__file__), "Datasets", name, "ItalyPowerDemand_TRAIN.txt"
        )
        data_path_test = os.path.join(
            os.path.dirname(__file__), "Datasets", name, "ItalyPowerDemand_TEST.txt"
        )

        # Load and combine data
        data_train = pd.read_csv(
            data_path_train, sep="  ", header=None, engine="python"
        )
        data_test = pd.read_csv(data_path_test, sep="  ", header=None, engine="python")
        data = pd.concat([data_train, data_test])
        label = np.array(data[0] == 1.0) * 1

    elif name in ["SUSY", "HIGGS"]:
        data_path = os.path.join(
            os.path.dirname(__file__), "Datasets", name, "data", f"{name}_1M.csv.gz"
        )

        # Load gzipped data
        data = pd.read_csv(data_path, compression="gzip", nrows=number_of_instances)
        label = np.array(data["0"] == 1.0) * 1

    # Create masking for auxiliary features
    if type == "trapezoidal" and name not in ["ItalyPowerDemand", "SUSY", "HIGGS"]:
        # Trapezoid-shaped availability of features
        num_chunks = 10
        chunk_size = int(number_of_instances / 10)
        aux_mask = np.zeros((number_of_instances, n_aux_feat))
        aux_feat_chunk_list = [
            round((n_feat / num_chunks) * i) - n_base_feat
            for i in range(1, num_chunks + 1)
        ]
        if aux_feat_chunk_list[0] < 0:
            aux_feat_chunk_list[0] = 0

        for i in range(num_chunks):
            aux_mask[
                chunk_size * i : chunk_size * (i + 1), : aux_feat_chunk_list[i]
            ] = 1

    elif type == "variable_p":
        # Random availability with probability aux_feat_prob
        if name in ["SUSY", "HIGGS"]:
            # Load pre-generated mask for large datasets
            mask_file_name = (
                f"{name}_1M_P_{int(aux_feat_prob*100)}_AuxFeat_{n_aux_feat}.data"
            )
            mask_path = os.path.join(
                os.path.dirname(__file__), "Datasets", name, "mask", mask_file_name
            )
            with open(mask_path, "rb") as file:
                aux_mask = pickle.load(file)
        else:
            aux_mask = (
                np.random.random((number_of_instances, n_aux_feat)) < aux_feat_prob
            ).astype(float)

    elif type == "obsolete_sudden" and name in ["SUSY", "HIGGS"]:
        # Pre-generated obsolescence pattern
        if name == "SUSY":
            mask_file_name = (
                f"{name}_1M_Start100k_Gap100k_Stream400k_AuxFeat_{n_aux_feat}.data"
            )
        else:  # HIGGS
            mask_file_name = (
                f"{name}_1M_Start50k_Gap50k_Stream200k_AuxFeat_{n_aux_feat}.data"
            )

        mask_path = os.path.join(
            os.path.dirname(__file__), "Datasets", name, "mask", mask_file_name
        )
        with open(mask_path, "rb") as file:
            aux_mask = pickle.load(file)

    else:
        valid_types = (
            "variable_p"
            + (", trapezoidal" if name not in ["ItalyPowerDemand"] else "")
            + (", obsolete_sudden" if name in ["SUSY", "HIGGS"] else "")
        )
        raise ValueError(
            f"Invalid type '{type}' for dataset '{name}'. Valid types: {valid_types}"
        )

    # Split data into base features, auxiliary features, and labels
    Y = np.array(data.iloc[:, :1])
    X_base = np.array(data.iloc[:, 1 : n_base_feat + 1], dtype=float)
    X_aux = np.array(data.iloc[:, n_base_feat + 1 :], dtype=float)

    # Apply mask to auxiliary features (set unavailable features to 0)
    if name in ["SUSY", "HIGGS"]:
        X_aux_new = np.where(aux_mask[:number_of_instances], X_aux, 0)
    else:
        X_aux_new = np.where(aux_mask, X_aux, 0)

    return n_base_feat, n_aux_feat, X_base, X_aux, X_aux_new, aux_mask, Y, label
