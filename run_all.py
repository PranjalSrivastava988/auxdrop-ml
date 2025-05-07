#!/usr/bin/env python3
"""
Run experiments with all combinations of datasets, types, and models.
Each run is logged to a separate file.
"""
import subprocess
import argparse
import os
from datetime import datetime
import itertools

# Available options
DATA_NAMES = [
    "german",
    "svmguide3",
    "magic04",
    "a8a",
    "ItalyPowerDemand",
    "SUSY",
    "HIGGS",
]

TYPES = ["variable_p", "trapezoidal", "obsolete_sudden"]

MODELS = [
    "AuxDrop_ODL",
    "AuxDrop_OGD",
    "AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer",
    "AuxDrop_ODL_RandomAllLayer",
    "AuxDrop_ODL_RandomInAuxLayer",
    "AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst",
]

# Default hyperparameters
DEFAULT_PARAMS = {
    "n": "0.01",
    "aux_feat_prob": "0.5",
    "dropout_p": "0.3",
    "max_num_hidden_layers": "3",
    "qtd_neuron_per_hidden_layer": "64",
    "n_classes": "2",
    "aux_layer": "3",
    "n_neuron_aux_layer": "32",
    "b": "0.9",
    "s": "0.1",
}


def parse_args():
    """Parse command line arguments for selective running"""
    parser = argparse.ArgumentParser(description="Run AuxDrop experiments")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATA_NAMES,
        help="Specific datasets to run (default: all)",
    )
    parser.add_argument(
        "--types", nargs="+", choices=TYPES, help="Specific types to run (default: all)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        help="Specific models to run (default: all)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory to store log files"
    )
    parser.add_argument(
        "--params", nargs="+", help="Override default params (format: param=value)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Use specified subsets or full lists
    datasets = args.datasets if args.datasets else DATA_NAMES
    types = args.types if args.types else TYPES
    models = args.models if args.models else MODELS

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    # Update parameters with any overrides
    params = DEFAULT_PARAMS.copy()
    if args.params:
        for param in args.params:
            if "=" in param:
                key, value = param.split("=", 1)
                params[key] = value

    # Start timestamp for run
    start_time = datetime.now()
    print(f"🚀 Starting run_all.py at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get valid combinations (some models don't work with certain datasets/types)
    combinations = list(itertools.product(datasets, types, models))
    total = len(combinations)

    # Initialize counters
    completed = 0
    skipped = 0
    failed = 0

    # Run all combinations
    for idx, (data, stream_type, model) in enumerate(combinations):
        # Create command
        cmd = [
            "python",
            "main.py",
            "--data_name",
            data,
            "--type",
            stream_type,
            "--model_to_run",
            model,
        ]

        # Add parameters
        for k, v in params.items():
            cmd.extend([f"--{k}", v])

        # Setup logging
        logfile = os.path.join(
            args.log_dir, f"results_{data}_{stream_type}_{model}.log"
        )
        progress = f"[{idx+1}/{total}]"
        print(f"{progress} Running: {data} | {stream_type} | {model} → {logfile}")

        try:
            with open(logfile, "w") as logf:
                result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

            if result.returncode == 0:
                completed += 1
            else:
                failed += 1
                print(f"  ❌ Failed with return code {result.returncode}")
        except Exception as e:
            failed += 1
            print(f"  ❌ Error: {str(e)}")

    # Print summary
    duration = datetime.now() - start_time
    print("\n============ Summary ============")
    print(f"Total runs: {total}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Time taken: {duration}")
    print("=================================")


if __name__ == "__main__":
    main()
