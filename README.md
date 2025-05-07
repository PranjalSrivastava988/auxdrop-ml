# Aux-Drop: Optimized Implementation for Online Learning with Auxiliary Dropouts

**Paper:** [Aux-Drop: Handling Haphazard Inputs in Online Learning Using Auxiliary Dropouts](https://openreview.net/forum?id=R9CgBkeZ6Z)  
**Citation:** See end of file.

## Overview
This repository contains an optimized and refactored implementation of the Aux-Drop framework for online learning with haphazard auxiliary inputs, as described in the referenced paper. The codebase has been extensively improved for clarity, efficiency, and maintainability, as part of a research internship selection project.

## What's New in This Version?

### Major Code Optimizations

#### AuxDrop.py
- Improved code structure with clear section comments and logical parameter organization
- Consistent method naming across classes
- PyTorch broadcasting for reduced memory allocations
- List comprehensions for concise code
- Method reuse across classes to reduce duplication
- Descriptive docstrings and improved variable names
- Enhanced comments for complex operations
- Removal of repetitive code patterns
- Backward compatibility with dependent code

#### baseline.py
- Modular structure with main() and run_experiments() functions
- Configuration dictionary for parameter management
- Enhanced error handling and result reporting
- Improved code readability and documentation
- Cleaned up logic flow and progress reporting

#### ODL.py
- Added docstrings and _build_network() for organized initialization
- Used list comprehensions for concise forward passes
- Improved parameter naming and error messages
- Cleaner initialization with torch.full()
- Logical code grouping and improved comments

#### main.py
- Centralized configuration with a Config class
- Helper functions for validation, model creation, and experiment execution
- Improved error handling and code readability
- Modular design with clear program flow
- Consistent result reporting and documentation

#### parse_logs.py (New)
- Command-line arguments for log directory/output
- Robust path handling with pathlib
- Error handling for file operations
- Modular functions and a main entry point
- Optimized regex usage and output formatting
- Context managers for safe file I/O

#### run_all.py (New)
- Batch Experiment Orchestration System
- Added to automate and streamline large-scale experiments:
- Centralized configuration management for hyperparameter combinations
- Supports combinatorial parameter testing through grid configuration
- Sequential execution of different model variants
- Automatic dataset switching between experiments
- Direct integration with parse_logs.py for automatic result analysis
- Structured log file naming convention for experiment tracing
- Visual progress bar with time estimation

#### dataset.py
- Optimized Data Handling System
- Memory-mapped data loading for large datasets
- On-the-fly feature scaling and normalization
- Dynamic mask application for auxiliary features
- Built-in support for variable probability distributions
- Automatic schema verification
- Unified interface for all supported datasets

## Repository Structure
```
/
├── AuxDrop.py              # Optimized Aux-Drop models
├── ODL.py                  # Optimized ODL baseline
├── baseline.py             # Improved baseline runner
├── main.py                 # Main experiment runner
├── parse_logs.py           # Log parsing and analysis tool
├── run_all.py              # Batch experiment runner
├── dataset.py              # Data loader utilities
├── Datasets/               # Data files and masks (see below)
├── README.md               # Project documentation
└── ...                     # Other supporting files
```

## Datasets
Seven datasets are used: german, svmguide3, magic04, a8a, ItalyPowerDemand, HIGGS, SUSY.

Download instructions and links are in the original paper's README and the Datasets/ folder.

Large datasets (HIGGS, SUSY) should be downloaded separately as described above.

## Installation
Dependencies:
- Python 3.7+
- numpy
- torch
- pandas
- tqdm

Install dependencies with:
```bash
pip install numpy torch pandas tqdm
```

## Usage

### Run Aux-Drop Models
Edit parameters in main.py as needed, then:
```bash
python main.py
```

### Run Baseline ODL Model
Edit parameters in baseline.py, then:
```bash
python baseline.py
```

### Parse Logs
```bash
python parse_logs.py --log_dir <log_directory> --output <output_file>
```

### Batch Experiments
```bash
python run_all.py
```

## Control Parameters
See the comments in main.py and baseline.py for full details on configurable parameters, such as:
- Dataset selection
- Model variant
- Learning rate, dropout, architecture
- Data stream type (variable_p, trapezoidal, obsolete_sudden)

## Contributing
Contributions, issues, and suggestions are welcome. Please open an issue or submit a pull request.

## Citation
If you use this code or the paper, please cite:
```
@article{agarwal2023auxdrop,
  title={Aux-Drop: Handling Haphazard Inputs in Online Learning Using Auxiliary Dropouts},
  author={Rohit Agarwal and Deepak Gupta and Alexander Horsch and Dilip K. Prasad},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://openreview.net/forum?id=R9CgBkeZ6Z},
  note={Reproducibility Certification}
}
```

## Acknowledgements
- Original authors of Aux-Drop and ODL
- Research internship selection committee

---

*For more details, see the original paper and code comments.*  
*This README was updated as part of a research internship project to document and highlight extensive code improvements and optimizations.*

*Happy experimenting!*
