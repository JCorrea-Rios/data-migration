# Data Migration Project - Source Code Structure

This document outlines the organization of the source code for the Data Migration project.

## Directory Structure

The source code is organized into the following modules:

```
src/
├── __init__.py                  # Package initialization
├── core/                        # Core functionality
│   ├── __init__.py
│   ├── preprocessing.py         # Data preprocessing
│   ├── features.py              # Feature engineering
│   ├── train.py                 # Model training
│   └── inference.py             # Model inference
├── clustering/                  # Clustering functionality
│   ├── __init__.py
│   ├── cluster.py               # Clustering algorithms
│   ├── blocking.py              # Blocking strategies
│   └── evaluation.py            # Clustering evaluation
├── sanitization/                # Company name sanitization
│   ├── __init__.py
│   ├── company_name_sanitiser.py # Main sanitization logic
│   ├── gliner_classifier.py     # GLiNER integration
│   └── test_integration.py      # Test scripts
├── storage/                     # Data storage and persistence
│   ├── __init__.py
│   ├── db_manager.py            # DuckDB manager
│   └── db_utils.py              # Database utilities
└── utils/                       # Common utilities
    ├── __init__.py
    └── common.py                # Shared utility functions
```

## Module Descriptions

### Core Module

The `core` module contains the fundamental functionality for data processing, feature engineering, model training, and inference.

- **preprocessing.py**: Functions for cleaning and normalizing text data
- **features.py**: Feature extraction and engineering for company name matching
- **train.py**: Model training pipelines and utilities
- **inference.py**: Model inference and prediction functions

### Clustering Module

The `clustering` module provides algorithms and utilities for clustering similar company names.

- **cluster.py**: Implementation of clustering algorithms (DBSCAN, Agglomerative, etc.)
- **blocking.py**: Blocking strategies to reduce the comparison space
- **evaluation.py**: Metrics and functions for evaluating clustering quality

### Sanitization Module

The `sanitization` module handles company name standardization and entity recognition.

- **company_name_sanitiser.py**: Main sanitization logic for company names
- **gliner_classifier.py**: Integration with GLiNER for entity recognition
- **test_integration.py**: Test scripts for the sanitization module

### Storage Module

The `storage` module manages data persistence and database operations.

- **db_manager.py**: DuckDB database manager for batch processing
- **db_utils.py**: Utilities for database management and reporting

### Utils Module

The `utils` module contains common utility functions used across the project.

- **common.py**: Shared utility functions (hashing, path management, etc.)

## Usage Examples

### Importing from Modules

You can import specific functions or classes:

```python
from src.core import load_model, predict_pairwise_similarity
from src.clustering import cluster_dbscan, evaluate_clustering
from src.sanitization import GLiNERClassifier, run_with_batches
from src.storage import ProcessingDBManager
from src.utils import generate_config_hash
```

### Running the Sanitizer with Batch Processing

```python
from src.sanitization import run_with_batches

run_with_batches(
    input_csv="data/unique_names.csv",
    output_csv="output/sanitised_names.csv",
    report_path="output/sanitised_report.md",
    use_gliner=True,
    batch_size=5000,
    db_path="processing.duckdb",
    resume=True
)
```

### Managing the Database

```python
from src.storage import list_failed_batches, reset_failed_batches, show_stats

# Show processing statistics
show_stats("processing.duckdb")

# List failed batches
list_failed_batches("processing.duckdb")

# Reset failed batches to try again
reset_failed_batches("processing.duckdb")
```
