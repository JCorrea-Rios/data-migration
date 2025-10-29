"""
Configuration for Company Entity Resolution Pipeline
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Data files
COMPANIES_FILE = DATA_DIR / "migration-analysis.xlsx"
UNIQUE_NAMES_FILE = DATA_DIR / "unique_names.csv"
COMMON_TERMS_FILE = DATA_DIR / "common_terms.txt"
TRAINING_SIMPLE_FILE = DATA_DIR / "training_simple.pkl"
TRAINING_HARD_FILE = DATA_DIR / "training_hard_negatives.pkl"
TRAINING_BINARY_FILE = DATA_DIR / "BinaryLabelled_CompanyMatching_Data.pkl"

# Model files
MODEL_SIMPLE = MODELS_DIR / "lgbm_simple.pkl"
MODEL_HARD = MODELS_DIR / "lgbm_hard.pkl"
MODEL_BINARY = MODELS_DIR / "lgbm_binary_labelled.pkl"
MODEL_ENSEMBLE = MODELS_DIR / "lgbm_ensemble.pkl"

# Training parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# LGBM hyperparameters
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'min_child_samples': 20,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_SEED,
    'verbose': -1
}

# Clustering parameters
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity to consider as potential match
DBSCAN_EPS = 0.3  # Distance threshold (1 - similarity)
DBSCAN_MIN_SAMPLES = 2  # Minimum samples to form a cluster
AGGLOMERATIVE_DISTANCE_THRESHOLD = 0.3  # Distance threshold for hierarchical clustering

# Feature columns
FEATURE_COLUMNS = [
    'levenshtein',
    'jaccard',
    'cosine',
    'hamming',
    'damerau_levenshtein',
    'editex'
]

# Enhanced feature columns
ENHANCED_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    'token_jaccard',
    'token_overlap',
    'core_name_similarity'
]

# Blocking parameters
BLOCKING_METHOD = 'letter'  # Currently only 'letter' is supported

# Output settings
SAVE_VISUALIZATIONS = True
SAMPLE_SIZE_FOR_REVIEW = 50  # Number of cluster examples to export for manual review

