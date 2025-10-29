"""
Core functionality for data processing, feature engineering, training, and inference.
"""

# Import key functions for easier access
from src.core.train import train_pipeline, load_model
from src.core.inference import (
    load_companies_from_excel,
    predict_pairwise_similarity,
    create_example_training_data
)
from src.core.features import extract_features
from src.core.preprocessing import preprocess_text

__all__ = [
    'train_pipeline',
    'load_model',
    'load_companies_from_excel',
    'predict_pairwise_similarity',
    'create_example_training_data',
    'extract_features',
    'preprocess_text'
]