"""
Model Training Pipeline for Company Entity Resolution
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.features import generate_distance_features, FEATURE_COLUMNS
from src import config


def load_training_data(filepath: Path, label_col: str = 'Label') -> pd.DataFrame:
    """
    Load training data from pickle file.
    
    Args:
        filepath: Path to training data file
        label_col: Name of label column
        
    Returns:
        DataFrame with training data
    """
    print(f"Loading training data from {filepath}...")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Training data not found: {filepath}")
    
    df = pd.read_pickle(filepath)
    print(f"Loaded {len(df)} training examples")
    
    # Check if data has required columns
    if 'Company1' not in df.columns or 'Company2' not in df.columns:
        raise ValueError("Training data must have 'Company1' and 'Company2' columns")
    
    if label_col not in df.columns:
        raise ValueError(f"Training data must have '{label_col}' column")
    
    # Display label distribution
    print("\nLabel distribution:")
    print(df[label_col].value_counts())
    print(f"Positive rate: {df[label_col].mean():.2%}")
    
    return df


def prepare_training_data(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: str = 'Label',
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        label_col: Name of label column
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test split)
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("\nSplitting data into train/val/test...")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df[label_col]
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for the reduced dataset
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val_df[label_col]
    )
    
    print(f"Train set: {len(train_df)} examples ({len(train_df)/len(df):.1%})")
    print(f"Validation set: {len(val_df)} examples ({len(val_df)/len(df):.1%})")
    print(f"Test set: {len(test_df)} examples ({len(test_df)/len(df):.1%})")
    
    return train_df, val_df, test_df


def train_lgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict = None
) -> lgb.LGBMClassifier:
    """
    Train LightGBM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: LightGBM parameters
        
    Returns:
        Trained LightGBM model
    """
    if params is None:
        params = config.LGBM_PARAMS.copy()
    
    print("\nTraining LightGBM model...")
    print(f"Parameters: {params}")
    
    # Create model
    model = lgb.LGBMClassifier(**params)
    
    # Train with validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    print(f"Training completed. Best iteration: {model.best_iteration_}")
    
    return model


def evaluate_model(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "Test"
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        dataset_name: Name of dataset for printing
        
    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluation on {dataset_name} Set")
    print(f"{'='*60}")
    
    # Get predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba)
    }
    
    # Print metrics
    for metric_name, value in metrics.items():
        print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Not Match', 'Match']))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return metrics


def plot_feature_importance(
    model: lgb.LGBMClassifier,
    feature_names: list,
    save_path: Optional[Path] = None
):
    """
    Plot and save feature importance.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        save_path: Path to save plot (optional)
    """
    print("\nGenerating feature importance plot...")
    
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.title('Feature Importance for Company Matching')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")
    
    plt.show()
    
    # Print feature importance
    print("\nFeature Importance:")
    for _, row in feature_importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")


def save_model(model: lgb.LGBMClassifier, filepath: Path):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save model
    """
    print(f"\nSaving model to {filepath}...")
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully")


def load_model(filepath: Path) -> lgb.LGBMClassifier:
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model
    """
    print(f"Loading model from {filepath}...")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print("Model loaded successfully")
    return model


def train_pipeline(
    training_file: Path,
    model_save_path: Path,
    dataset_name: str = "Simple"
) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    """
    Complete training pipeline.
    
    Args:
        training_file: Path to training data
        model_save_path: Path to save trained model
        dataset_name: Name of dataset for logging
        
    Returns:
        Tuple of (trained model, test metrics)
    """
    print(f"\n{'='*60}")
    print(f"Training Pipeline - {dataset_name} Dataset")
    print(f"{'='*60}\n")
    
    # Load data
    df = load_training_data(training_file)
    
    # Generate features
    df_with_features, feature_cols = generate_distance_features(df, 'Company1', 'Company2')
    
    # Split data
    train_df, val_df, test_df = prepare_training_data(
        df_with_features,
        feature_cols,
        test_size=config.TEST_SIZE,
        val_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_SEED
    )
    
    # Prepare feature matrices
    X_train = train_df[feature_cols]
    y_train = train_df['Label']
    X_val = val_df[feature_cols]
    y_val = val_df['Label']
    X_test = test_df[feature_cols]
    y_test = test_df['Label']
    
    # Train model
    model = train_lgbm_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on validation set
    evaluate_model(model, X_val, y_val, "Validation")
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Plot feature importance
    plot_path = config.OUTPUT_DIR / f"feature_importance_{dataset_name.lower()}.png"
    plot_feature_importance(model, feature_cols, plot_path)
    
    # Save model
    save_model(model, model_save_path)
    
    return model, test_metrics

