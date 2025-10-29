"""
Inference Module for Company Entity Resolution

This module handles making predictions on new company pairs
and clustering new company lists.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from pathlib import Path
import pickle

from src.core.features import generate_distance_features, generate_pairwise_features
from src import config


def load_companies_from_excel(filepath: Path, sheet_name: str = None) -> List[str]:
    """
    Load company names from Excel file.
    
    Args:
        filepath: Path to Excel file
        sheet_name: Name of sheet to read (if None, reads all sheets)
        
    Returns:
        List of unique company names
    """
    print(f"Loading companies from {filepath}...")
    
    if sheet_name:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        sheets_data = {sheet_name: df}
    else:
        sheets_data = pd.read_excel(filepath, sheet_name=None)
    
    # Extract company names from all sheets
    all_companies = []
    
    for sheet, df in sheets_data.items():
        print(f"  Sheet '{sheet}': {len(df)} rows")
        
        # Look for column with company names
        # Try common column names
        company_col = None
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'client_name' in col_lower or 'company' in col_lower or 'name' in col_lower:
                company_col = col
                break
        
        if company_col:
            companies = df[company_col].dropna().astype(str).str.strip().tolist()
            all_companies.extend(companies)
            print(f"    Found {len(companies)} companies in column '{company_col}'")
        else:
            print(f"    No company name column found")
    
    # Remove duplicates and empty strings
    unique_companies = list(set([c for c in all_companies if c and c.strip()]))
    unique_companies.sort()
    
    print(f"\nTotal unique companies: {len(unique_companies)}")
    
    return unique_companies


def predict_company_pairs(
    model,
    company_pairs: pd.DataFrame,
    col1: str = 'Company1',
    col2: str = 'Company2',
    threshold: float = None
) -> pd.DataFrame:
    """
    Predict whether company pairs are matches.
    
    Args:
        model: Trained model
        company_pairs: DataFrame with company pairs
        col1: Name of first company column
        col2: Name of second company column
        threshold: Similarity threshold for binary classification
        
    Returns:
        DataFrame with predictions
    """
    if threshold is None:
        threshold = config.SIMILARITY_THRESHOLD
    
    print(f"Predicting {len(company_pairs)} company pairs...")
    
    # Generate features
    df_with_features, feature_cols = generate_distance_features(
        company_pairs, col1, col2
    )
    
    # Make predictions
    X = df_with_features[feature_cols]
    predictions = model.predict(X)
    prediction_probas = model.predict_proba(X)[:, 1]
    
    # Add to dataframe
    df_with_features['prediction'] = predictions
    df_with_features['prediction_proba'] = prediction_probas
    df_with_features['is_match'] = prediction_probas >= threshold
    
    return df_with_features


def predict_pairwise_similarity(
    model,
    companies: List[str],
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    Predict similarity for all pairs of companies.
    
    This generates a complete pairwise comparison which can be used
    for clustering.
    
    Args:
        model: Trained model
        companies: List of company names
        batch_size: Batch size for processing
        
    Returns:
        DataFrame with all pairwise predictions
    """
    print(f"\nGenerating pairwise predictions for {len(companies)} companies...")
    
    # Generate all pairwise features
    pairwise_df = generate_pairwise_features(companies, batch_size)
    
    # Get feature columns (excluding company names and indices)
    feature_cols = config.FEATURE_COLUMNS
    
    # Make predictions
    print("Making predictions on pairwise comparisons...")
    X = pairwise_df[feature_cols]
    
    # Predict in batches to manage memory
    all_probas = []
    total_batches = (len(X) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(X))
        X_batch = X.iloc[start_idx:end_idx]
        
        probas = model.predict_proba(X_batch)[:, 1]
        all_probas.extend(probas)
        
        print(f"Batch {batch_idx + 1}/{total_batches} completed", end='\r')
    
    print(f"\nCompleted all predictions")
    
    pairwise_df['prediction_proba'] = all_probas
    pairwise_df['prediction'] = (np.array(all_probas) >= config.SIMILARITY_THRESHOLD).astype(int)
    
    return pairwise_df


def find_similar_companies(
    pairwise_df: pd.DataFrame,
    company_name: str,
    companies_list: List[str],
    top_n: int = 10,
    min_similarity: float = None
) -> pd.DataFrame:
    """
    Find the most similar companies to a given company.
    
    Args:
        pairwise_df: DataFrame with pairwise predictions
        company_name: Target company name
        companies_list: List of all companies
        top_n: Number of top similar companies to return
        min_similarity: Minimum similarity threshold
        
    Returns:
        DataFrame with similar companies sorted by similarity
    """
    if min_similarity is None:
        min_similarity = config.SIMILARITY_THRESHOLD
    
    try:
        company_idx = companies_list.index(company_name)
    except ValueError:
        print(f"Company '{company_name}' not found in list")
        return pd.DataFrame()
    
    # Find all comparisons involving this company
    similar = pairwise_df[
        ((pairwise_df['idx1'] == company_idx) | (pairwise_df['idx2'] == company_idx)) &
        (pairwise_df['prediction_proba'] >= min_similarity)
    ].copy()
    
    # Get the other company in each pair
    similar['other_company'] = similar.apply(
        lambda row: row['Company2'] if row['idx1'] == company_idx else row['Company1'],
        axis=1
    )
    
    # Sort by similarity
    similar = similar.sort_values('prediction_proba', ascending=False).head(top_n)
    
    result = similar[['other_company', 'prediction_proba']].copy()
    result.columns = ['similar_company', 'similarity_score']
    
    return result.reset_index(drop=True)


def create_example_training_data():
    """
    Create example training data files if they don't exist.
    This is a placeholder - user should provide their own training data.
    """
    print("\nCreating example training data...")
    
    # Example data with some matching and non-matching pairs
    simple_examples = {
        'Company1': [
            'Microsoft Corporation', 'Apple Inc', 'Google LLC', 'Amazon.com Inc',
            'Tesla Inc', 'Meta Platforms', 'Netflix Inc', 'Adobe Inc',
            'Microsoft', 'Apple', 'Random Corp', 'Different Company'
        ],
        'Company2': [
            'Microsoft Corp', 'Apple Incorporated', 'Google', 'Amazon',
            'Tesla Motors', 'Facebook Inc', 'Netflix', 'Adobe Systems',
            'Apple Inc', 'Google LLC', 'Microsoft Corp', 'Amazon.com'
        ],
        'Label': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    }
    
    simple_df = pd.DataFrame(simple_examples)
    simple_df.to_pickle(config.TRAINING_SIMPLE_FILE)
    print(f"Created example simple training data: {config.TRAINING_SIMPLE_FILE}")
    print(f"  {len(simple_df)} examples")
    
    # Hard negatives - similar looking but different companies
    hard_examples = {
        'Company1': [
            'Delta Airlines', 'Bank of America', 'Ford Motor Company',
            'Delta Corp', 'American Bank', 'Ford Motors India'
        ],
        'Company2': [
            'Delta Air Lines', 'Bank of America Corp', 'Ford Motor Co',
            'Delta Airlines', 'Bank of America', 'Ford Motor Company'
        ],
        'Label': [1, 1, 1, 0, 0, 0]
    }
    
    hard_df = pd.DataFrame(hard_examples)
    hard_df.to_pickle(config.TRAINING_HARD_FILE)
    print(f"Created example hard negatives data: {config.TRAINING_HARD_FILE}")
    print(f"  {len(hard_df)} examples")
    
    print("\nNOTE: These are just examples. Replace with your actual training data!")

