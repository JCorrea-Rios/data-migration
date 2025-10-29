"""
Feature Engineering Module for Company Entity Resolution

This module provides functions to generate string distance features
for comparing company names. All features are normalized to [0,1] range
where 1 indicates high similarity.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import textdistance
import warnings
warnings.filterwarnings('ignore')

from src.preprocessing import normalize_company_name, tokenize_company_name, extract_company_core_name

# Define feature columns constant
FEATURE_COLUMNS = [
    'levenshtein',
    'jaccard',
    'cosine',
    'hamming',
    'damerau_levenshtein',
    'editex'
]

# Define enhanced feature columns with token-based features
ENHANCED_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    'token_jaccard',
    'token_overlap',
    'core_name_similarity'
]


# Note: normalize_company_name is now imported from src.preprocessing


def calculate_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate normalized Levenshtein similarity (0=different, 1=identical).
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    s1_norm = normalize_company_name(s1)
    s2_norm = normalize_company_name(s2)
    
    if not s1_norm or not s2_norm:
        return 0.0
    
    # Calculate normalized similarity
    distance = textdistance.levenshtein.distance(s1_norm, s2_norm)
    max_len = max(len(s1_norm), len(s2_norm))
    
    if max_len == 0:
        return 1.0
    
    similarity = 1 - (distance / max_len)
    return max(0.0, min(1.0, similarity))


def calculate_jaccard_similarity(s1: str, s2: str) -> float:
    """
    Calculate Jaccard similarity using character bigrams.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    s1_norm = normalize_company_name(s1)
    s2_norm = normalize_company_name(s2)
    
    if not s1_norm or not s2_norm:
        return 0.0
    
    # Create bigrams
    bigrams1 = set([s1_norm[i:i+2] for i in range(len(s1_norm)-1)])
    bigrams2 = set([s2_norm[i:i+2] for i in range(len(s2_norm)-1)])
    
    if not bigrams1 and not bigrams2:
        return 1.0
    if not bigrams1 or not bigrams2:
        return 0.0
    
    intersection = len(bigrams1.intersection(bigrams2))
    union = len(bigrams1.union(bigrams2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_cosine_similarity(s1: str, s2: str) -> float:
    """
    Calculate cosine similarity using character frequency vectors.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    s1_norm = normalize_company_name(s1)
    s2_norm = normalize_company_name(s2)
    
    if not s1_norm or not s2_norm:
        return 0.0
    
    # Create character frequency vectors
    chars = set(s1_norm + s2_norm)
    vec1 = np.array([s1_norm.count(c) for c in chars])
    vec2 = np.array([s2_norm.count(c) for c in chars])
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return max(0.0, min(1.0, similarity))


def calculate_hamming_similarity(s1: str, s2: str) -> float:
    """
    Calculate normalized Hamming similarity.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    s1_norm = normalize_company_name(s1)
    s2_norm = normalize_company_name(s2)
    
    if not s1_norm or not s2_norm:
        return 0.0
    
    # Pad shorter string
    max_len = max(len(s1_norm), len(s2_norm))
    s1_padded = s1_norm.ljust(max_len)
    s2_padded = s2_norm.ljust(max_len)
    
    # Count matching positions
    matches = sum(c1 == c2 for c1, c2 in zip(s1_padded, s2_padded))
    
    return matches / max_len if max_len > 0 else 0.0


def calculate_damerau_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate normalized Damerau-Levenshtein similarity.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    s1_norm = normalize_company_name(s1)
    s2_norm = normalize_company_name(s2)
    
    if not s1_norm or not s2_norm:
        return 0.0
    
    distance = textdistance.damerau_levenshtein.distance(s1_norm, s2_norm)
    max_len = max(len(s1_norm), len(s2_norm))
    
    if max_len == 0:
        return 1.0
    
    similarity = 1 - (distance / max_len)
    return max(0.0, min(1.0, similarity))


def calculate_editex_similarity(s1: str, s2: str) -> float:
    """
    Calculate normalized Editex similarity (phonetic-aware edit distance).
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    s1_norm = normalize_company_name(s1)
    s2_norm = normalize_company_name(s2)
    
    if not s1_norm or not s2_norm:
        return 0.0
    
    try:
        distance = textdistance.editex.distance(s1_norm, s2_norm)
        max_len = max(len(s1_norm), len(s2_norm))
        
        if max_len == 0:
            return 1.0
        
        # Editex can have larger distances, so we normalize differently
        similarity = 1 - min(distance / (max_len * 2), 1.0)
        return max(0.0, min(1.0, similarity))
    except:
        # Fallback to levenshtein if editex fails
        return calculate_levenshtein_similarity(s1, s2)


def calculate_token_jaccard_similarity(s1: str, s2: str) -> float:
    """
    Calculate Jaccard similarity using word tokens.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    tokens1 = set(tokenize_company_name(s1))
    tokens2 = set(tokenize_company_name(s2))
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    return intersection / union if union > 0 else 0.0

def calculate_token_overlap(s1: str, s2: str) -> float:
    """
    Calculate token overlap ratio (intersection/min length).
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    tokens1 = set(tokenize_company_name(s1))
    tokens2 = set(tokenize_company_name(s2))
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1.intersection(tokens2))
    min_length = min(len(tokens1), len(tokens2))
    
    return intersection / min_length if min_length > 0 else 0.0

def calculate_core_name_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity between core company names (without legal suffixes).
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    core1 = extract_company_core_name(s1)
    core2 = extract_company_core_name(s2)
    
    return calculate_levenshtein_similarity(core1, core2)

def generate_distance_features(
    df: pd.DataFrame,
    col1: str = 'Company1',
    col2: str = 'Company2'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate string distance features for company name pairs.
    
    This function creates multiple similarity metrics that are interpretable
    and proven effective for entity resolution tasks. All features are
    normalized to [0,1] where 1 indicates high similarity.
    
    Args:
        df: DataFrame containing company name pairs
        col1: Name of first company column
        col2: Name of second company column
        
    Returns:
        Tuple of (DataFrame with added features, list of feature column names)
        
    Feature Descriptions:
        - levenshtein: Edit distance (character insertions/deletions/substitutions)
        - jaccard: Similarity of character bigram sets
        - cosine: Similarity of character frequency vectors
        - hamming: Matching characters at same positions
        - damerau_levenshtein: Edit distance allowing transpositions
        - editex: Phonetic-aware edit distance
    """
    df_copy = df.copy()
    
    print("Generating distance features...")
    
    # Calculate all features
    df_copy['levenshtein'] = df_copy.apply(
        lambda row: calculate_levenshtein_similarity(row[col1], row[col2]), axis=1
    )
    
    df_copy['jaccard'] = df_copy.apply(
        lambda row: calculate_jaccard_similarity(row[col1], row[col2]), axis=1
    )
    
    df_copy['cosine'] = df_copy.apply(
        lambda row: calculate_cosine_similarity(row[col1], row[col2]), axis=1
    )
    
    df_copy['hamming'] = df_copy.apply(
        lambda row: calculate_hamming_similarity(row[col1], row[col2]), axis=1
    )
    
    df_copy['damerau_levenshtein'] = df_copy.apply(
        lambda row: calculate_damerau_levenshtein_similarity(row[col1], row[col2]), axis=1
    )
    
    df_copy['editex'] = df_copy.apply(
        lambda row: calculate_editex_similarity(row[col1], row[col2]), axis=1
    )
    
    # Add token-based features
    df_copy['token_jaccard'] = df_copy.apply(
        lambda row: calculate_token_jaccard_similarity(row[col1], row[col2]), axis=1
    )
    
    df_copy['token_overlap'] = df_copy.apply(
        lambda row: calculate_token_overlap(row[col1], row[col2]), axis=1
    )
    
    df_copy['core_name_similarity'] = df_copy.apply(
        lambda row: calculate_core_name_similarity(row[col1], row[col2]), axis=1
    )
    
    feature_cols = FEATURE_COLUMNS
    enhanced_feature_cols = ENHANCED_FEATURE_COLUMNS
    
    print(f"Generated {len(feature_cols)} features: {feature_cols}")
    
    return df_copy, feature_cols


def generate_pairwise_features(
    companies: List[str],
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    Generate pairwise features for all company combinations.
    
    This is used for clustering inference to compare all companies
    against each other.
    
    Args:
        companies: List of company names
        batch_size: Process in batches to manage memory
        
    Returns:
        DataFrame with all pairwise comparisons and their features
    """
    print(f"Generating pairwise features for {len(companies)} companies...")
    print(f"This will create {len(companies) * (len(companies) - 1) // 2} comparisons")
    
    pairs = []
    
    # Create all unique pairs (excluding self-comparisons)
    for i in range(len(companies)):
        for j in range(i + 1, len(companies)):
            pairs.append({
                'Company1': companies[i],
                'Company2': companies[j],
                'idx1': i,
                'idx2': j
            })
    
    # Convert to DataFrame
    pairs_df = pd.DataFrame(pairs)
    
    # Generate features in batches
    all_results = []
    total_batches = (len(pairs_df) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(pairs_df))
        batch_df = pairs_df.iloc[start_idx:end_idx].copy()
        
        print(f"Processing batch {batch_idx + 1}/{total_batches}...", end='\r')
        
        # Generate features for this batch
        batch_with_features, _ = generate_distance_features(batch_df, 'Company1', 'Company2')
        all_results.append(batch_with_features)
    
    print(f"\nCompleted pairwise feature generation")
    
    return pd.concat(all_results, ignore_index=True)

