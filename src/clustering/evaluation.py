"""
Evaluation Module for Company Entity Resolution

This module provides functions to evaluate clustering quality
and compare different clustering approaches.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def calculate_cluster_metrics(
    cluster_df: pd.DataFrame,
    similarity_matrix: np.ndarray
) -> pd.DataFrame:
    """
    Calculate metrics for each cluster.
    
    Args:
        cluster_df: DataFrame with company and cluster_id columns
        similarity_matrix: Similarity matrix
        
    Returns:
        DataFrame with cluster metrics
    """
    # Get cluster sizes
    cluster_sizes = cluster_df['cluster_id'].value_counts()
    
    # Initialize metrics DataFrame
    metrics_df = pd.DataFrame({
        'cluster_id': cluster_sizes.index,
        'size': cluster_sizes.values
    })
    
    # Calculate metrics for each cluster
    avg_similarities = []
    min_similarities = []
    max_similarities = []
    
    for cluster_id in metrics_df['cluster_id']:
        # Skip noise cluster
        if cluster_id == -1:
            avg_similarities.append(np.nan)
            min_similarities.append(np.nan)
            max_similarities.append(np.nan)
            continue
        
        # Get indices of companies in this cluster
        indices = cluster_df[cluster_df['cluster_id'] == cluster_id].index.tolist()
        
        if len(indices) <= 1:
            # Single-element cluster
            avg_similarities.append(1.0)
            min_similarities.append(1.0)
            max_similarities.append(1.0)
        else:
            # Calculate pairwise similarities
            sims = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_i = indices[i]
                    idx_j = indices[j]
                    sims.append(similarity_matrix[idx_i, idx_j])
            
            avg_similarities.append(np.mean(sims))
            min_similarities.append(np.min(sims))
            max_similarities.append(np.max(sims))
    
    metrics_df['avg_similarity'] = avg_similarities
    metrics_df['min_similarity'] = min_similarities
    metrics_df['max_similarity'] = max_similarities
    
    return metrics_df.sort_values('size', ascending=False)

def visualize_similarity_distribution(
    similarity_matrix: np.ndarray,
    save_path: Optional[Path] = None
):
    """
    Visualize distribution of similarity scores.
    
    Args:
        similarity_matrix: Similarity matrix
        save_path: Path to save plot
    """
    # Get upper triangular values (excluding diagonal)
    similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(similarities, bins=50, alpha=0.7)
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pairwise Similarity Scores')
    
    # Add vertical lines for thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.5,
                   label=f'Threshold: {threshold}' if threshold == thresholds[0] else None)
    
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved similarity distribution plot to {save_path}")
    
    plt.show()

def compare_clustering_results(
    results: Dict[str, Dict],
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Compare different clustering results.
    
    Args:
        results: Dictionary mapping method name to results dict
        save_path: Path to save comparison
        
    Returns:
        DataFrame with comparison results
    """
    comparison = []
    
    for method_name, result in results.items():
        labels = result.get('labels')
        stats = result.get('stats', {})
        
        if labels is None:
            continue
        
        # Calculate basic metrics
        n_clusters = stats.get('n_clusters', 0)
        n_noise = stats.get('n_noise', 0)
        total = len(labels) if labels is not None else 0
        noise_ratio = n_noise / total if total > 0 else 0
        
        # Add to comparison
        comparison.append({
            'method': method_name,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'silhouette_score': stats.get('silhouette_score', np.nan)
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison)
    
    # Save if requested
    if save_path:
        comparison_df.to_csv(save_path, index=False)
        print(f"Saved comparison to {save_path}")
    
    return comparison_df
