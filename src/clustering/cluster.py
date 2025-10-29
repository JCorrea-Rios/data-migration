"""
Clustering Module for Company Entity Resolution

This module implements clustering algorithms to group similar companies
based on their pairwise similarity scores.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src import config
from src.clustering.blocking import apply_blocking


def build_similarity_matrix(
    pairwise_df: pd.DataFrame,
    n_companies: int
) -> np.ndarray:
    """
    Build a symmetric similarity matrix from pairwise comparisons.
    
    Args:
        pairwise_df: DataFrame with idx1, idx2, and prediction_proba columns
        n_companies: Total number of companies
        
    Returns:
        Symmetric similarity matrix
    """
    print(f"Building {n_companies}x{n_companies} similarity matrix...")
    
    # Initialize matrix with diagonal as 1 (company is identical to itself)
    similarity_matrix = np.eye(n_companies)
    
    # Fill in pairwise similarities
    for _, row in pairwise_df.iterrows():
        i = int(row['idx1'])
        j = int(row['idx2'])
        sim = row['prediction_proba']
        
        # Matrix is symmetric
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim
    
    return similarity_matrix


def similarity_to_distance(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Convert similarity matrix to distance matrix.
    
    Args:
        similarity_matrix: Matrix with values in [0, 1] where 1 = most similar
        
    Returns:
        Distance matrix where 0 = most similar
    """
    return 1 - similarity_matrix


def cluster_dbscan(
    similarity_matrix: np.ndarray,
    eps: float = None,
    min_samples: int = None
) -> Tuple[np.ndarray, Dict]:
    """
    Cluster companies using DBSCAN algorithm.
    
    DBSCAN is density-based and automatically determines the number of clusters.
    It can also identify noise points (outliers).
    
    Args:
        similarity_matrix: Symmetric similarity matrix
        eps: Maximum distance between samples (if None, use config)
        min_samples: Minimum samples to form a cluster (if None, use config)
        
    Returns:
        Tuple of (cluster labels, statistics dict)
    """
    if eps is None:
        eps = config.DBSCAN_EPS
    if min_samples is None:
        min_samples = config.DBSCAN_MIN_SAMPLES
    
    print(f"\nClustering with DBSCAN (eps={eps}, min_samples={min_samples})...")
    
    # Convert similarity to distance
    distance_matrix = similarity_to_distance(similarity_matrix)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(distance_matrix)
    
    # Calculate statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    stats = {
        'algorithm': 'DBSCAN',
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'eps': eps,
        'min_samples': min_samples
    }
    
    print(f"Found {n_clusters} clusters and {n_noise} noise points")
    
    # Calculate silhouette score (if there are at least 2 clusters)
    if n_clusters > 1:
        # Exclude noise points for silhouette score
        mask = labels != -1
        if mask.sum() > 0:
            try:
                silhouette = silhouette_score(
                    distance_matrix[mask][:, mask],
                    labels[mask],
                    metric='precomputed'
                )
                stats['silhouette_score'] = silhouette
                print(f"Silhouette Score: {silhouette:.4f}")
            except:
                stats['silhouette_score'] = None
    
    return labels, stats


def cluster_agglomerative(
    similarity_matrix: np.ndarray,
    distance_threshold: float = None,
    n_clusters: Optional[int] = None,
    linkage: str = 'average'
) -> Tuple[np.ndarray, Dict]:
    """
    Cluster companies using Agglomerative (Hierarchical) Clustering.
    
    This algorithm builds a hierarchy of clusters. You can either specify
    the number of clusters or a distance threshold.
    
    Args:
        similarity_matrix: Symmetric similarity matrix
        distance_threshold: Distance threshold for cutting the tree (if None, use config)
        n_clusters: Number of clusters (if specified, distance_threshold is ignored)
        linkage: Linkage criterion ('average', 'complete', 'single')
        
    Returns:
        Tuple of (cluster labels, statistics dict)
    """
    if distance_threshold is None and n_clusters is None:
        distance_threshold = config.AGGLOMERATIVE_DISTANCE_THRESHOLD
    
    print(f"\nClustering with Agglomerative Clustering...")
    print(f"  Linkage: {linkage}")
    if n_clusters:
        print(f"  Number of clusters: {n_clusters}")
    else:
        print(f"  Distance threshold: {distance_threshold}")
    
    # Convert similarity to distance
    distance_matrix = similarity_to_distance(similarity_matrix)
    
    # Apply Agglomerative Clustering
    if n_clusters:
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage=linkage
        )
    else:
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage=linkage
        )
    
    labels = agg.fit_predict(distance_matrix)
    
    # Calculate statistics
    n_clusters_found = len(set(labels))
    
    stats = {
        'algorithm': 'Agglomerative',
        'n_clusters': n_clusters_found,
        'distance_threshold': distance_threshold,
        'linkage': linkage
    }
    
    print(f"Found {n_clusters_found} clusters")
    
    # Calculate silhouette score (if there are at least 2 clusters)
    if n_clusters_found > 1 and n_clusters_found < len(labels):
        try:
            silhouette = silhouette_score(
                distance_matrix,
                labels,
                metric='precomputed'
            )
            stats['silhouette_score'] = silhouette
            print(f"Silhouette Score: {silhouette:.4f}")
        except:
            stats['silhouette_score'] = None
    
    return labels, stats


def connected_components_clustering(
    similarity_matrix: np.ndarray,
    threshold: float = 0.7
) -> Tuple[np.ndarray, Dict]:
    """
    Cluster using connected components on thresholded similarity graph.
    
    Args:
        similarity_matrix: Symmetric similarity matrix
        threshold: Similarity threshold for edge creation
        
    Returns:
        Tuple of (cluster labels, statistics dict)
    """
    print(f"\nClustering with Connected Components (threshold={threshold})...")
    
    # For large matrices, create sparse adjacency matrix directly
    n = similarity_matrix.shape[0]
    rows = []
    cols = []
    data = []
    
    # Only iterate through upper triangle to save memory
    print("Creating sparse adjacency matrix...")
    for i in range(n):
        # Add non-zero elements from upper triangle
        for j in range(i+1, n):
            if similarity_matrix[i, j] >= threshold:
                rows.append(i)
                cols.append(j)
                data.append(1)
                
                # Add symmetric element
                rows.append(j)
                cols.append(i)
                data.append(1)
        
        # Print progress for large matrices
        if n > 1000 and i % 1000 == 0:
            print(f"  Processed {i}/{n} rows ({i/n*100:.1f}%)")
    
    # Create sparse matrix
    sparse_adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # Find connected components
    print("Finding connected components...")
    n_components, labels = connected_components(sparse_adjacency, directed=False)
    
    # Calculate statistics
    stats = {
        'algorithm': 'ConnectedComponents',
        'n_clusters': n_components,
        'threshold': threshold
    }
    
    print(f"Found {n_components} clusters")
    
    return labels, stats

def two_phase_clustering(
    companies: List[str],
    similarity_matrix: np.ndarray,
    eps: float = 0.3,
    min_samples: int = 2
) -> Tuple[np.ndarray, Dict]:
    """
    Two-phase clustering with blocking and DBSCAN.
    
    Args:
        companies: List of company names
        similarity_matrix: Similarity matrix
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
        
    Returns:
        Tuple of (cluster labels, statistics dict)
    """
    print(f"\nApplying Two-Phase Clustering with DBSCAN...")
    print(f"  Blocking method: {config.BLOCKING_METHOD}")
    print(f"  DBSCAN parameters: eps={eps}, min_samples={min_samples}")
    
    # Initialize all companies as noise
    labels = np.full(len(companies), -1)
    
    # Group companies by first letter
    from src.preprocessing import get_company_first_letter
    letter_groups = {}
    for idx, company in enumerate(companies):
        letter = get_company_first_letter(company)
        if letter not in letter_groups:
            letter_groups[letter] = []
        letter_groups[letter].append(idx)
    
    # Apply DBSCAN to each letter group
    next_cluster_id = 0
    for letter, indices in letter_groups.items():
        if len(indices) < 2:
            continue
            
        # Extract submatrix for this group
        submatrix = similarity_matrix[np.ix_(indices, indices)]
        
        # Convert to distance matrix
        distance_matrix = 1 - submatrix
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        sub_labels = dbscan.fit_predict(distance_matrix)
        
        # Map back to original indices, skipping noise points
        for i, sub_label in enumerate(sub_labels):
            if sub_label != -1:
                labels[indices[i]] = next_cluster_id + sub_label
        
        # Update next cluster ID
        if len(set(sub_labels) - {-1}) > 0:  # If we found any clusters
            next_cluster_id += max(sub_labels) + 1
    
    # Calculate statistics
    n_clusters = len(set(labels) - {-1})
    n_noise = list(labels).count(-1)
    
    stats = {
        'algorithm': 'TwoPhase_DBSCAN',
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'eps': eps,
        'min_samples': min_samples
    }
    
    print(f"Found {n_clusters} clusters and {n_noise} noise points")
    
    return labels, stats

def analyze_clusters(
    companies: List[str],
    labels: np.ndarray,
    similarity_matrix: np.ndarray,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Analyze cluster composition and quality.
    
    Args:
        companies: List of company names
        labels: Cluster labels
        similarity_matrix: Similarity matrix
        top_n: Number of top clusters to show details
        
    Returns:
        DataFrame with cluster analysis
    """
    print(f"\nAnalyzing clusters...")
    
    # Create DataFrame
    cluster_df = pd.DataFrame({
        'company': companies,
        'cluster_id': labels
    })
    
    # Calculate cluster sizes
    cluster_sizes = cluster_df['cluster_id'].value_counts().sort_values(ascending=False)
    
    print(f"\nCluster size distribution:")
    print(f"  Mean: {cluster_sizes.mean():.2f}")
    print(f"  Median: {cluster_sizes.median():.0f}")
    print(f"  Max: {cluster_sizes.max()}")
    print(f"  Min: {cluster_sizes.min()}")
    
    # Show top N largest clusters
    print(f"\nTop {min(top_n, len(cluster_sizes))} largest clusters:")
    for cluster_id, size in cluster_sizes.head(top_n).items():
        print(f"\nCluster {cluster_id} ({size} companies):")
        cluster_companies = cluster_df[cluster_df['cluster_id'] == cluster_id]['company'].tolist()
        
        # Show first few companies
        for comp in cluster_companies[:5]:
            print(f"  - {comp}")
        if size > 5:
            print(f"  ... and {size - 5} more")
        
        # Calculate average intra-cluster similarity
        cluster_indices = cluster_df[cluster_df['cluster_id'] == cluster_id].index.tolist()
        if len(cluster_indices) > 1:
            sims = []
            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    idx_i = cluster_indices[i]
                    idx_j = cluster_indices[j]
                    sims.append(similarity_matrix[idx_i, idx_j])
            avg_sim = np.mean(sims)
            print(f"  Average intra-cluster similarity: {avg_sim:.4f}")
    
    return cluster_df


def visualize_clusters(
    cluster_df: pd.DataFrame,
    stats: Dict,
    save_path: Optional[Path] = None
):
    """
    Create visualizations of clustering results.
    
    Args:
        cluster_df: DataFrame with company and cluster_id columns
        stats: Statistics dictionary from clustering
        save_path: Path to save plot
    """
    print("\nGenerating cluster visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Cluster size distribution
    cluster_sizes = cluster_df['cluster_id'].value_counts().sort_values(ascending=False)
    
    ax1 = axes[0]
    ax1.bar(range(len(cluster_sizes)), cluster_sizes.values)
    ax1.set_xlabel('Cluster (sorted by size)')
    ax1.set_ylabel('Number of Companies')
    ax1.set_title(f'Cluster Size Distribution\n({stats["algorithm"]}, {stats["n_clusters"]} clusters)')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Cluster size histogram
    ax2 = axes[1]
    ax2.hist(cluster_sizes.values, bins=min(20, len(cluster_sizes)), edgecolor='black')
    ax2.set_xlabel('Cluster Size')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Cluster Sizes')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def export_clusters(
    cluster_df: pd.DataFrame,
    save_path: Path,
    similarity_matrix: Optional[np.ndarray] = None
):
    """
    Export clustering results to CSV.
    
    Args:
        cluster_df: DataFrame with clustering results
        save_path: Path to save CSV
        similarity_matrix: Optional similarity matrix to include confidence scores
    """
    print(f"\nExporting clusters to {save_path}...")
    
    # Sort by cluster ID and company name
    export_df = cluster_df.sort_values(['cluster_id', 'company']).reset_index(drop=True)
    
    # Add cluster size information
    cluster_sizes = export_df['cluster_id'].value_counts()
    export_df['cluster_size'] = export_df['cluster_id'].map(cluster_sizes)
    
    # If similarity matrix provided, add average similarity within cluster
    if similarity_matrix is not None:
        avg_sims = []
        for idx, row in export_df.iterrows():
            cluster_id = row['cluster_id']
            cluster_indices = export_df[export_df['cluster_id'] == cluster_id].index.tolist()
            
            if len(cluster_indices) > 1:
                sims = []
                my_idx = idx
                for other_idx in cluster_indices:
                    if other_idx != my_idx:
                        sims.append(similarity_matrix[my_idx, other_idx])
                avg_sim = np.mean(sims) if sims else 1.0
            else:
                avg_sim = 1.0  # Singleton cluster
            
            avg_sims.append(avg_sim)
        
        export_df['avg_similarity_in_cluster'] = avg_sims
    
    # Save to CSV
    export_df.to_csv(save_path, index=False)
    print(f"Exported {len(export_df)} companies in {export_df['cluster_id'].nunique()} clusters")

