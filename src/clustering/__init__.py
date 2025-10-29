"""
Clustering algorithms and evaluation metrics for company entity resolution.
"""

# Import key functions for easier access
from src.clustering.cluster import (
    build_similarity_matrix,
    cluster_dbscan,
    cluster_agglomerative,
    analyze_clusters,
    visualize_clusters,
    export_clusters
)
from src.clustering.evaluation import evaluate_clustering
from src.clustering.blocking import apply_blocking

__all__ = [
    'build_similarity_matrix',
    'cluster_dbscan',
    'cluster_agglomerative',
    'analyze_clusters',
    'visualize_clusters',
    'export_clusters',
    'evaluate_clustering',
    'apply_blocking'
]