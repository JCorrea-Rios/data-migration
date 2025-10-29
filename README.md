# Company Entity Resolution and Clustering

A conservative, explainable entity resolution system for clustering companies based on name similarity using LightGBM and string distance features.

## Overview

This project implements a complete pipeline for:
1. **Entity Resolution**: Identifying whether two company names refer to the same entity
2. **Company Clustering**: Grouping similar companies together based on pairwise similarities
3. **Explainable Features**: Using interpretable string distance metrics for transparency

## Features

### String Distance Metrics
- **Levenshtein Distance**: Character-level edit distance
- **Jaccard Similarity**: Character bigram overlap
- **Cosine Similarity**: Character frequency vector similarity
- **Hamming Distance**: Position-wise character matching
- **Damerau-Levenshtein**: Edit distance with transpositions
- **Editex**: Phonetic-aware edit distance

### Machine Learning
- **LightGBM Classifier**: Fast, accurate gradient boosting
- **Two-Stage Training**: Separate models for simple and hard negative examples
- **Calibrated Probabilities**: Reliable similarity scores for clustering

### Clustering Algorithms
- **DBSCAN**: Density-based clustering (auto-determines number of clusters)
- **Agglomerative**: Hierarchical clustering with distance threshold

## Project Structure

```
.
├── config.py              # Configuration parameters
├── main.py               # Main orchestration script
├── requirements.txt      # Python dependencies
├── src/
│   ├── features.py       # Feature engineering
│   ├── train.py          # Model training
│   ├── cluster.py        # Clustering algorithms
│   └── inference.py      # Prediction and inference
├── data/
│   ├── migration-analysis.xlsx         # Companies to cluster
│   ├── training_simple.pkl            # Simple training examples
│   └── training_hard_negatives.pkl    # Hard negative examples
├── models/               # Trained models
└── output/              # Results and visualizations
```

## Installation

### Using uv (recommended)

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install pandas openpyxl numpy scikit-learn lightgbm matplotlib seaborn textdistance jellyfish python-dotenv
```

### Using pip

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the full pipeline (training + clustering):

```bash
python main.py --mode full --companies data/migration-analysis.xlsx
```

### Step-by-Step Usage

#### 1. Prepare Training Data

You need two training datasets:
- **Simple negatives**: Basic matching/non-matching company pairs
- **Hard negatives**: Challenging pairs (similar names but different companies)

Format: Pickle file with columns `Company1`, `Company2`, `Label` (1=match, 0=no match)

Create example data:
```bash
python main.py --create-examples
```

Then replace the example files with your actual training data.

#### 2. Train Models

```bash
python main.py --mode train
```

This trains models on both simple and hard negative datasets and selects the best one.

#### 3. Cluster Companies

```bash
python main.py --mode cluster --companies data/migration-analysis.xlsx --algorithm both
```

Options:
- `--algorithm dbscan`: Use DBSCAN only
- `--algorithm agglomerative`: Use Agglomerative clustering only
- `--algorithm both`: Use both algorithms (default)

### Configuration

Edit `config.py` to adjust parameters:

```python
# Training
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# LGBM hyperparameters
LGBM_PARAMS = {
    'learning_rate': 0.05,
    'n_estimators': 100,
    'num_leaves': 31,
    # ...
}

# Clustering
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity for potential match
DBSCAN_EPS = 0.3           # DBSCAN distance threshold
DBSCAN_MIN_SAMPLES = 2     # Minimum cluster size
```

## Output

The pipeline generates several outputs in the `output/` directory:

1. **Cluster CSV files**:
   - `clusters_dbscan.csv`: DBSCAN clustering results
   - `clusters_agglomerative.csv`: Agglomerative clustering results
   
   Columns:
   - `company`: Company name
   - `cluster_id`: Assigned cluster ID
   - `cluster_size`: Number of companies in cluster
   - `avg_similarity_in_cluster`: Average similarity within cluster

2. **Visualizations**:
   - `clustering_dbscan.png`: DBSCAN cluster size distributions
   - `clustering_agglomerative.png`: Agglomerative cluster distributions
   - `feature_importance_simple.png`: Feature importance for simple model
   - `feature_importance_hard.png`: Feature importance for hard model

3. **Trained Models**:
   - `models/lgbm_simple.pkl`: Model trained on simple negatives
   - `models/lgbm_hard.pkl`: Model trained on hard negatives

## Methodology

### 1. Feature Engineering

For each company pair, we calculate 6 normalized similarity scores (0-1 range):

```python
features = [
    'levenshtein',       # Edit distance
    'jaccard',           # Bigram overlap
    'cosine',            # Character frequency similarity
    'hamming',           # Position-wise matching
    'damerau_levenshtein',  # Edit distance with transpositions
    'editex'             # Phonetic-aware distance
]
```

All features are normalized and **explainable** - you can trace why two companies were deemed similar.

### 2. Model Training

We use a **two-stage approach**:

1. **Simple Model**: Trained on clear-cut examples (e.g., "Microsoft Corp" vs "Microsoft Corporation")
2. **Hard Model**: Trained on challenging cases (e.g., "Delta Airlines" vs "Delta Corp")

The model with the highest F1 score on the validation set is selected.

### 3. Clustering

For clustering, we:
1. Generate all pairwise comparisons (N×(N-1)/2 pairs)
2. Predict similarity scores using the trained model
3. Build a similarity matrix
4. Apply clustering algorithms:
   - **DBSCAN**: Good for finding arbitrarily shaped clusters and handling noise
   - **Agglomerative**: Good for hierarchical relationships and flexible thresholds

## Evaluation Metrics

### Classification Metrics (Entity Resolution)
- Accuracy, Precision, Recall, F1 Score
- ROC-AUC
- Confusion Matrix

### Clustering Metrics
- Number of clusters found
- Cluster size distribution
- Silhouette score (cluster quality)
- Average intra-cluster similarity

## Limitations & Future Work

### Current Limitations
1. **Quadratic Complexity**: Pairwise comparison scales as O(N²)
   - For 1000 companies: ~500,000 comparisons
   - For 10,000 companies: ~50 million comparisons
   
2. **No Country/Geography Features**: Currently only uses company names

3. **No Abbreviation Dictionary**: Doesn't recognize domain-specific abbreviations

### Planned Improvements (from Plan)
1. **Neo4j Integration**: Store companies and similarities in a graph database
2. **PyTorch Geometric**: Graph neural networks for better clustering
3. **Active Learning**: Incorporate user feedback to improve model
4. **Blocking/Indexing**: Reduce pairwise comparisons using smart filtering

## Troubleshooting

### "No training data found"
Create training data files or use example data:
```bash
python main.py --create-examples
```

### "No companies found in Excel"
Ensure your Excel file has a column with "name", "company", or "client_name" in the header.

### Memory issues with large datasets
Reduce batch size in `config.py` or use incremental processing.

## Contributing

This is a prototype implementation. Suggested improvements:
- Add country-based features
- Implement abbreviation normalization
- Add industry/sector information
- Optimize for larger datasets
- Add API for real-time inference

## License

MIT License

## References

- LightGBM: https://lightgbm.readthedocs.io/
- TextDistance: https://github.com/life4/textdistance
- DBSCAN: https://scikit-learn.org/stable/modules/clustering.html#dbscan
- Agglomerative Clustering: https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

