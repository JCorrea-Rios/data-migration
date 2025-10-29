<!-- e69af860-5e7d-4649-8c8c-371bc51ac2e5 31e5da9c-8e7c-4660-a85d-1f0669fd968d -->
# Company Entity Resolution Prototype Plan

## Objective

Create a conservative, explainable entity resolution system to cluster companies from `migration-analysis.xlsx`, leveraging existing LGBM models with string-distance features.

## Approach Rationale

- **LGBM over Deep Learning**: More explainable, faster inference, easier to debug
- **String Distance Features**: Well-understood, interpretable, proven effective in notebooks
- **Two-Stage Training**: Simple negatives first (baseline), then hard negatives (robustness)
- **Similarity-Based Clustering**: Convert pairwise similarity to clusters using DBSCAN/Agglomerative

## Implementation Steps

### 1. Data Pipeline Setup

- Load and inspect `data/migration-analysis.xlsx` structure
- Create data loading utilities for:
  - Companies to cluster (from Excel)
  - Training data with simple negatives
  - Training data with hard negatives
- Validate data formats and handle missing values

### 2. Feature Engineering Module

- Extract and modularize feature generation from `lgbm.ipynb`
- Use existing explainable features:
  - Levenshtein distance
  - Jaccard similarity
  - Cosine similarity
  - Hamming distance
  - Damerau-Levenshtein distance
  - Editex distance
- Add optional country-based features if country data available
- Create feature documentation for explainability

### 3. Model Training Pipeline

- Train LGBM model on simple negatives dataset first (baseline)
- Evaluate on validation set
- Train separate LGBM model on hard negatives dataset
- Compare performance between models
- Select best model or ensemble approach
- Focus on outputting calibrated similarity scores (0-1 range)

### 4. Clustering Implementation

- Generate pairwise similarity matrix for all companies in Excel
- Implement two clustering approaches:
  - **DBSCAN**: Density-based, auto-determines number of clusters
  - **Agglomerative Clustering**: Hierarchical with similarity threshold
- Create visualization of clusters
- Output cluster assignments with confidence scores

### 5. Evaluation & Validation

- Classification metrics: Precision, Recall, F1, ROC-AUC
- Clustering metrics: Silhouette score, cluster size distribution
- Generate sample outputs for manual review
- Create confusion analysis for hard negatives

### 6. Production-Ready Code Structure

```
project/
├── data/
│   ├── migration-analysis.xlsx
│   ├── training_simple.pkl
│   └── training_hard_negatives.pkl
├── models/
│   ├── lgbm_simple.pkl
│   └── lgbm_hard.pkl
├── src/
│   ├── features.py        # Feature engineering
│   ├── train.py           # Model training
│   ├── cluster.py         # Clustering logic
│   └── inference.py       # Production inference
├── config.py              # Configuration
├── main.py               # Orchestration script
└── requirements.txt
```

### 7. Deliverables

- Trained LGBM model(s) with evaluation metrics
- Clustered company list with cluster IDs and confidence
- Feature importance analysis for explainability
- Sample validation set for manual review
- Documentation of approach and results

## Key Files to Reference

- `lgbm.ipynb`: Feature engineering functions, LGBM training code
- `siamese_training.ipynb`: Alternative approach reference
- `.env`: Configuration parameters to reuse

## Next Steps After Prototype

- If successful, explore Neo4j integration for graph-based clustering
- Consider ensemble with Siamese model for difficult cases
- Implement active learning with user feedback on clusters

### To-dos

- [ ] Load and inspect migration-analysis.xlsx and understand data structure
- [ ] Create modular feature engineering module from lgbm.ipynb
- [ ] Build training pipeline for LGBM with simple and hard negatives datasets
- [ ] Train and evaluate LGBM models on both datasets
- [ ] Implement DBSCAN and Agglomerative clustering on similarity scores
- [ ] Apply clustering to migration-analysis.xlsx companies
- [ ] Generate evaluation metrics, visualizations, and review samples