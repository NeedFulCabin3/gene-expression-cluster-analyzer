# Gene Expression Cluster Analyzer

Unsupervised Genomic Dimensionality Reduction and Density Clustering

## Overview
High-dimensional gene expression tables present severe analytical friction because traditional distance metrics degrade rapidly across thousands of feature columns. This utility provides a streamlined command-line pipeline to ingest expression matrices, handle missing values, normalize features, compress dimensions, and cluster samples without pre-specifying cluster counts.

## How It Works
1. **Data Ingestion**: Loads user-provided CSV expression tables (samples as rows, genes as columns) or dynamically initializes synthetic gene expression data with artificial cluster signatures.
2. **Data Sanitization**: Scans for null values across the DataFrame and applies column-wise mean imputation to prevent matrix computation failures.
3. **Feature Standardization**: Applies scikit-learn's `StandardScaler` to center features around zero with unit variance, neutralizing scale disparities.
4. **Dimensionality Reduction**: Computes singular value decomposition via `PCA`, evaluates cumulative explained variance ratios to target an 80% variance threshold, and restricts components between 2 and 10 dimensions.
5. **Density Clustering**: Executes `DBSCAN` on the reduced component space using terminal-configured epsilon and minimum sample thresholds, isolating biological clusters and identifying noise points.
6. **Diagnostic Export**: Generates and saves a matplotlib 2D scatter plot visualizing principal component variance ratios and cluster assignments.

## Key Features
* Automatic synthetic gene expression dataset generation with built-in cluster patterns.
* CSV file ingestion with automatic dimension validation and missing value imputation.
* Dynamic principal component retention driven by cumulative variance thresholds (80%).
* Customizable DBSCAN hyperparameter tuning via terminal-driven prompts.
* Automated matplotlib visualization export mapping PC1 against PC2.

## Tech Stack & Core Dependencies Breakdown
* **Python 3.10+**: Core runtime environment utilizing modern type hints and robust exception handling.
* **NumPy (`numpy`)**: Array manipulation, vector math, and random state seeding.
* **Pandas (`pandas`)**: Tabular data structure management, DataFrame indexing, and missing value handling.
* **Scikit-Learn (`scikit-learn`)**: Machine learning pipeline components (`PCA`, `DBSCAN`, `StandardScaler`).
* **Matplotlib (`matplotlib`)**: 2D scatter plot generation and image export.

## Environment & Web-Based Quick Start
### GitHub Codespaces
Open this repository directly inside a browser-based GitHub Codespace. The environment provisions Python automatically.

### Virtual Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execution
```bash
python main.py
```

## Repository Structure

```bash
gene-expression-cluster-analyzer/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── README.md
├── requirements.txt
└── main.py
```

## Roadmap

Migrate heavy matrix ingestion and transformation steps to multi-processing pools or asynchronous task runners for multi-gigabyte genomic datasets.

Introduce strict type annotations and static analysis compliance using mypy.

Implement structured logging via structlog to replace standard print statements for production logging pipelines.