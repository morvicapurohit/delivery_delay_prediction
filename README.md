# Predicting Delivery Delays in Logistics Supply Chains

**MSIN0097 Predictive Analytics — Individual Coursework 2025-26**

## Project Overview

End-to-end multi-class classification pipeline to predict delivery delay status for a global logistics supply chain. The dataset contains 15,549 real-world order records with 41 variables.

| Label | Meaning | Approximate Proportion |
|-------|---------|------------------------|
| 1 | On Time / Early delivery | ~58% |
| 0 | Moderate delay | ~23% |
| -1 | Significant delay | ~19% |

## Repository Structure

```
predictive-dataset/
├── delivery_delay_prediction.ipynb     # Main analysis notebook (all code and results)
├── incom2024_delay_example_dataset.csv # Dataset (15,549 records, 41 features)
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Setup

### Prerequisites

- Python 3.9 or higher
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch and run the notebook

```bash
jupyter notebook
```

Open `delivery_delay_prediction.ipynb`, then select **Kernel > Restart & Run All**.

> **Note:** XGBoost and SHAP install automatically if not present. TensorFlow is required for the neural network section (Section 8) and will also self-install if missing. GridSearchCV runs 27 combinations × 3 folds for both XGBoost and Random Forest — allow 5–15 minutes depending on hardware. All random states are fixed (`random_state=42`) for full reproducibility.

## Notebook Structure

| Section | Content |
|---------|---------|
| 1. Introduction and Problem Framing | Target definition, success metric (Macro F1), assumptions |
| 2. Setup and Imports | All library imports |
| 3. Data Loading and Initial Inspection | Shape, dtypes, missing values |
| 4. Exploratory Data Analysis | Target distribution, feature engineering (`days_to_ship`), numeric distributions, categorical analysis, correlation heatmap |
| 5. Data Preparation | Feature selection rationale, preprocessing pipeline (median imputation + StandardScaler / mode imputation + OHE), stratified 80/20 split, class weighting |
| 6. Model Training | Logistic Regression, Random Forest, XGBoost (with LabelEncoder), 5-fold cross-validation, GridSearchCV tuning |
| 7. Learning Curves | Bias-variance diagnosis for Logistic Regression vs Random Forest |
| 8. Neural Network | Feedforward network (128→64→32), BatchNorm, Dropout, EarlyStopping on val_loss |
| 9. Model Comparison | Ranked bar chart and table of all model variants by test Macro F1 |
| 10. Error Analysis | Confusion matrices (Tuned RF, Tuned XGBoost, NN), Precision-Recall curves |
| 11. Feature Importance and Interpretability | Gini-based importance, SHAP beeswarm (Significant Delay class), global mean SHAP bar chart, waterfall plot for individual prediction |
| 12. Ablation Study | Full vs logistics-only vs financial-only feature subsets using tuned Random Forest |
| 13. Final Solution | Model selection rationale, model card, limitations and next steps |

## Key Results

- **Primary metric:** Macro F1-score (chosen due to class imbalance)
- **Best model:** Tuned XGBoost with balanced class weighting
- **Most important features:** `days_to_ship`, `order_status`, `shipping_mode` (confirmed by both Gini importance and SHAP)
- **Ablation finding:** Logistics features carry more predictive signal than financial features; combining both yields the best performance

## Data

The dataset `incom2024_delay_example_dataset.csv` is included in this repository and must be in the **same directory** as the notebook. No external downloads or API keys are required.

## Environment

Tested on Python 3.10+, Windows 11, Jupyter Notebook 7.x.
