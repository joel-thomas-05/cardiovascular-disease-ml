# Cardiovascular Disease Classification and Risk Modeling

This project compares several machine learning models for cardiovascular disease prediction and cardiovascular risk classification.

The notebook uses two datasets:

1. `heart.csv` — heart disease classification using clinical variables such as age, cholesterol, chest pain type, maximum heart rate, and related measurements.
2. `CVD Dataset.csv` — cardiovascular disease risk classification using demographic, clinical, biochemical, and lifestyle-related variables.


## Methods

The notebook compares the following models:

- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting

For Dataset 1, the target variable is `target`, where the model predicts whether heart disease is present.

For Dataset 2, the target variable is `CVD Risk Level`. Columns that directly reveal or duplicate the target, such as `CVD Risk Score`, are removed before model training to reduce target leakage.

## Main Analysis Steps

1. Load and inspect both datasets
2. Prepare features and target labels
3. Train multiple classification models
4. Compare accuracy, precision, recall, F1 score, and ROC-AUC
5. Generate confusion matrices for each model
6. Plot model accuracy with 95% confidence interval bands from repeated train/test splits
7. Plot ROC curves with bootstrap confidence bands
8. Calculate Random Forest feature importance
9. Calculate permutation importance for Dataset 1
10. Test PCA-based models for Dataset 1
11. Run optional SHAP analysis for feature interpretation
12. Save result tables as CSV files

## How to Run

Open the notebook in Google Colab or Jupyter Notebook.

If using Colab, upload the datasets to Google Drive and update these path variables in the notebook:

```python
HEART_PATH = '/content/drive/MyDrive/Colab Notebooks/ENBC 321/ML Project/heart.csv'
CVD_PATH = '/content/drive/MyDrive/Colab Notebooks/ENBC 321/ML Project/CVD Dataset.csv'
```

Then run the notebook from top to bottom.

## Outputs

The notebook saves result tables for model performance and feature importance. These outputs can be used for a report, presentation, or further analysis.

Key outputs include:

- Model comparison tables
- Confusion matrices
- ROC curves
- Accuracy comparison plots with confidence interval bands
- Feature importance plots
- SHAP summary plots

## Notes

The CVD risk dataset includes variables that can directly reveal the final risk label. These columns are removed before training to make the model evaluation more realistic.

The confidence interval plots are based on repeated train/test splits, so they give a better view of model stability than a single train/test split alone.

## Requirements

```text
numpy
pandas
matplotlib
scikit-learn
shap
```

SHAP is optional. The notebook installs it only if it is not already available.
