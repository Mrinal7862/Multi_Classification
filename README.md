# Multi_Classification

# Logistic Regression Model Comparison

This notebook explores two approaches for Logistic Regression for multi-class classification:

1. **One-vs-Rest (OvR)**
2. **Multinomial**

## Data Preparation

- We use `make_classification` from `sklearn.datasets` to generate a synthetic dataset with 1000 samples, 10 features, and 2 classes.
- The dataset is split into training and testing sets using `train_test_split` with a test size of 30%.

## Model Training and Evaluation

### One-vs-Rest (OvR)
