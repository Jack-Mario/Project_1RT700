# Project in Statistical Machine Learning 1RT700 VT2026
Authors: Jack Bossel, David Olson, and Tim Slettenmark Gille

## Overview
This repository contains a machine learning workflow for classifying bike demand into two classes: low or high. The project uses multiple classification models with hyperparameter tuning via GridSearchCV to find optimal parameters. It includes both model evaluation on validation data and prediction generation for unseen test data.

The main entry point is `run_models.py`, which orchestrates data loading, model training, evaluation, and prediction.

## Dataset
- **Training data**: `training_data_VT2026.csv`
- **Test data**: `test_data_VT2026.csv`
- **Target column**: `increase_stock`
- **Target mapping**: `low_bike_demand` → 0, `high_bike_demand` → 1
- **Categorical features**: hour_of_day, day_of_week, month, holiday, weekday, summertime

## Data Preprocessing

Both training and test data undergo the following preprocessing steps:

1. **One-Hot Encoding**: Categorical features are encoded using `OneHotEncoder`
2. **Feature Scaling**: All features are standardized using `StandardScaler`
3. **Train/Test Split**: Training data is split 80/20 with stratification to maintain class balance

## Models

All models use `GridSearchCV` with 3-fold cross-validation to optimize hyperparameters:

1. **K-Nearest Neighbors (KNN)**
   - Parameters: `n_neighbors` (7, 9, 10, 12), `weights` (uniform, distance)
   - Optimized for F1 score

2. **Random Forest**
   - Parameters: `n_estimators`, `max_depth`, `max_features`
   - Optimized for accuracy

3. **Logistic Regression**
   - Tuned regularization and solver parameters

4. **Naive Model**
   - Baseline for comparison

### Predictions
- Predictions for test data are saved to `predictions.csv`
- Contains one column: `prediction` (0 or 1)
- Generated using the best KNN model
