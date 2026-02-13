# Project in Statistical Machine Learning 1RT700 VT2026
Authors: Jack Bossel, David Olson, and Tim Slettenmark Gille

## Overview
This repository contains a small machine learning workflow for classifying bike demand into two classes: low or high. The project reads a CSV dataset, splits it into train and test sets, trains multiple models, and reports common classification metrics.

The main entry point is `run_models.py`, which loads the data and runs the selected models.

## Dataset
- File: `training_data_VT2026.csv`
- Target column: `increase_stock`
- Target mapping: `low_bike_demand` -> 0, `high_bike_demand` -> 1

## Project structure
- `data_load.py`: loads the dataset and performs the train/test split
- `run_models.py`: entry point for running models
- `random_forest.py`: random forest with grid search
- `knn_model.py`: k-nearest neighbors with grid search
- `logistic_regression.py`: logistic regression with grid search
- `naive_model.py`: baseline model
- `ploting_data.py`: plotting utilities (if used)
- `Graphs/`: output plots (if generated)

## Setup
1. Install dependencies:
	- `pandas`
    - `numpy`
    - `scikit-learn`
    - `matplotlib`
    - `seaborn` if confusionmatrix is used
2. Make sure `training_data_VT2026.csv` is in the project root.

## Run
From the project root:
- `python run_models.py`

You can toggle which models run by commenting lines in `run_models.py`.

## Output
Each model returns or prints standard classification metrics, such as:
- Accuracy
- F1 score
- Precision
- Recall

The random forest script can also generate a decision tree plot.

## Notes
- Grid searches can be slow; reduce the parameter grid or use fewer CV folds if needed.
- If you see errors related to `max_features='auto'`, switch to `sqrt` or `log2` in newer scikit-learn versions.