import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from data_load import read_data

def logistic_regression(x_train, x_test, y_train, y_test):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=10000, random_state=0)
    )
    param_grid = {
    "logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "logisticregression__penalty": ["l2"]
}

    GS_lr = GridSearchCV(clf, param_grid, scoring = "accuracy", cv=3, n_jobs = -1)
    GS_lr.fit(x_train, y_train)

    print("Best parameters:", GS_lr.best_params_)
    best_model = GS_lr.best_estimator_
    y_pred = best_model.predict(x_test)


    performance_metrics = [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)]
    return performance_metrics
