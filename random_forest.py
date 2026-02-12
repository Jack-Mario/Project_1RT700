import matplotlib.pyplot as plt
import numpy as np
from data_load import read_data
import pandas as df
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def randomforest(x_train, x_test, y_train, y_test):
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    accuracy= accuracy_score(y_test, y_pred)
    #print(f'Accuracy: {accuracy * 100:.2f}%')

    tree_to_plot = classifier.estimators_[0]
    plt.figure(figsize=(20, 10))
    plot_tree(tree_to_plot, max_depth = 4, feature_names=x_train.columns.tolist(), filled=True, rounded=True, fontsize=6)
    #plot_tree(tree_to_plot, feature_names=df.columns.tolist(), filled=True, rounded=True, fontsize=10)
    #plot_tree(tree_to_plot, filled=True, rounded=True, fontsize=10)
    
    plt.title("Decision Tree from Random Forest")
    plt.show()
    performance_metrics = [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)]
    return performance_metrics


