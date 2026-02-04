import matplotlib.pyplot as plt
import numpy as np
import sklearn
import seaborn as sns
from data_load import read_data
import pandas as df
from sklearn.tree import plot_tree


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def randomforest():
    x_train, x_test, y_train, y_test = read_data()
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    accuracy= accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    """
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix)

    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    feature_importaces = classifier.feature_importances_
    header = x_test.columns
    plt.barh(header, feature_importaces)
    plt.xlabel('Feature importance')
    plt.title('Feature importance in Random Forest Classifier')
    plt.show()
    """

    tree_to_plot = classifier.estimators_[0]
    plt.figure(figsize=(20, 10))
    plot_tree(tree_to_plot, max_depth = 4, feature_names=x_train.columns.tolist(), filled=True, rounded=True, fontsize=6)
    #plot_tree(tree_to_plot, feature_names=df.columns.tolist(), filled=True, rounded=True, fontsize=10)
    #plot_tree(tree_to_plot, filled=True, rounded=True, fontsize=10)
    
    plt.title("Decision Tree from Random Forest")
    plt.show()

    plt.show()


randomforest()