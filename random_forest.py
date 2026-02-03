import matplotlib.pyplot as plt
import numpy as np
import sklearn
import seaborn as sns
from data_load import read_data


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

randomforest()