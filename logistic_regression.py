import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import seaborn as sns
from data_load import read_data

x_train, x_test, y_train, y_test = read_data()

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=10000, random_state=0)
)

clf.fit(x_train, y_train)

acc = accuracy_score(y_test, clf.predict(x_test)) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")