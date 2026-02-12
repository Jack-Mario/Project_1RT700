from random_forest import randomforest
from knn_model import train_knn
from data_load import read_data
from logistic_regression import logistic_regression
from naive_model import naive_model

x_train, x_test, y_train, y_test = read_data()

print(f'{'=' * 80}\n ')
print(f'Randomforest data: {randomforest(x_train, x_test, y_train, y_test)}\n {'=' * 80}')
print(f'KNN data: {train_knn(x_train, x_test, y_train, y_test)}\n {'=' * 80}')
print(f'naive model data: {naive_model(x_train, x_test, y_train, y_test)}\n {'=' * 80}')
print(f'Logistic regression data: {logistic_regression(x_train, x_test, y_train, y_test)}\n {'=' * 80}')


