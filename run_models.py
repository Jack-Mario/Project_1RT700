from src.random_forest import randomforest
from src.knn_model import train_knn as knn_model
from src.read_train import read_data
from src.logistic_regression import logistic_regression
from src.naive_model import naive_model

from src.read_test import read_test
from src.test_model import test_knn

x_train, x_test, y_train, y_test = read_data()
test_data = read_test()

width = 100

functions = [knn_model]
#functions = [randomforest, knn_model, logistic_regression, naive_model]
choosen_function = 'knn_model'

for func in functions:
    print(f"{len(str(func)) * " "} accuracy_score | f1_score | precision_score | recall_score")
    print(f"{func} Data: {func(x_train, x_test, y_train, y_test)}\n")
    print(f"{'=' * width}\n")

test_knn(x_train, y_train, test_data)