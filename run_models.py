from src.random_forest import randomforest
from src.knn_model import train_knn
from src.data_load import read_data
from src.logistic_regression import logistic_regression
from src.naive_model import naive_model

x_train, x_test, y_train, y_test = read_data()
width = 100

functions = [randomforest, train_knn, logistic_regression, naive_model] 

for func in functions:
    print(f"{len(str(func)) * " "} accuracy_score | f1_score | precision_score | recall_score")
    print(f"{func} Data: {func(x_train, x_test, y_train, y_test)}\n")
    print(f"{'=' * width}\n")