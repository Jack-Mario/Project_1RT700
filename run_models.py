from random_forest import randomforest
#from knn_model import train_knn
from data_load import read_data

x_train, x_test, y_train, y_test = read_data()

print(randomforest(x_train, x_test, y_train, y_test))


