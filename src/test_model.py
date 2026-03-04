from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def test_knn(x_train, y_train, test_data):
    n_neighbors = 10
    weights = 'distance'

    knn_best_model = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights)
    knn_best_model.fit(x_train, y_train)
    y_pred = knn_best_model.predict(test_data)

    df_pred = pd.DataFrame(y_pred, columns=['prediction'])

    return df_pred.to_csv('predictions.csv', columns=['prediction'], index = False)
    
