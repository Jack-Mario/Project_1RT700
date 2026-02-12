from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_knn(x_train, x_test, y_train, y_test):
    grid_params = {'n_neighbors' : [80, 90, 100],
                   'weights' : ['uniform', 'distance'],
                   'metric' : ['minkowski','euclidean','manhattan']}
    
    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1, scoring = 'accuracy')
    g_res = gs.fit(x_train, y_train)

    g_res.best_score_
    g_res.best_params_

    print(g_res.best_params_)
    knn = KNeighborsClassifier(**g_res.best_params_)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    performance_metrics = [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)]
    return performance_metrics
