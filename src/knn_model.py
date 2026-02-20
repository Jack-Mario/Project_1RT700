from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_knn(x_train, x_test, y_train, y_test):
    #De parametrat vi loopar igenom
    grid_params = {'n_neighbors' : [5, 10, 15, 20, 30, 40, 50 , 60, 70, 80],
                   'weights' : ['uniform', 'distance'],
                   'metric' : ['minkowski','euclidean','manhattan']}
    
    #Skapar och tränar KNN model baserat på grid_params, baserat på accuracy
    GS_knn = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1, scoring = 'accuracy')
    GS_knn.fit(x_train, y_train)
    
    #Får bästa parametrar, samt om man vill bästa score (MODELLNAMN.best_score_)
    #Printen gör att vi kan uppdatera GridSearch, när det blir många parametrar blir det väldigt data-tungt
    print(f"Best params: {GS_knn.best_params_}")

    #Skapar en ny model med de bästa parametrarna och kör testdata
    #Man måste köra "fit" då det är en ny modell som skapas
    knn_best_model = KNeighborsClassifier(**GS_knn.best_params_)
    knn_best_model.fit(x_train, y_train)
    y_pred = knn_best_model.predict(x_test)

    #classification report

    performance_metrics = [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)]
    return performance_metrics
