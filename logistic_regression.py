from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


def logistic_regression(x_train, x_test, y_train, y_test):
    #De parametrat vi loopar igenom
    #Tog bort "logisticregression__penalty": ["l2"] ur param_grid pga varningar

    # TODO Ska vi ha random_state = 0?
    param_grid = {
    "C": [100],
    'max_iter' : [1000]
    }

    #Skapar och tränar KNN model baserat på grid_params, baserat på accuracy
    #n_jobs = -1 är för att vi ska köra så många träd som möjligt samtidigt
    GS_lr = GridSearchCV(LogisticRegression(), param_grid, scoring = "accuracy", cv=3, n_jobs = -1)
    GS_lr.fit(x_train, y_train)

    #Får bästa parametrar, samt om man vill bästa score (MODELLNAMN.best_score_)
    #Printen gör att vi kan uppdatera GridSearch, när det blir många parametrar blir det väldigt data-tungt
    print(f"Best params: {GS_lr.best_params_}")

    #Skapar en ny model med de bästa parametrarna och kör testdata
    #Man måste köra "fit" då det är en ny modell som skapas
    lr_best_model = LogisticRegression(**GS_lr.best_params_)
    lr_best_model.fit(x_train, y_train)
    y_pred = lr_best_model.predict(x_test)

    performance_metrics = [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)]
    return performance_metrics
