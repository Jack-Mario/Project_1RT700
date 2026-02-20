import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def randomforest(x_train, x_test, y_train, y_test):
    #De parametrat vi loopar igenom
    param_grid = { 'n_estimators' : [50, 75, 100, 125, 150],
                  'max_features' : ['sqrt', 'log2'],
                  'max_depth' : [5, 7, 9, 11, 15],
    }
    #Skapar och tränar RF model baserat på grid_params, baserat på accuracy
    #n_jobs = -1 är för att vi ska köra så många träd som möjligt samtidigt
    GS_rf = GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid, cv = 3, scoring = 'accuracy', n_jobs = -1)
    GS_rf.fit(x_train, y_train)

    #Får bästa parametrar, samt om man vill bästa score (MODELLNAMN.best_score_)
    #Printen gör att vi kan uppdatera GridSearch, när det blir många parametrar blir det väldigt data-tungt
    print(f"Best params: {GS_rf.best_params_}")

    #Skapar en ny model med de bästa parametrarna och kör testdata
    #Man måste köra "fit" då det är en ny modell som skapas
    rf_best_model = RandomForestClassifier(**GS_rf.best_params_)
    rf_best_model.fit(x_train, y_train)
    y_pred = rf_best_model.predict(x_test)

    #Visualisering av ett träd
    
    #tree_to_plot = rf_best_model.estimators_[0]
    #plt.figure(figsize=(20, 10))
    #plot_tree(tree_to_plot, max_depth = 4, feature_names=x_train.columns.tolist(), filled=True, rounded=True, fontsize=6)
    #plt.title("Decision Tree from Random Forest")
    #plt.show()

    performance_metrics = [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)]
    return performance_metrics


