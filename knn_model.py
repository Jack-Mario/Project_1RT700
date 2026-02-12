import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_knn(x_train, x_test, y_train, y_test):
    K = []
    training = []
    test = []
    scores = {}
    grid_params = {'n_neighbors' : [5, 10, 50, 100, 200],
                   'weights' : ['uniform', 'distance'],
                   'metric' : ['minkowski','euclidean','manhattan']}
    
    
    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1)
    g_res = gs.fit(x_train, y_train)

    g_res.best_score_
    g_res.best_params_

    knn = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform',algorithm = 'brute',metric = 'minkowski')
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    """
    for k in range(1,200,5):
        clf = KNeighborsClassifier(n_neighbors = k)
        clf.fit(x_train, y_train)

        training_score = clf.score(x_train, y_train)
        test_score = clf.score(x_test, y_test)
        K.append(k)

        training.append(training_score)
        test.append(test_score)
        scores[k] = [training_score, test_score]
    """
        
    """
    for keys, values in scores.items():
        print(keys, ":", values)
    """
    #ax = sns.stripplot(x=K, y=training)
    #ax.set(xlabel='Values of k', ylabel='Training Score')
    #ax = sns.stripplot(x=K, y=test)
    #ax.set(xlabel='Values of k', ylabel='Test Score')

    plt.xlabel('Values of k')
    plt.ylabel('Score')
    plt.title('Test Score = green, Training Score = blue')
    plt.plot(K, training, color='b')
    plt.plot(K, test, color='g')
    #plt.show()

    performance_metrics = [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)]
    return performance_metrics
