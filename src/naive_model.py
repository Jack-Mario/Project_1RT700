from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def naive_model(x_train, x_test, y_train, y_test):
    majority_class = y_train.mode()
    y_pred = [majority_class] * len(y_test)

    performance_metrics = [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)]
    return performance_metrics