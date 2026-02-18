from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def naive_model(x_train, x_test, y_train, y_test, seed=0):

    rng = np.random.default_rng(seed)

    # Räkna klassfördelning från träningsdatan
    classes, counts = np.unique(y_train, return_counts=True)
    probs = counts / counts.sum()

    # Slumpa prediction enligt samma fördelning
    y_pred = rng.choice(classes, size=len(y_test), p=probs)

    performance_metrics = [
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, zero_division=0),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0)
    ]

    return performance_metrics 