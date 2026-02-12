import matplotlib.pyplot as plt
import seaborn as sns

from data_load import read_data
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def randomforest():
    x_train, x_test, y_train, y_test = read_data()

    # Basmodell (utan att låsa hyperparametrar)
    rf = RandomForestClassifier(random_state=42)

    # Grid att testa (du kan minska/öka vid behov)
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="accuracy",   # byt till "f1_macro" om klasserna är obalanserade
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    # TUNING på träningsdata
    grid.fit(x_train, y_train)

    # Bästa modellen
    classifier = grid.best_estimator_
    print("Bästa parametrar:", grid.best_params_)
    print(f"Bästa CV-score: {grid.best_score_:.4f}")

    # Samma typ av output som tidigare (prediktion på test)
    y_pred = classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    """
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")

    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    feature_importances = classifier.feature_importances_
    header = x_test.columns
    plt.figure()
    plt.barh(header, feature_importances)
    plt.xlabel('Feature importance')
    plt.title('Feature importance in Random Forest Classifier')
    plt.show()
    """

    # Samma träd-plot som du gjorde innan
    tree_to_plot = classifier.estimators_[0]

    plt.figure(figsize=(20, 10))
    plot_tree(
        tree_to_plot,
        max_depth=4,
        feature_names=x_train.columns.tolist(),
        filled=True,
        rounded=True,
        fontsize=6
    )
    plt.title("Decision Tree from Tuned Random Forest")
    plt.show()


randomforest()
