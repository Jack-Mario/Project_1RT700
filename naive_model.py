from data_load import read_data

def naive_model():
    x_train, x_test, y_train, y_test = read_data()
    majority_class = y_train.mode()[0]
    y_pred = [majority_class] * len(y_test)

    accuracy = sum(y_pred == y_test) / len(y_test)
    print(f'Accuracy: {accuracy * 100}%')

naive_model()

    