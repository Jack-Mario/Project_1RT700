from pathlib import Path

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def read_data(): 
    #Importera data
    data_path = Path(__file__).with_name("training_data_VT2026.csv")
    df = pd.read_csv(data_path)

    #Gör vår "klass" kolumn numrerisk
    df["increase_stock"] = df["increase_stock"].map({
    "low_bike_demand": 0,
    "high_bike_demand": 1,
    })

    #Skapa x och y kolumner
    y = df["increase_stock"]
    x = df.drop(columns=["increase_stock"])

    categorical_columns = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 'summertime']
    categorical_data = x[categorical_columns]
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(categorical_data)
    one_hot_df = pd.DataFrame(
        one_hot_encoded,
        columns=encoder.get_feature_names_out(categorical_columns)
    )

    x = pd.concat([x, one_hot_df], axis = 1)
    x = x.drop(columns=categorical_columns)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify = y)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    #Skapa test och träningsdata, test_size bestämmer storlek på test data (%)
    #Stratify ser till att proportionen test/träning är samma i våra folds
    
    #print(x_train, y_train)
    return x_train, x_test, y_train, y_test
