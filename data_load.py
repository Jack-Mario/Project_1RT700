from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler 

def read_data():
    #Importera data
    df = pd.read_csv("training_data_VT2026.csv")

    #Gör vår "klass" kolumn numrerisk
    df["increase_stock"] = df["increase_stock"].map({
    "low_bike_demand": 0,
    "high_bike_demand": 1,
    })

    #Skapa x och y kolumner
    y = df["increase_stock"]
    x = df.drop(columns=["increase_stock"])

    #Skala om x
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    #Skapa test och träningsdata
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    #print(x_train, y_train)
    return x_train, x_test, y_train, y_test