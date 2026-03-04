import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def read_test(): 
    #Importera data
    df_test = pd.read_csv("test_data_VT2026.csv")

    categorical_columns = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 'summertime']
    categorical_data_test = df_test[categorical_columns]

    encoder = OneHotEncoder(sparse_output=False)
    encoded_test = encoder.fit_transform(categorical_data_test)

    one_hot_df_test = pd.DataFrame(
        encoded_test,
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    
    df_test = pd.concat([df_test, one_hot_df_test], axis = 1)
    df_test = df_test.drop(columns=categorical_columns)
    
    scaler = StandardScaler()
    df_test = scaler.fit_transform(df_test)

    return df_test