import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("training_data_VT2026.csv")

data["increase_stock"] = data["increase_stock"].map({
    "low_bike_demand": 0,
    "high_bike_demand": 1,
})

print_values = ["hour_of_day ", "day_of_week", "month", "holiday", "weekday", "precip", "snow"]

for value in print_values:
    hourly_mean = data.groupby("precip")["increase_stock"].mean()
    hourly_mean.plot(kind='bar')                                 
    #data.plot(x='hour_of_day', y='increase_stock', kind='hist')
    plt.ylabel("Mean of increase stock")
    plt.title("Mean of increase stock depending on the precipitation")
    plt.show()

print(data["precip"])