import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("training_data_VT2026.csv")

data["increase_stock"] = data["increase_stock"].map({
    "low_bike_demand": 0,
    "high_bike_demand": 1,
})

print_values = ["hour_of_day", "day_of_week", "month", "holiday", "weekday", "precip", "snowdepth"]

for value in print_values:
    series_to_plot = data.groupby(value)["increase_stock"].sum()
    ax = series_to_plot.plot(kind='bar')

    ax.set_ylabel("Number of 'increase stock' observations")
    ax.set_title(f"Sum of increase stock depending on {value}")
    if value == "precip":
        ax.set_xlim(left=0, right=data["precip"].max()/2)
    
    fig = ax.get_figure() 
    name = value + ".png"
    fig.savefig(name)

#print(data["precip"])