%matplotlib qt

jebvliBEV
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("training_data_VT2026.csv")

data["increase_stock"] = data["increase_stock"].map({
    "low_bike_demand": 0,
    "high_bike_demand": 1
})

counts = data.groupby("hour_of_day")["increase_stock"].sum()

counts.plot(kind="bar")
plt.figure()

plt.xlabel("Hour of day")
plt.ylabel("High bike demand frequency")
plt.title("High bike demand per hour")
plt.show()
