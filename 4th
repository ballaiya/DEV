Exp = 4== "Implementation of feature based representations of time series "
required= city_temperature.zip

import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("city_temperature.csv", low_memory=False, na_values=-99)

df["AvgTempC"] = (df["AvgTemperature"] - 32) * 5 / 9
cities = ["New York", "London", "Delhi", "Sydney"]
df = df[df["City"].isin(cities) & df["Year"].between(2010, 2019)]
df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")

monthly = (
    df.groupby(["City", pd.Grouper(key="date", freq="M")])["AvgTempC"]
    .mean()
    .reset_index()
)

ts_matrix = (
    monthly.pivot(index="date", columns="City", values="AvgTempC")
    .dropna()
    .T
)

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
clusters = kmeans.fit_predict(ts_matrix)
ts_matrix["cluster"] = clusters
print(ts_matrix[["cluster"]])
