Exp = 4
required= city_temperature.zip

import zipfile
import pandas as pd
from sklearn.cluster import KMeans

with zipfile.ZipFile("city_temperature.zip") as z:
    with z.open("city_temperature.csv") as f:
        df = pd.read_csv(f, low_memory=False, na_values="-99")

df["AvgTempC"] = df["AvgTemperature"]
cities = ["New York", "London", "Delhi", "Sydney"]
df = df[(df["City"].isin(cities)) & (df["Year"].between(2010, 2019))]
df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")

monthly = df.groupby(["City", pd.Grouper(key="date", freq="M")])["AvgTempC"].mean().reset_index()
ts_matrix = monthly.pivot(index="date", columns="City", values="AvgTempC").dropna().T

ts_matrix = (ts_matrix - ts_matrix.mean(axis=1).values.reshape(-1, 1)) / ts_matrix.std(axis=1).values.reshape(-1, 1)

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
clusters = kmeans.fit_predict(ts_matrix)
ts_matrix["cluster"] = clusters

print("===== City Cluster Assignments =====")
print(ts_matrix[["cluster"]])
