# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1.Start the program.

2.Import required libraries like pandas, matplotlib, and sklearn.

3.Load the Mall_Customers.csv dataset.

4.Select Annual Income and Spending Score as features.

5.Initialize an empty list to store WCSS values.

6.Apply K-Means for cluster values from 1 to 10.

7.Store inertia (WCSS) for each cluster value.

8.Plot the Elbow graph to find the optimal number of clusters.

9.Train K-Means again using the chosen number of clusters (e.g., 5).

10.Predict clusters, assign them to customers, and visualize the clusters using scatter plot.


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: HALIMA HARISHA A
RegisterNumber:  212224040094
*/


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=0,
        n_init=10
    )

    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')

plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.show()

km = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=0,
    n_init=10
)

km.fit(data.iloc[:, 3:])

y_pred = km.predict(data.iloc[:, 3:])

print(y_pred)

data["cluster"] = y_pred

df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

plt.figure(figsize=(10,6))

plt.scatter(
    df0["Annual Income (k$)"],
    df0["Spending Score (1-100)"],
    c="red",
    label="cluster0"
)

plt.scatter(
    df1["Annual Income (k$)"],
    df1["Spending Score (1-100)"],
    c="black",
    label="cluster1"
)

plt.scatter(
    df2["Annual Income (k$)"],
    df2["Spending Score (1-100)"],
    c="blue",
    label="cluster2"
)

plt.scatter(
    df3["Annual Income (k$)"],
    df3["Spending Score (1-100)"],
    c="green",
    label="cluster3"
)

plt.scatter(
    df4["Annual Income (k$)"],
    df4["Spending Score (1-100)"],
    c="magenta",
    label="cluster4"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")

plt.legend()
plt.show()

```

## Output:

<img width="800" height="553" alt="image" src="https://github.com/user-attachments/assets/627e9b10-97c8-4487-930c-53ecd02f8a10" />

<img width="788" height="532" alt="image" src="https://github.com/user-attachments/assets/8d512c4b-8964-44d5-8cad-e8f1e5474f3d" />


<img width="841" height="265" alt="image" src="https://github.com/user-attachments/assets/94f011b2-7acc-4470-9872-1bfeba0fe897" />

<img width="862" height="609" alt="image" src="https://github.com/user-attachments/assets/0cbda04a-7f0c-48b0-8431-942c06d6a89f" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
