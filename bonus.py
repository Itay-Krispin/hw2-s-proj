import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


def calculate_inertia(data, centroids, labels):
    inertia = 0
    for i in range(len(data)):
        centroid = centroids[labels[i]]
        inertia += np.sum((data[i] - centroid) ** 2)
    return inertia


iris = load_iris()
data = iris.data

k_values = range(1, 11)
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0)
    kmeans.fit(data)
    inertia = calculate_inertia(data, kmeans.cluster_centers_, kmeans.labels_)
    inertia_values.append(inertia)

print(inertia_values)

plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia_values, marker="o")
plt.title('Elbow Method for Selection of Optimal "K" Clusters')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.xticks(k_values)
plt.grid(True)

elbow_point = 2

plt.annotate(
    "Elbow Point",
    xy=(elbow_point, inertia_values[elbow_point] + 75),
    xytext=(elbow_point + 1, inertia_values[elbow_point] + 500),
    arrowprops=dict(facecolor="black", shrink=0.05),
)

# Save the plot
plt.savefig("elbow.png")
plt.show()
