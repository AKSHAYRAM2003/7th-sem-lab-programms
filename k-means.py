import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
# Load the dataset
data = pd.read_csv("/content/iris1.csv")
X = data.iloc[:, :-1].values # Features
# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
# Apply EM clustering
em = GaussianMixture(n_components=3, random_state=42)
em_labels = em.fit_predict(X_scaled)
# Ground truth labels from the dataset
#true_labels = data["variety"].map({"setosa": 0, "versicolor": 1, "virginica": 2})
true_labels = data["variety"]
# Evaluate clustering results using Adjusted Rand Index (ARI)
ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
ari_em = adjusted_rand_score(true_labels, em_labels)
print("K-means ARI:", ari_kmeans)
print("EM ARI:", ari_em)