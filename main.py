
# ## Clustering Project

# ### Define Dataset

import pandas as pd

df = pd.read_csv("info_dataset.csv")

print(df.shape)
df.tail(3)


df.info()


# ## Preprocessing


df["Gender"].value_counts()


# map gender categories to numeric values (Male=0, Female=1).
df["new_Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
df.head()


# ### KMeans Clusterer


# We do not use the Gender feature in the K-Means algorithm because it is a categorical feature, even if it is encoded with numeric values.
x_km = df.values[:, 2:5]
x_km[:5]


from sklearn.preprocessing import StandardScaler

x_scaled_km = StandardScaler().fit_transform(x_km) # Since the K-Means algorithm is distance-based, the data must be normalized.

x_scaled_km[:5]


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np


interies = [] # We use the inertia parameter to plot the elbow method chart.
silhouette = [] 
mapping = {} # This dictionary stores the inertia and silhouette score for each cluster (k).
K = range(2, 11) # We consider cluster numbers from 2 to 11 and evaluate the inertia and silhouette score for each one.

for k in K:
    km_model = KMeans(n_clusters=k, init="k-means++", n_init=10).fit(x_scaled_km)
    km_labels = km_model.labels_

    interies.append(km_model.inertia_)
    silhouette.append(silhouette_score(x_scaled_km, km_labels)) # The silhouette score evaluates the compactness within clusters and the separation between clusters.
    mapping[k] = (interies[-1], silhouette[-1])


# We use the silhouette score and the elbow method plots to determine the optimal number of clusters (best k).

for key, val in mapping.items():
    print(f"for n_cluster = {key} : inertia == {val[0]} | average slihouette score == {val[1]}")

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)

ax1.plot(K, interies, color="red")
ax2.plot(K, silhouette, color="green")

ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method")

ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("score")
ax2.set_title("The silhouette_score")
plt.show()


# The value of k that has the highest silhouette score and corresponds to the elbow point in the Elbow Method chart is the best choice for the number of clusters.

km_model = KMeans(n_clusters=6, init="k-means++", n_init=10).fit(x_scaled_km) # According to the two plots above, the optimal number of clusters is 6.


km_labels = km_model.labels_
km_labels


km_clusters = km_model.cluster_centers_
km_clusters


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d') # Add a 3D subplot to the figure

colors = plt.cm.Spectral(np.linspace(0, 1, len(set(km_labels))))

for k, color in zip(range(len(km_clusters)), colors):

    this_members = (km_labels == k) # Select datapoints belonging to cluster k
    cluster_center = km_clusters[k] # Select the centroid of this cluster.


    ax.scatter(
        x_scaled_km[this_members, 0], # Age feature
        x_scaled_km[this_members, 1], # Annual Income feature
        x_scaled_km[this_members, 2], # Spending Score feature
        color=color,
        marker="o",
        alpha=0.7
        )

    ax.scatter(
        cluster_center[0],
        cluster_center[1],
        cluster_center[2],
        marker="o",
        color=color,
        edgecolor="black", # Set the marker edge (outline) color to black
        s=50,
        alpha=0.7
        )

ax.set_xlabel("Age", labelpad=-10) # Labelpad adjust the distance between the label and the axis (فاصله بین برچسب محور و محور را تنظیم می‌کند)
ax.set_ylabel("Annual Income", labelpad=-10)
ax.set_zlabel("Spending Score", labelpad=-10)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.show()



silhouette_score(x_scaled_km, km_labels)


# ### Hierarchical Clusterer


# For hierarchical clustering, we can use categorical features if they have been converted into numerical values.

x_hier = df.values[:, 2:] # We use "new_Gender" feature which is categorical and numerical.
x_hier[:5]


from sklearn.metrics.pairwise import euclidean_distances

dist_matrix = euclidean_distances(x_hier, x_hier) # Use the euclidean_distances function to obtain the proximity matrix.
dist_matrix


from scipy.cluster import hierarchy

z = hierarchy.linkage(dist_matrix, "complete") # Compute the linkage (hierarchical relationships) between the data points.


# Plot the hierarchy (dendrogram).
import pylab

fig = pylab.figure(figsize=(18, 50))

def llf(id):
    return "%s - %s - %s" % (df["Gender"][id], df["Annual Income (k$)"][id], df["Spending Score (1-100)"][id])

dendro = hierarchy.dendrogram(Z=z, leaf_label_func=llf, leaf_font_size=12, leaf_rotation=0, orientation="right")


from sklearn.cluster import AgglomerativeClustering

agglo_model = AgglomerativeClustering(n_clusters=6, linkage="complete") # Using the AgglomerativeClustering function to obtain the flat clusters.
agglo_model.fit(dist_matrix)

agglo_model.labels_



df["cluster"] = agglo_model.labels_ # Add the cluster column to the dataframe.
df.head()


# ##### The silhouette score of this hierarchical clustering model is 0.41, indicating a moderate clustering quality with reasonably separated clusters


silhouette_score(x_hier, agglo_model.labels_)


import matplotlib.cm as cm

n_cluster = max(agglo_model.labels_) + 1 # Because the label numbering starts from 0.
colors = cm.rainbow(np.linspace(0, 1, n_cluster)) # Generate distinct colors for each cluster using the rainbow colormap
cluster_labels = list(range(0, n_cluster)) # -> [0, 1, 2, 3, 4, 5]

plt.figure(figsize=(16, 14))

for color, label in zip(colors, cluster_labels):
    subset = df[df.cluster == label] # Filter rows where the cluster value equals the current label(subset -> dataframe).

    for i in subset.index: # We iterate over the indices of the subset table; note that these indices are the same as the indices of the original table.
        # Write the "Customer ID" value at the coordinates defined by "Age" (x-axis) and "Annual Income" (y-axis).
        plt.text(subset["Age"][i], subset["Annual Income (k$)"][i], "id=" + str(subset["CustomerID"][i]), rotation=25)
    ''' Plot all data points with "Age" (x-axis) and "Annual Income" (y-axis), while the bubble size of each sample is determined by its "Spending Score".
    Since the loop iterates over one cluster at a time, all samples belonging to the same cluster are displayed with the same color. '''
    plt.scatter(subset["Age"], subset["Annual Income (k$)"], s=subset["Spending Score (1-100)"]*10, c=color, label="cluster"+str(label), alpha=0.5)


plt.legend()
plt.title('Clusters')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.show()



# Stores the average Age, Annual Income, and Spending Score for each (cluster, Gender) group in mean_df
mean_df = df.groupby(["cluster", "Gender"])[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean() # The output is a DataFrame, not a Series.
mean_df


plt.figure(figsize=(16, 10))

for color, label in zip(colors, cluster_labels):
    subset = mean_df.loc[(label, ), ] # Select rows from mean_df where the cluster level of the index equals the given label

    for i in subset.index: # The value of i is either "Male" or "Female".
        ''' At the coordinates of the mean Age (x-axis) and mean Annual Income (y-axis),
        write the value of i (the gender) and the mean Spending Score of this group from the original dataframe. '''
        plt.text(subset.loc[i][0], subset.loc[i][1], "Gender=" + str(i) + ", score=" + str(int(subset.loc[i][2])))
    ''' Now we plot the mean Age and mean Annual Income of all groups on the chart,
    and the bubble size for each group represents that group's mean Spending Score. '''
    plt.scatter(subset.Age, subset["Annual Income (k$)"], s=subset["Spending Score (1-100)"] * 10, label="cluster" + str(label))

plt.legend()
plt.title('Clusters')
plt.xlabel('Age')
plt.ylabel('Annual Income')


# ### DBSCAN


# For DBSCAN, we also use only features that are not categorical, even if they have been converted to numeric values (Age column).
x_db = df.values[:, 2:5]
x_db[:5]


x_scaled_db = StandardScaler().fit_transform(x_db) # We must normalize the data because this algorithm is distance-based.

x_scaled_db[:5]


# This part of the code is used to find the best value for epsilon.
from sklearn.neighbors import NearestNeighbors

m_sample, k = 3 * 2, 5

neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(x_scaled_db) 

distances, indices = neigh.kneighbors(x_scaled_db) # We calculate the distance of each sample from all of its k neighbors. 
k_distances = distances[:, k-1] # We only need the distance from the last neighbor to determine the epsilon.
k_distances = np.sort(k_distances) # We sort the distances in ascending order.

plt.figure(figsize=(10, 6))
plt.plot(k_distances, color="green", linewidth=2)
plt.title(f"k-Distance Graph (for k={k})", fontsize=14)
plt.xlabel("Data Points sorted by distance", fontsize=12)
plt.ylabel(f"Distance to k-th neighbor (Epsilon)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()



from sklearn.cluster import DBSCAN

epsilon = 0.57
db_model = DBSCAN(eps=epsilon, min_samples=m_sample).fit(x_scaled_db)
db_labels = db_model.labels_

db_labels


core_sample_msk =  np.zeros_like(db_labels, dtype=bool) # Create a boolean array of False with the same shape as db_labels
core_sample_msk[db_model.core_sample_indices_] = True # Set True for positions of core samples in DBSCAN using the indices stored in core_sample_indices_

core_sample_msk


n_cluster_db = len(set(db_labels)) - (1 if -1 in db_labels else 0) # Count unique clusters in db_labels and exclude the noise label (-1) if it exists
n_cluster_db


unique_db_labels = set(db_labels)
unique_db_labels


# We plot a 3D scatter plot of the data to better examine the clusters.

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection="3d") # Create a 3D subplot (1x1 grid, first position) for plotting

colors = cm.rainbow(np.linspace(0, 1, len(unique_db_labels)))

for label, color in zip(unique_db_labels, colors):
    if label == -1: # Check if the current label represents noise in DBSCAN
        color="k" # Set the color to black for noise points

    cluster_member_msk = (db_labels==label) # Create a boolean mask selecting points whose DBSCAN label equals the current cluster label

    core_datapoints = x_scaled_db[cluster_member_msk & core_sample_msk] # Select core points of the current cluster
    ax.scatter(core_datapoints[:, 0], core_datapoints[:, 1], core_datapoints[:, 2], s=100, c=[color], marker="o", alpha=0.5) 
    border = x_scaled_db[cluster_member_msk & ~core_sample_msk] # Select border points of the current cluster
    ax.scatter(border[:, 0], border[:, 1], border[:, 2], s=30, c=[color], marker="o", alpha=0.5)

plt.title(f"3D scatter plot (for k={k} and eps={epsilon})", fontsize=14)



# ##### The Silhouette Score obtained for this clustering is 0.24, which indicates a relatively weak but still meaningful cluster structure


silhouette_score(x_scaled_db, db_labels)

