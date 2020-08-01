from __future__ import print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics

def create_dataset():
    # Generate sample points
    centers = [[3,5], [5,1], [8,2], [6,8], [9,7]]
    X, y = make_blobs(n_samples=1000,centers=centers,cluster_std=[0.5, 0.5, 1, 1, 1],random_state=3320)
    # =======================================
    # Complete the code here.
    # Plot the data points in a scatter plot.
    # Use color to represents the clusters.
    plt.scatter(X[:, 0], X[:, 1], c=y)
    # =======================================
    return [X, y]

def my_clustering(X, y, n_clusters):
    # =======================================
    # Complete the code here.
    # you need to
    #   1. Implement the k-means by yourself
    #   and cluster samples into n_clusters clusters using your own k-means
    #
    #   2. Print out all cluster centers.
    #
    #   3. Plot all clusters formed,
    #   and use different colors to represent clusters defined by k-means.
    #   Draw a marker (e.g., a circle or the cluster id) at each cluster center.
    #
    #   4. Return scores like this: return [score, score, score, score]
    kmeans = KMeans(n_clusters).fit(X)
    centers = kmeans.cluster_centers_
    print("All cluster centers: \n", centers)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    for i in range(n_clusters):
        plt.annotate(i, kmeans.cluster_centers_[i])
        plt.scatter(centers[i][0], centers[i][1], c="red")
    plt.show()
    y_pre = kmeans.labels_
    ari_score = metrics.cluster.adjusted_rand_score(y, y_pre)
    mri_score = metrics.adjusted_mutual_info_score(y, y_pre)
    v_measure_score = metrics.v_measure_score(y, y_pre)
    silhouette_avg = metrics.silhouette_score(X, y_pre, metric='euclidean')
    # =======================================
    return [ari_score,mri_score,v_measure_score,silhouette_avg]  # You won't need this line when you are done

def main():
    X, y = create_dataset()
    range_n_clusters = [2, 3, 4, 5, 6]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)

    for n_clusters in range_n_clusters:
        i = n_clusters - range_n_clusters[0]
        print("Number of clusters is: ", n_clusters)
        # Implement the k-means by yourself in the function my_clustering
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering(X, y, n_clusters)
        print('The ARI score is: ', ari_score[i])
        print('The MRI score is: ', mri_score[i])
        print('The v-measure score is: ', v_measure_score[i])
        print('The average silhouette score is: ', silhouette_avg[i])

    # =======================================
    # Complete the code here.
    # Plot scores of all four evaluation metrics as functions of n_clusters in a single figure.
    plt.plot(range_n_clusters, ari_score, label='ARI')
    plt.plot(range_n_clusters, mri_score, label='MRI')
    plt.plot(range_n_clusters, v_measure_score, label='V-measure')
    plt.plot(range_n_clusters, silhouette_avg, label='Silhouette Coefficient')
    plt.legend(loc='upper right')
    plt.show()
    # =======================================

if __name__ == '__main__':
    main()

