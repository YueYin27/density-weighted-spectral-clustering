import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.utils import shuffle


def kmeans(image, k=4, max_iters=100):
    """
    Perform K-means clustering on image

    :parameter image: Input image as a 3D NumPy array (height, width, 3).
    :parameter k: Number of clusters.
    :parameter max_iters: Maximum number of iterations to run K-means.
    :return: labels: Cluster labels for each pixel reshaped to the original image shape.
    :return: centroids: RGB values of the cluster centroids.
    """
    pixels = image.reshape(-1, 3)  # Reshape image to a 2D array of pixels, where each row is a pixel in RGB space
    centroids = shuffle(pixels)[:k]  # Initialize centroids by selecting k random points

    for _ in range(max_iters):
        # Assign pixels to the nearest centroid
        distances = np.sqrt(((pixels - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        closest_centroids = np.argmin(distances, axis=0)

        # Update centroids to be the mean of assigned pixels
        new_centroids = np.array([pixels[closest_centroids == i].mean(axis=0) for i in range(k)])

        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Reshape labels back to the original image shape (excluding color dimension)
    labels = closest_centroids.reshape(image.shape[0], image.shape[1])
    return labels, centroids


def spectral_clustering(image, k=4, affinity='nearest_neighbors', n_neighbors=10):
    """
    Perform Spectral Clustering on an image.

    :parameter image: Input image as a 3D NumPy array (height, width, 3).
    :parameter k: Number of clusters.
    :parameter affinity: Type of affinity matrix to use ('nearest_neighbors' or 'rbf').
    :parameter n_neighbors: Number of nearest neighbors to consider when using 'nearest_neighbors' affinity.
    :return: labels: Cluster labels for each pixel reshaped to the original image shape.
    :return: centroids: RGB values of the cluster centroids.
    """
    pixels = image.reshape(-1, 3)  # Reshape image to a 2D array of pixels (num_pixels, 3)
    # pixels = shuffle(pixels)  # Shuffle pixels to ensure random sampling

    # Apply spectral clustering
    if affinity == 'nearest_neighbors':
        spectral = SpectralClustering(n_clusters=k, affinity=affinity, n_neighbors=n_neighbors, assign_labels='kmeans')
    else:
        spectral = SpectralClustering(n_clusters=k, affinity=affinity, assign_labels='kmeans')

    labels = spectral.fit_predict(pixels)

    # Get centroids by averaging the pixels assigned to each cluster
    centroids = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])

def spectral_clustering_self(image, k=4):
    n_clusters = k
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Compute the affinity matrix
    sigma = 30.0
    pairwise_dists = squareform(pdist(pixels, 'sqeuclidean'))
    affinity_matrix = np.exp(-pairwise_dists / (2.0 * sigma ** 2))

    # Compute the degree matrix
    degree_matrix = np.diag(affinity_matrix.sum(axis=1))

    # Compute the unnormalized graph Laplacian
    laplacian_matrix = degree_matrix - affinity_matrix

    # Compute the normalized graph Laplacian
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree_matrix.diagonal()))
    normalized_laplacian = np.dot(d_inv_sqrt, np.dot(laplacian_matrix, d_inv_sqrt))

    # Compute the first k eigenvectors
    _, eigenvectors = eigh(normalized_laplacian, subset_by_index=[0, n_clusters-1])

    # Use k-means to cluster the eigenvectors
    kmeans = KMeans(n_clusters=n_clusters).fit(eigenvectors)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # Convert centroids to original pixel space colors
    pixel_centroids = np.zeros((n_clusters, 3))
    for i in range(n_clusters):
        cluster_pixels = image[segmented_image == i]
        if len(cluster_pixels) > 0:
            pixel_centroids[i] = np.mean(cluster_pixels, axis=0)

    return labels, pixel_centroids

    # Reshape labels back to the original image shape (height, width)
    labels = labels.reshape(image.shape[0], image.shape[1])
    return labels, centroids
