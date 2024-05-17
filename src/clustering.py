import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.utils import shuffle


def kmeans(image, k=4, max_iters=100):
    """ Perform K-means clustering on image """
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

    Parameters:
    - image: Input image as a 3D NumPy array (height, width, 3).
    - k: Number of clusters.
    - affinity: Affinity type to use ('nearest_neighbors' or other types).
    - n_neighbors: Number of neighbors to use when constructing the affinity graph (relevant if affinity='nearest_neighbors').

    Returns:
    - labels: Cluster labels for each pixel reshaped to the original image shape.
    - centroids: RGB values of the cluster centroids.
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

    # Reshape labels back to the original image shape (height, width)
    labels = labels.reshape(image.shape[0], image.shape[1])
    return labels, centroids
