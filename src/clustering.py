import numpy as np
import cv2
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from skimage.transform import resize

from src.autoencoder import *


def kmeans(image, k=4, max_iters=100):
    """
    Perform K-means clustering on image

    :param image: Input image as a 3D NumPy array (height, width, 3).
    :param k: Number of clusters.
    :param max_iters: Maximum number of iterations to run K-means.
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


def spectral_clustering(image, graph_method, k=4, sigma=30.0, n_neighbors=10, max_iters=100, epsilon=1e-10, image_channel=3):
    """
    Perform Spectral Clustering on an image.

    :param image: Input image as a 3D NumPy array (height, width, 3).
    :param k: Number of clusters.
    :param sigma: Standard deviation for Gaussian kernel.
    :param n_neighbors: Number of neighbors for k-nearest neighbors affinity.
    :param max_iters: Maximum number of iterations to run K-means.
    :param epsilon: Small positive value to avoid division by zero.
    :param graph_method: Method to construct the graph ('knn' or 'fully_connected').
    :param image_channel: Number of channels in the image.
    :return: labels: Cluster labels for each pixel reshaped to the original image shape.
    :return: centroids: RGB values of the cluster centroids.
    """
    n_clusters = k
    pixels = image.reshape(-1, image_channel)

    if graph_method == 'knn':
        # Compute k-nearest neighbors graph
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(pixels)
        distances, indices = nbrs.kneighbors(pixels)

        # Compute the affinity matrix using the Gaussian (RBF) kernel
        similarity_matrix = np.zeros((pixels.shape[0], pixels.shape[0]))
        for i in range(pixels.shape[0]):
            for j in range(1, n_neighbors):
                similarity_matrix[i, indices[i, j]] = np.exp(-distances[i, j]**2 / (2.0 * sigma**2))
                similarity_matrix[indices[i, j], i] = similarity_matrix[i, indices[i, j]]

    elif graph_method == 'fully_connected':
        # Compute the similarity matrix using the Gaussian (RBF) kernel
        pairwise_dists = squareform(pdist(pixels, 'sqeuclidean'))
        similarity_matrix = np.exp(-pairwise_dists / (2.0 * sigma ** 2))

    # Compute the degree matrix
    degree_matrix = np.diag(similarity_matrix.sum(axis=1))

    # Ensure there are no zero entries in the degree matrix
    degree_matrix[degree_matrix == 0] = epsilon

    # Compute the normalized graph Laplacian
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree_matrix.diagonal()))
    normalized_laplacian = np.dot(d_inv_sqrt, np.dot(degree_matrix - similarity_matrix, d_inv_sqrt))

    # Compute the first k eigenvectors
    _, eigenvectors = eigh(normalized_laplacian, subset_by_index=[0, n_clusters - 1])

    # Normalize eigenvectors row-wise
    eigenvectors = normalize(eigenvectors, axis=1)

    # Use K-means++ to cluster the eigenvectors
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=max_iters, random_state=42)
    labels = kmeans.fit_predict(eigenvectors)

    # Convert centroids to original pixel space colors
    labels = labels.reshape(image.shape[:2])
    centroids = np.zeros((n_clusters, image_channel))
    for i in range(n_clusters):
        cluster_pixels = image[labels == i]
        if len(cluster_pixels) > 0:
            centroids[i] = np.mean(cluster_pixels, axis=0)

    return labels, centroids


def spectral_clustering_with_autoencoder(image, graph_method, k=4, sigma=30.0, n_neighbors=10, max_iters=100,
                                         epsilon=1e-10, hidden_dim=128, epochs=50, image_channel=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_clusters = k
    pixels = image.reshape(-1, image_channel) / 255.0  # Normalize pixel values to [0, 1]

    # Train autoencoder
    autoencoder = train_autoencoder(pixels, input_dim=image_channel, hidden_dim=hidden_dim, epochs=epochs)
    autoencoder.eval()
    with torch.no_grad():
        encoded_pixels, _ = autoencoder(torch.tensor(pixels, dtype=torch.float32).to(device))
    encoded_pixels = encoded_pixels.cpu().numpy()

    if graph_method == 'knn':
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(encoded_pixels)
        distances, indices = nbrs.kneighbors(encoded_pixels)
        similarity_matrix = np.zeros((encoded_pixels.shape[0], encoded_pixels.shape[0]))
        for i in range(encoded_pixels.shape[0]):
            for j in range(1, n_neighbors):
                similarity_matrix[i, indices[i, j]] = np.exp(-distances[i, j] ** 2 / (2.0 * sigma ** 2))
                similarity_matrix[indices[i, j], i] = similarity_matrix[i, indices[i, j]]

    elif graph_method == 'fully_connected':
        pairwise_dists = squareform(pdist(encoded_pixels, 'sqeuclidean'))
        similarity_matrix = np.exp(-pairwise_dists / (2.0 * sigma ** 2))

    degree_matrix = np.diag(similarity_matrix.sum(axis=1))
    degree_matrix[degree_matrix == 0] = epsilon
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree_matrix.diagonal()))
    normalized_laplacian = np.dot(d_inv_sqrt, np.dot(degree_matrix - similarity_matrix, d_inv_sqrt))
    _, eigenvectors = eigh(normalized_laplacian, subset_by_index=[0, n_clusters - 1])
    eigenvectors = normalize(eigenvectors, axis=1)

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=max_iters, random_state=42)
    labels = kmeans.fit_predict(eigenvectors)
    labels = labels.reshape(image.shape[:2])
    centroids = np.zeros((n_clusters, image_channel))
    for i in range(n_clusters):
        cluster_pixels = image[labels == i]
        if len(cluster_pixels) > 0:
            centroids[i] = np.mean(cluster_pixels, axis=0)

    return labels, centroids


def multi_scale_clustering(image, graph_method, k=4, sigma=30.0, n_neighbors=10, max_iters=100, epsilon=1e-10):
    labels_list = []
    scales = [0.5, 1, 2]
    for scale in scales:
        # Scale the image
        if scale < 1:
            # Downscaling
            scaled_image = resize(image, (int(32 * scale), int(32 * scale)), anti_aliasing=True)
        elif scale > 1:
            # Upscaling with smoothing
            smoothed_image = cv2.GaussianBlur(image, (5, 5), sigmaX=1)
            scaled_image = resize(smoothed_image, (int(32 * scale), int(32 * scale)), anti_aliasing=True)
        else:
            # Original scale
            scaled_image = image

        # Perform clustering
        labels, _ = spectral_clustering(scaled_image, graph_method, k, sigma, n_neighbors, max_iters, epsilon,
                                        image_channel=3)
        if scale != 1:
            labels = resize(labels.astype(float), (32, 32), order=0, anti_aliasing=False)

        labels_list.append(labels)

    # Majority voting across scales
    labels_stack = np.stack(labels_list, axis=-1)
    labels_final = np.apply_along_axis(lambda x: np.bincount(x.astype(int), minlength=k).argmax(), axis=2,
                                       arr=labels_stack)

    # Recalculate centroids for final labels
    centroids = np.zeros((k, 3))
    for i in range(k):
        centroids[i] = np.mean(image[labels_final == i], axis=0)

    return labels_final, centroids

