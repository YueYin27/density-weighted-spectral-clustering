import os
import numpy as np
from src.clustering import *
from src.dataloader import *
from src.segmentation import *
import warnings
import random

# Set the random seed for reproducibility
random_seed = 7535192
random.seed(random_seed)
np.random.seed(random_seed)

# Ignore all warnings
warnings.filterwarnings('ignore')


def main():
    data_path = './data'
    output_path = './testresults'
    class_ids = [7, 4, 0]  # CIFAR-10 IDs for horse, deer, and airplane

    # Load images from specified classes
    extract_cifar10_dataset(data_path)
    train_images, train_labels, test_images, test_labels = load_cifar_10(data_path, class_ids)  # images (18000, 32, 32, 3), labels (18000,)

    # only use images with index: 26, 59, 129, 250, 335, 352
    # test_images = test_images[[85, 18, 129, 245, 335, 352]]
    # test_labels = test_labels[[85, 18, 129, 245, 335, 352]]

    # Check if images were loaded
    if not test_images.size:
        print("No images loaded. Check the dataset and class IDs.")
        return

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # # Process and segment images using K-means
    # segment_images("kmeans", test_images, test_labels, os.path.join(output_path, "kmeans"), k=4, max_iters=100)

    # Process and segment images using Spectral clustering
    # segment_images("spectral", test_images, test_labels, os.path.join(output_path, "spectral_knn_5_sigma_20"), graph_method="knn", k=4, max_iters=100, sigma=20, n_neighbors=5)

    segment_images("spectral_with_autoencoder", test_images, test_labels, os.path.join(output_path, "spectral_knn_20_sigma_13_enhanced"),
                   graph_method="knn", k=4, max_iters=200, sigma=13.0, n_neighbors=20, hidden_dim=64, epochs=50)


if __name__ == "__main__":
    main()
