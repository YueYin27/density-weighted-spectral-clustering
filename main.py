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
    output_path = './results'
    class_ids = [7, 4, 0]  # CIFAR-10 IDs for horse, deer, and airplane

    # Load images from specified classes
    extract_cifar10_dataset(data_path)
    train_images, train_labels, test_images, test_labels = load_cifar_10(data_path, class_ids)  # images (18000, 32, 32, 3), labels (18000,)

    # Check if images were loaded
    if not test_images.size:
        print("No images loaded. Check the dataset and class IDs.")
        return

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Process and segment images using K-means
    segment_images("kmeans", test_images, test_labels, os.path.join(output_path, "kmeans"), k=4, max_iters=100)

    # Process and segment images using Spectral clustering
    segment_images("spectral", test_images, test_labels, os.path.join(output_path, "spectral"), k=4, n_neighbors=5)


if __name__ == "__main__":
    main()