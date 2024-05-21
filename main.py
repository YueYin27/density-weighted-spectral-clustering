import argparse
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
    parser = argparse.ArgumentParser(description='Process and segment images using various clustering methods.')
    parser.add_argument('--method', type=str, default='spectral',
                        help='Segmentation method to use (kmeans, spectral, etc.)')
    parser.add_argument('--output', type=str, help='Output directory for segmented images')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the data directory')
    parser.add_argument('--graph_method', type=str, default='knn', help='Graph construction method for spectral clustering')
    parser.add_argument('--k', type=int, default=4, help='Number of clusters for K-means')
    parser.add_argument('--max_iters', type=int, default=200, help='Maximum number of iterations for K-means')
    parser.add_argument('--sigma', type=float, default=13.0, help='Sigma value for the Gaussian kernel')
    parser.add_argument('--n_neighbors', type=int, default=20, help='Number of neighbors for the KNN graph')
    parser.add_argument('--density_sigma', type=float, default=5.0, help='Sigma value for the density kernel')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for the autoencoder')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training the autoencoder')

    args = parser.parse_args()

    output_path = os.path.join('./testresults', args.output)
    data_path = args.data_path
    method = args.method

    class_ids = [7, 4, 0]  # CIFAR-10 IDs for horse, deer, and airplane

    # Load images from specified classes
    extract_cifar10_dataset(data_path)
    train_images, train_labels, test_images, test_labels = load_cifar_10(data_path, class_ids)  # images (18000, 32, 32, 3), labels (18000,)

    # only use images with index: 26, 59, 129, 250, 335, 352
    test_images = test_images[[2, 11, 13, 18, 26, 27, 37, 47, 48, 59, 60, 82, 84, 85, 129, 143, 167, 189, 223, 245, 282,
                               288,  305, 330, 335, 338, 342, 377, 434, 441, 442, 488, 551, 611, 730, 191, 8, 273, 274,
                               315, 2943, 2739, 2741, 2491, 2424, 2370, 1267]]
    test_labels = test_labels[[2, 11, 13, 18, 26, 27, 37, 47, 48, 59, 60, 82, 84, 85, 129, 143, 167, 189, 223, 245, 282,
                               288,  305, 330, 335, 338, 342, 377, 434, 441, 442, 488, 551, 611, 730, 191, 8, 273, 274,
                               315, 2943, 2739, 2741, 2491, 2424, 2370, 1267]]

    # Check if images were loaded
    if not test_images.size:
        print("No images loaded. Check the dataset and class IDs.")
        return

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    segment_images(method, test_images, test_labels, output_path, graph_method=args.graph_method, k=args.k,
                   max_iters=args.max_iters, sigma=args.sigma, n_neighbors=args.n_neighbors)

    # Process and segment images using K-means
    # segment_images("kmeans", test_images, test_labels, os.path.join(output_path, "kmeans"), k=4, max_iters=200)

    # Process and segment images using Spectral clustering
    # segment_images("spectral", test_images, test_labels,
    #                os.path.join(output_path, "spectral_knn_20_sigma_13"),
    #                graph_method="knn", k=4, max_iters=200, sigma=13.0, n_neighbors=20)

    # segment_images("spectral_with_autoencoder", test_images, test_labels,
    #                os.path.join(output_path, "spectral_knn_10_sigma_13_autoencoder_enhanced"),
    #                graph_method="knn", k=4, max_iters=200, sigma=13.0, n_neighbors=10, hidden_dim=128, epochs=50)
    #
    # segment_images("spectral_with_other_color_space", test_images, test_labels,
    #                os.path.join(output_path, "spectral_knn_20_sigma_13_multi_color_space_enhanced"),
    #                graph_method="knn", k=4, max_iters=200, sigma=13.0, n_neighbors=20)

    # segment_images("spectral_with_multi_resolution", test_images, test_labels,
    #                os.path.join(output_path, "spectral_with_multi_resolution"),
    #                graph_method="knn", k=4, max_iters=200, sigma=13.0, n_neighbors=20)

    # segment_images("spectral_with_density_weighted", test_images, test_labels,
    #                os.path.join(output_path, "spectral_with_density_weighted"),
    #                graph_method="knn", k=4, max_iters=200, sigma=13.0, n_neighbors=20, density_sigma=5.0)


if __name__ == "__main__":
    main()
