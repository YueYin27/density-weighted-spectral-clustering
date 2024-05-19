import os
import cv2
from src.clustering import *
from src.dataloader import label_map
from tqdm import tqdm


def segment_images(method, images, gt_labels, output_folder, **kwargs):
    """
    Apply specified segmentation method to each image and save the output results in a folder using gt_labels for naming
    :param method: Segmentation method to use ('kmeans' or 'spectral')
    :param images: List of input images as 3D NumPy arrays (height, width, 3).
    :param gt_labels: List of ground truth labels for each image.
    :param output_folder: Folder to save segmented images.
    :param kwargs: Additional keyword arguments for the segmentation method.
    """
    os.makedirs(output_folder, exist_ok=True)

    if method == 'kmeans':
        with tqdm(total=len(images)) as pbar:
            pbar.set_description('Processing Kmeans:')
            for idx, (image, gt_label) in enumerate(zip(images, gt_labels)):
                labels, centroids = kmeans(image, **kwargs)
                segmented_image = centroids[labels].astype(
                    np.uint8)  # Replace each label in the labels array with the corresponding centroid's RGB values
                gt_label_name = label_map[gt_label]  # Use the ground truth label for naming the image
                image_path = os.path.join(output_folder, f"seg_{idx}_{gt_label_name}.png")
                cv2.imwrite(image_path, segmented_image)
                pbar.update(1)

    elif method == 'spectral':
        with tqdm(total=len(images)) as pbar:
            pbar.set_description('Processing Spectral:')
            for idx, (image, gt_label) in enumerate(zip(images, gt_labels)):
                labels, centroids = spectral_clustering(image, **kwargs)
                segmented_image = centroids[labels].astype(
                    np.uint8)  # Replace each label in the labels array with the corresponding centroid's RGB values
                gt_label_name = label_map[gt_label]  # Use the ground truth label for naming the image
                image_path = os.path.join(output_folder, f"seg_{idx}_{gt_label_name}.png")
                cv2.imwrite(image_path, segmented_image)
                pbar.update(1)

    elif method == 'spectral_enhanced':
        with tqdm(total=len(images)) as pbar:
            pbar.set_description('Processing Improved Spectral:')
            for idx, (image, gt_label) in enumerate(zip(images, gt_labels)):
                labels, centroids = spectral_clustering_enhanced(image, **kwargs)
                segmented_image = centroids[labels].astype(
                    np.uint8)
                gt_label_name = label_map[gt_label]
                image_path = os.path.join(output_folder, f"seg_{idx}_{gt_label_name}.png")
                cv2.imwrite(image_path, segmented_image)
                pbar.update(1)

    else:
        raise ValueError("Unsupported method. Use 'kmeans' or 'spectral'.")

    print(f"Segmented images saved to {output_folder}")
