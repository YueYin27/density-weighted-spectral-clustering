import os

import cv2
from tqdm import tqdm

from src.clustering import *
from src.dataloader import label_map


def apply_morphological_operations(labels, kernel_size=3):
    """
    Apply morphological operations to refine cluster labels.

    :param labels: Cluster labels as a 2D array.
    :param kernel_size: Size of the morphological operation kernel.
    :return: Refined labels.
    """
    # Ensure labels are in the correct format
    labels = labels.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Perform opening to remove small objects
    opened_labels = cv2.morphologyEx(labels, cv2.MORPH_OPEN, kernel)
    # Perform closing to close small holes within objects
    closed_labels = cv2.morphologyEx(opened_labels, cv2.MORPH_CLOSE, kernel)
    return closed_labels


def convert_color_space(image, target_space='Lab'):
    """
    Convert the input image to a specified color space.

    :param image: Input image as a 3D NumPy array (height, width, 3).
    :param target_space: Target color space ('HSV', 'Lab', 'YCbCr').
    :return: Image converted to the target color space.
    """
    if target_space == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif target_space == 'Lab':
        return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    elif target_space == 'YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        raise ValueError("Unsupported color space.")


def segment_images(method, images, gt_labels, output_folder, **kwargs):
    """
    Apply specified segmentation method to each image and save the output results in a folder using gt_labels for naming
    :param method: Segmentation method to use ('kmeans' or 'spectral')
    :param images: List of input images as 3D NumPy arrays (height, width, 3).
    :param gt_labels: List of ground truth labels for each image.
    :param output_folder: Folder to save segmented images.
    :param autoencoder: Autoencoder model for spectral clustering with autoencoder.
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
                cv2.imwrite(image_path, cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
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
                cv2.imwrite(image_path, cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
                pbar.update(1)

    elif method == 'spectral_with_autoencoder':
        with tqdm(total=len(images)) as pbar:
            pbar.set_description('Processing Spectral:')
            for idx, (image, gt_label) in enumerate(zip(images, gt_labels)):
                # # Convert the image to the target color space
                # hsv_image = convert_color_space(image, 'HSV')
                # lab_image = convert_color_space(image, 'Lab')
                # ycrcb_image = convert_color_space(image, 'YCrCb')
                # combined_image = np.concatenate((image, hsv_image, ycrcb_image), axis=2)  # 32x32x9
                labels, centroids = spectral_clustering_with_autoencoder(image, image_channel=3, **kwargs)
                segmented_image = centroids[labels].astype(np.uint8)[..., :3]
                gt_label_name = label_map[gt_label]  # Use the ground truth label for naming the image
                image_path = os.path.join(output_folder, f"seg_{idx}_{gt_label_name}.png")
                cv2.imwrite(image_path, cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
                pbar.update(1)

    elif method == 'spectral_with_morphological_operations':
        with tqdm(total=len(images)) as pbar:
            pbar.set_description('Processing Spectral:')
            for idx, (image, gt_label) in enumerate(zip(images, gt_labels)):
                labels, centroids = spectral_clustering(image, **kwargs)

                # Apply morphological operations to refine the segmentation
                labels = apply_morphological_operations(labels, kernel_size=3)

                segmented_image = centroids[labels].astype(
                    np.uint8)  # Replace each label in the labels array with the corresponding centroid's RGB values
                gt_label_name = label_map[gt_label]  # Use the ground truth label for naming the image
                image_path = os.path.join(output_folder, f"seg_{idx}_{gt_label_name}.png")
                cv2.imwrite(image_path, cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
                pbar.update(1)

    elif method == 'spectral_with_other_color_space':
        with tqdm(total=len(images)) as pbar:
            pbar.set_description('Processing Spectral:')
            for idx, (image, gt_label) in enumerate(zip(images, gt_labels)):
                # Convert the image to the target color space
                hsv_image = convert_color_space(image, 'HSV')
                lab_image = convert_color_space(image, 'Lab')
                ycrcb_image = convert_color_space(image, 'YCrCb')
                combined_image = np.concatenate((image, hsv_image, ycrcb_image), axis=2)  # 32x32x9
                labels, centroids = spectral_clustering(combined_image, image_channel=9, **kwargs)

                segmented_image = centroids[labels].astype(np.uint8)[..., :3]
                gt_label_name = label_map[gt_label]
                image_path = os.path.join(output_folder, f"seg_{idx}_{gt_label_name}.png")
                cv2.imwrite(image_path, cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
                pbar.update(1)

    else:
        raise ValueError("Unsupported method")

    print(f"Segmented images saved to {output_folder}")
