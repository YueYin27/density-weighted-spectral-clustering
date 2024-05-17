import os
import cv2
from src.clustering import *
from src.dataloader import label_map


def segment_images(method, images, gt_labels, output_folder, **kwargs):
    """ Apply specified segmentation method to each image and save the output results in a folder using gt_labels for naming """
    os.makedirs(output_folder, exist_ok=True)

    for idx, (image, gt_label) in enumerate(zip(images, gt_labels)):
        if method == 'kmeans':
            labels, centroids = kmeans(image, **kwargs)
        elif method == 'spectral':
            labels, centroids = spectral_clustering(image, **kwargs)
        else:
            raise ValueError("Unsupported method. Use 'kmeans' or 'spectral'.")

        segmented_image = centroids[labels].astype(np.uint8)  # Replace each label in the labels array with the corresponding centroid's RGB values

        # Use the ground truth label for naming the image
        gt_label_name = label_map[gt_label]

        image_path = os.path.join(output_folder, f"seg_{idx}_{gt_label_name}.png")
        cv2.imwrite(image_path, segmented_image)

    print(f"Segmented images saved to {output_folder}")
