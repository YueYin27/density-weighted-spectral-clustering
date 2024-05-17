import pickle
import os
import tarfile
import glob
import cv2
import numpy as np

label_map = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


def unpickle(file):
    with open(file, 'rb') as fo:
        dict_i = pickle.load(fo, encoding='latin1')
    return dict_i


def extract_cifar10_dataset(cifar10_dataset_folder_path, archive_name='cifar-10-python.tar.gz'):
    """Extract and process CIFAR-10 dataset"""

    # Paths for the train and test classification files
    train_txt_path = os.path.join(cifar10_dataset_folder_path, "train_cls.txt")
    test_txt_path = os.path.join(cifar10_dataset_folder_path, "test_cls.txt")

    # Check if train_cls.txt and test_cls.txt already exist
    if os.path.exists(train_txt_path) and os.path.exists(test_txt_path):
        print("train_cls.txt and test_cls.txt already exist. Skipping extraction.")
        return

    # Extract the dataset if not already extracted
    tar_path = os.path.join(cifar10_dataset_folder_path, archive_name)
    if not os.path.exists(os.path.join(cifar10_dataset_folder_path, 'cifar-10-batches-py')):
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=cifar10_dataset_folder_path)
        print("CIFAR-10 dataset extracted.")
    else:
        print("CIFAR-10 dataset already extracted.")

    # Paths for training and testing data
    train_data = glob.glob(os.path.join(cifar10_dataset_folder_path, "cifar-10-batches-py/data_batch_*"))
    test_data = glob.glob(os.path.join(cifar10_dataset_folder_path, "cifar-10-batches-py/test_batch"))

    # Process training data
    count = 0
    with open(train_txt_path, "w") as file_train:
        for batch in train_data:
            dict_i = unpickle(batch)

            labels = dict_i['labels']
            images = dict_i['data']
            filenames = dict_i['filenames']

            for idx, file in enumerate(filenames):
                text_line = '{} {}'.format(file, labels[idx])
                file_train.write(text_line)
                file_train.write("\n")

                img_path = 'train/{}'.format(file)

                save_image = images[idx].reshape(3, 32, 32)
                save_image = save_image.transpose(1, 2, 0)
                # array is RGB. cv2 needs BGR
                save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)

                cv2.imwrite(img_path, save_image)
                count += 1
    print("Training data extracted:", count)  # 50000

    # Process test data
    count = 0
    with open(test_txt_path, "w") as file_test:
        for batch in test_data:
            dict_i = unpickle(batch)

            labels = dict_i['labels']
            images = dict_i['data']
            filenames = dict_i['filenames']

            for idx, file in enumerate(filenames):
                text_line = '{} {}'.format(file, labels[idx])
                file_test.write(text_line)
                file_test.write("\n")

                img_path = 'test/{}'.format(file)

                save_image = images[idx].reshape(3, 32, 32)
                save_image = save_image.transpose(1, 2, 0)
                # array is RGB. cv2 needs BGR
                save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)

                cv2.imwrite(img_path, save_image)
                count += 1
    print("Test data extracted:", count)  # 10000


def load_cifar_10(data_path, class_ids):
    """ Load CIFAR-10 images from specified classes. """

    # Paths for training and testing data
    train_data = glob.glob(os.path.join(data_path, "cifar-10-batches-py/data_batch_*"))
    test_data = glob.glob(os.path.join(data_path, "cifar-10-batches-py/test_batch"))

    def load_images(data_files, class_ids):
        images = []
        labels = []
        for file in data_files:
            dict_i = unpickle(file)
            data = dict_i['data']
            file_labels = dict_i['labels']

            for idx, label in enumerate(file_labels):
                if label in class_ids:
                    img = data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
                    images.append(img)
                    labels.append(label)
        return images, labels

    # Load training images
    train_images, train_labels = load_images(train_data, class_ids)

    # Load testing images
    test_images, test_labels = load_images(test_data, class_ids)

    # Combine training and testing data
    all_images = train_images + test_images
    all_labels = train_labels + test_labels

    # Convert to NumPy arrays
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    return all_images, all_labels
