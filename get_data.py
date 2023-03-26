"""
This script contains useful functions to manipulate the data.
Credits for the data: https://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh
"""

import os
import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt


class IDFormatError(Exception):
    """
    Exception raised when the ID of the image being queried is not correct (wrong length).
    """

    def __init__(self):
        pass


def fetch_particular_series(
    image_id: str,
    view_id: int,
    data_directory: str = "data/images/",
    grayscale: bool = True,
    crop_pad: bool = True,
):
    """
    Return all data that have the specified ID and view angle.

    - Inputs:
        - image_id: must be a string with 6 characters, like '009900'
        - view_id: either one of {0, 1, 2, 3, 4, 5} or -1 to get all 6 views at once
        - grayscale: set to True to load images in grayscale model, False to get BGR
        - crop_pad : set to True if the loaded images should have the Google Street View navigation pad cropped
    """
    if len(image_id) != 6:
        raise IDFormatError
    if view_id >= 0:
        # used to detect when -1 was queried
        im_filename = str(image_id) + "_" + str(view_id) + ".jpg"
        if grayscale:
            im_loading_format = cv2.IMREAD_GRAYSCALE
        else:
            im_loading_format = cv2.IMREAD_COLOR

        im = cv2.imread(data_directory + im_filename, im_loading_format)
        # crop the top-left corner Google Street View pad
        if crop_pad:
            im[:150, :100] = 0.0
        return im

    else:
        placeholder = cv2.imread(data_directory + "009900_0.jpg")
        # return all the views that match the id
        if grayscale:
            X = np.zeros(shape=(6, placeholder.shape[0], placeholder.shape[1]))
        else:
            X = np.zeros(shape=(6, placeholder.shape[0], placeholder.shape[1], 3))
        for view in range(6):
            im_filename = str(image_id) + "_" + str(view) + ".jpg"
            if grayscale:
                X[view, :, :] = cv2.imread(
                    data_directory + im_filename, cv2.IMREAD_GRAYSCALE
                )
                if crop_pad:
                    X[view, :150, :100] = 0.0
            else:
                X[view, :, :] = cv2.imread(
                    data_directory + im_filename, cv2.IMREAD_COLOR
                )
                if crop_pad:
                    X[view, :150, :100, :] = [0.0, 0.0, 0.0]
    return X


def load_data(num_images: int, data_directory: str = "data/images/"):
    """
    Load the database images from the data directory.

    - Inputs:
        - num_images: number of images to return (from the beginning, the whole database is huge, 70+Go)
    """
    data_filenames = os.listdir(data_directory)
    placeholder_filename = data_filenames[0]
    view_id = int(placeholder_filename[:-4].split("_")[1])
    # load a placeholder to access the data's standard shape
    placeholder = cv2.imread(data_directory + placeholder_filename, cv2.IMREAD_COLOR)

    X = np.zeros(
        shape=(num_images, 6, placeholder.shape[0], placeholder.shape[1], 3),
        dtype=np.uint8,
    )

    count = 0
    for filename in data_filenames:

        view_id = int(placeholder_filename[:-4].split("_")[1])

        im = cv2.imread(data_directory + filename, cv2.IMREAD_COLOR)
        X[count, view_id, :, :] = im
        count += 1
        if count == num_images:
            break

    return X


def load_label(num_labels: int, path: str = "data/labels.mat"):
    """
    Load labels in (lat, long) format. Labels have the following format: (lat, long).

    - Inputs:
        - num_labels: number of labels to load (taken from the beginning)
    """
    mat = scipy.io.loadmat(path)
    labels = mat["GPS_Compass"]
    return labels[:num_labels, :2]


def load_particular_label(image_id: str, path: str = "data/labels.mat"):
    """
    Load the label that corresponds to the image_id given in arg.
    """
    mat = scipy.io.loadmat(path)
    image_id = int(image_id)
    # beware: IDs start at 000001
    (lat, long, _) = mat["GPS_Compass"][image_id - 1]
    return (lat, long)


def plot_series(data):
    """
    Plot a series of views given in arg.
    """
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_title("View ID: 0")
    ax[0, 0].imshow(data[0, :, :].astype(np.uint8))
    ax[0, 1].set_title("View ID: 1")
    ax[0, 1].imshow(data[1, :, :].astype(np.uint8))
    ax[0, 2].set_title("View ID: 2")
    ax[0, 2].imshow(data[2, :, :].astype(np.uint8))
    ax[1, 0].set_title("View ID: 3")
    ax[1, 0].imshow(data[3, :, :].astype(np.uint8))
    ax[1, 1].set_title("View ID: 4")
    ax[1, 1].imshow(data[4, :, :].astype(np.uint8))
    ax[1, 2].set_title("View ID: 5")
    ax[1, 2].imshow(data[5, :, :].astype(np.uint8))
    plt.show()


if __name__ == "__main__":
    # load_data()
    labs = load_label(3)
    print(labs)
    # X = fetch_particular_series("009988", -1)
    # plot_series(X)
