"""
Script that one can use to download the project's data from the Web.

Credits for the data: https://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh
"""

import os
import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt


class IDFormatError(Exception):
    """
    Raised when the ID of the image that is queried is not correct.
    """

    def __init__(self):
        pass


def load_data(data_directory: str = "data/images/"):
    """
    Load images from the data directory.
    """
    data_filenames = os.listdir(data_directory)
    # for data_filename in data_filenames:
    # print(data_filename)
    placeholder_filename = data_filenames[10]
    view_id = int(placeholder_filename[:-4].split("_")[1])
    placeholder = cv2.imread(data_directory + placeholder_filename, cv2.IMREAD_COLOR)

    X = np.zeros(
        shape=(20, 6, placeholder.shape[0], placeholder.shape[1], 3), dtype=np.uint8
    )
    X[0, view_id, :, :] = placeholder

    plt.imshow(X[0, view_id, :, :].astype(np.uint8))
    plt.show()
    return X


def fetch_particular_series(
    image_id: str,
    view_id: int,
    data_directory: str = "data/images/",
    grayscale: bool = True,
):
    """
    Return all data that have the specified ID and view angle.
    """
    if len(image_id) != 6:
        raise IDFormatError
    if view_id >= 0:
        im_filename = str(image_id) + "_" + str(view_id) + ".jpg"
        if grayscale:
            im = cv2.imread(data_directory + im_filename, cv2.IMREAD_GRAYSCALE)
            im[:150, :100] = 0.0
            return im
        else:
            im = cv2.imread(data_directory + im_filename, cv2.IMREAD_COLOR)
            im[:150, :100] = [0.0, 0.0, 0.0]
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
                X[view, :150, :100] = 0.0
            else:
                X[view, :, :] = cv2.imread(
                    data_directory + im_filename, cv2.IMREAD_COLOR
                )
                X[view, :150, :100, :] = [0.0, 0.0, 0.0]
    return X


def load_label(path: str = "data/labels.mat"):
    """
    Load labels: (lat, long) in cartesian coordinates.
    """
    mat = scipy.io.loadmat(path)
    print(mat["GPS_Compass"])


def load_particular_label(image_id: str, path: str = "data/labels.mat"):
    """
    Load labels: (lat, long) in cartesian coordinates.
    """
    mat = scipy.io.loadmat(path)
    image_id = int(image_id)
    (x, y, z) = mat["GPS_Compass"][image_id - 1]
    return (x, y, z)


def plot_series(X):
    """
    Plot a series of views.
    """
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_title("View ID: 0")
    ax[0, 0].imshow(X[0, :, :].astype(np.uint8))
    ax[0, 1].set_title("View ID: 1")
    ax[0, 1].imshow(X[1, :, :].astype(np.uint8))
    ax[0, 2].set_title("View ID: 2")
    ax[0, 2].imshow(X[2, :, :].astype(np.uint8))
    ax[1, 0].set_title("View ID: 3")
    ax[1, 0].imshow(X[3, :, :].astype(np.uint8))
    ax[1, 1].set_title("View ID: 4")
    ax[1, 1].imshow(X[4, :, :].astype(np.uint8))
    ax[1, 2].set_title("View ID: 5")
    ax[1, 2].imshow(X[5, :, :].astype(np.uint8))
    plt.show()


if __name__ == "__main__":
    # load_data()
    load_label()
    # X = fetch_particular_series("009988", -1)
    # plot_series(X)
