"""  
Plot the spatial distribution of data in the dataset.
"""

from get_data import load_label

import os
import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt


# constants
N_images_in_dataset = 10343
locations = {
    "Manhattan": [40.776676, -73.971321],
    "Orlando": [28.538336, -81.379234],
    "Pittsburgh": [40.440624, -79.995888],
}


def get_id_location_mapping():
    """
    Return a dictionnary with correspondance between (ID, location)
    """
    id_location_mapping = {}

    count = 0
    labels = load_label(N_images_in_dataset)
    for label in labels:
        closest_delta = np.inf
        best_location = None
        for location, gps in locations.items():
            # beware: the following is NOT a Cartesian distance (not a Cartesian reference frame)
            delta = (label[0] - gps[0]) ** 2 + (label[1] - gps[1]) ** 2
            if delta < closest_delta:
                closest_delta = delta
                best_location = location

        id_location_mapping[count] = best_location
        count += 1
    return id_location_mapping


def plot_id_location_mapping(id_location_mapping):
    """
    Plot the distribution ID/location.
    """
    ids = range(N_images_in_dataset)
    locs = []
    for index in ids:
        if id_location_mapping[index] == "Manhattan":
            locs.append(1)
        elif id_location_mapping[index] == "Pittsburgh":
            locs.append(2)
        else:
            # Orlando
            locs.append(3)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ids, locs, "bx")
    ax.set_title("Distribution des localisations en fonction des ID du jeu de données")
    ax.text(
        4300,
        2.65,
        "Orlando, FL \n (28.538336, -81.379234)",
        style="italic",
        bbox={"facecolor": "white", "alpha": 0.5, "pad": 10},
    )
    ax.text(
        450,
        2.2,
        "Pittsburgh, PA \n (40.440624, -79.995888)",
        style="italic",
        bbox={"facecolor": "white", "alpha": 0.5, "pad": 10},
    )
    ax.text(
        6000,
        1.2,
        "Manhattan, NY \n (40.776676, -73.971321)",
        style="italic",
        bbox={"facecolor": "white", "alpha": 0.5, "pad": 10},
    )
    ax.set_xlabel("ID")
    ax.set(yticklabels=[])
    ax.tick_params(axis="x", rotation=50)
    plt.show()


id_loc_mapping = get_id_location_mapping()
plot_id_location_mapping(id_loc_mapping)
