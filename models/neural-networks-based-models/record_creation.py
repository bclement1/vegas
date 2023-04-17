import tensorflow as tf
import pandas as pd
import os
import numpy as np
import tqdm
import scipy.io

paths = [
    "D:/CV_Data/DATA/4CHANNELS/test/",
    "D:/CV_Data/DATA/4CHANNELS/train/",
    "D:/CV_Data/DATA/4CHANNELS/small/test/",
    "D:/CV_Data/DATA/4CHANNELS/small/train/",
]
# df = pd.read_csv("D:/CV_Data/DATA/ASSEMBLY/train_labels.csv")
mat = scipy.io.loadmat("D:\CV_Data\GPS_Long_Lat_Compass.mat")

def tf_creation(data_path,mat):
    # Create a TFRecordWriter
    print(data_path)
    writer = tf.io.TFRecordWriter(data_path + "tfrecord_2D.tfrecord")
    i = 0
    # Iterate over the list of arrays and labels
    for file in tqdm.tqdm(os.listdir(data_path)):
        if file.split(".")[-1] == "npz":
            array = np.load(data_path + file)["xtrain"]
            # Convert the array to a bytes-like object
            array_bytes = array.tobytes()
            label = mat["GPS_Compass"][
                int("".join(file.split(".")[:-1]))-1][:2w]
            # Create features containing the array and label
            label_bytes = label.tobytes()
            feature = {
                "array": tf.train.Feature(bytes_list=tf.train.BytesList(value=[array_bytes])),
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
            }

            # Create a Example message
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize the example message and write it to the TFRecord file
            writer.write(example.SerializeToString())

    # Close the writer
    writer.close()


def parse_function(example_proto):
    features = {"array": tf.io.FixedLenFeature([], tf.string), "label": tf.io.FixedLenFeature([], tf.string)}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    array = tf.io.decode_raw(parsed_features["array"], np.int8)
    label = tf.io.decode_raw(parsed_features["label"], np.float64)
    array = tf.reshape(array, [1024, 5120, 3])
    label = tf.reshape(label, [2])
    # label = tf.reshape(parsed_features["label"], [1])
    # label = tf.cast(label, tf.float64)
    return array, label


if __name__ == "__main__":
    for path in paths:
        tf_creation(path,mat)

    tfrecord_dataset = tf.data.TFRecordDataset(
        "D:/CV_Data/DATA/4CHANNELS/small/test/tfrecord_2D.tfrecord"
)
    dataset = tfrecord_dataset.map(parse_function)
    print("loaded")

    for element in dataset.as_numpy_iterator():
        print(np.shape(element[1]))
        print(np.shape(element[0]))
        print(type(element[1]))
        # print(np.shape(element[0]))
