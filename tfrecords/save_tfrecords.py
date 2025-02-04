######################################################
# Save a dataset as a serialized tfrecords file.
# The TFRecords format is a simple format for storing a sequence 
# of binary records.
# Protocoll buffers are a cross-platform, cross-language library for efficient 
# serialization of structured data.
#
# Arasch Lagies
# First Version: 1/18/2021
# Latest Update: 1/18/2021
#
# References:
# https://www.tensorflow.org/tutorials/load_data/tfrecord 
# https://medium.com/swlh/using-tfrecords-to-train-a-cnn-on-mnist-aec141d65e3d
# https://colab.research.google.com/drive/19Ms8CwvTardmte9fk_jBWdBEZB8j5NP_?usp=sharing#scrollTo=VaW9AWWYSc7S
#
######################################################
import tensorflow as tf
import numpy as np
import os
import datetime
import logging
import matplotlib.pyplot as plt

# from skimage import data, color
from skimage.transform import resize
import cv2

# import tensorflow_datasets as tfds
AUTOTUNE = tf.data.experimental.AUTOTUNE

DATA=r'data'            # Folder to save the tfrecords files to...
DATASET = 'mnist'       # This is the TF Datasets data-collection name...

TARGET_HEIGHT = 16  # Height of frames to be saved as tfrecords
TARGET_WIDTH  = 16  # Width of frames to be saved as tfrecords
TARGET_DEPTH  = 1   # Depth of frames to be saved as tfrecords

class save_to_tfrecords:
    def __init__(self, dataFolder=DATA, datasetName=DATASET, timestamp=None):
        # Timestamp for tfrecord files...
        self.timeStamp = datetime.datetime.now().strftime("_%d-%m-%Y_%I-%M-%S_%p") if timestamp is None else timestamp
        # Create the data folder if not yet existent...
        self.folder = dataFolder
        if not os.path.exists(self.folder):
            print(f"[INFO] Creating the data-folder {self.folder} in {__name__}.")
            logging.info(f"[INFO] Creating the data-folder {self.folder} in {__name__}.")
            os.makedirs(self.folder)
        print(f"[INFO] Saving tfrecord files to the folder {self.folder}")
        logging.info(f"[INFO] Saving tfrecord files to the folder {self.folder}")

        print(f"[INFO] Loading the dataset {datasetName}.")
        logging.info(f"[INFO] Loading the dataset {datasetName}.")

        # Test dataset...:
        mnist = tf.keras.datasets.mnist
        (train_images_raw, train_labels), (test_images_raw, test_labels) = mnist.load_data()  
        num_train, _, _ = train_images_raw.shape
        num_test, _, _  = test_images_raw.shape

        print(f"[INFO] The original training frames have following shape: {train_images_raw.shape}")
        print(f"[INFO] The original testing frames have following shape: {test_images_raw.shape}")
        logging.info(f"[INFO] The original training frames have following shape: {train_images_raw.shape}")
        logging.info(f"[INFO] The original testing frames have following shape: {test_images_raw.shape}")
        # Reshaping all images to 16x16:
        train_images = np.zeros((num_train, TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
        test_images  = np.zeros((num_test, TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
        for i, frame in enumerate(train_images_raw):
            train_images[i] = cv2.resize(frame, (TARGET_HEIGHT, TARGET_WIDTH), interpolation=cv2.INTER_CUBIC)
        for i, frame in enumerate(test_images_raw):
            test_images[i] = cv2.resize(frame, (TARGET_HEIGHT, TARGET_WIDTH), interpolation=cv2.INTER_CUBIC)

        # Creating a dict, that contains the training and testing data.
        # The data to be saved in the tfrecords files is organized here.
        self.data = {"train": (train_images, train_labels), "test": (test_images, test_labels)}
        print(f"[INFO] Resized the frames of the training data to: {self.data['train'][0].shape}")
        print(f"[INFO] Resized the frames of the testing data to: {self.data['test'][0].shape}")
        logging.info(f"[INFO] Resized the frames of the training data to: {self.data['train'][0].shape}")
        logging.info(f"[INFO] Resized the frames of the testing data to: {self.data['test'][0].shape}")

    def _bytes_feature(self, value):
        """ Returns a bytes_list from a string / byte """
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """ Returns a float_list from a float / double """
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """ Returns an int64_list from bool / enum / int / unint. """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_array(self, array):
        array = tf.io.serialize_tensor(array)
        return array

    def save_data(self):
        ## for each split create a TFRecordWriter, which writes the parsed
        ## examples to file.
        ## This creates the empty tfrecord files.
        tfrecords_files = []
        for d in self.data:
            print("[INFO] Saving {}:".format(d))
            logging.info("[INFO] Saving {}:".format(d))
            subset = self.data[d]
            
            filename = os.path.join(self.folder, d + self.timeStamp + ".tfrecords")
            writer = tf.io.TFRecordWriter(filename)
            logging.info(f"[INFO] Generated the file {filename}.")
            tfrecords_files.append(filename)
            count = 0

            for image, label in zip(subset[0], subset[1]):
                data={
                    'height': self._int64_feature(TARGET_HEIGHT),
                    'width': self._int64_feature(TARGET_WIDTH),
                    'depth': self._int64_feature(TARGET_DEPTH),
                    'label': self._int64_feature(label),
                    'image_raw': self._bytes_feature(self.serialize_array(image))
                }

                out = tf.train.Example(features=tf.train.Features(feature=data))
                writer.write(out.SerializeToString())
                count += 1
            writer.close()
            print(f"   {count} samples.")
            logging.info(f"   {count} samples.")
        return tfrecords_files


if __name__=="__main__":
    saveData = save_to_tfrecords()
    saveData.save_data()
