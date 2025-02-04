######################################################
# Load a datasets from tfrecords files.
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
import os
import datetime
import numpy as np
# import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
AUTOTUNE = tf.data.experimental.AUTOTUNE

DATA='data/'            # Folder to save the tfrecords files to...

# BATCH_SIZE = 32

class load_from_tfrecords:
    def __init__( self, filename=None ):
        """ Read one sample from the tfrecords file to get the fix parameters """
        if filename==None:
            pass
        else:
            dataset = tf.data.TFRecordDataset(filename)
            parsed_tfrecords = dataset.map(map_func=self.parse_tfr_elem)
            for tfrecord in parsed_tfrecords:
                self.h, self.w, self.d = tfrecord[0].shape
                break

    def parse_tfr_elem(self, element):
        """ Create a dictionary
            with all the fileds we want to use from the example dataset
            Note: The last field is of type tf.string (even though it was 
            written as a bytes list). All other fields are initialized with the
            same type as during saving...
        """
        parse_dict = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string)
        }
        ## Obtain the original image data:
        example_message = tf.io.parse_single_example(element, parse_dict)

        img_raw = example_message['image_raw']
        height  = example_message['height']
        width   = example_message['width']
        depth   = example_message['depth']
        label   = example_message['label']

        # Set uint8 as the datatype as mnist is scaled between 0 and 255.
        # If the frames contain floats, this would need to be float64.
        feature = tf.io.parse_tensor(img_raw, out_type=tf.uint8)
        feature = tf.reshape(feature, shape=[height, width, depth])   # This load function performs a reshape of the frames (adding depth)...

        return (feature, label)

    def decode(self, serialized_example):
        """ https://sebastianwallkoetter.wordpress.com/2018/02/24/optimize-tf-input-pipeline/ """
        features = tf.parse_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        #image.set_shape((mnist.IMAGE_PIXELS))

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int32)
        
        return image, label

    def augment(self, image, label):
        """ https://sebastianwallkoetter.wordpress.com/2018/02/24/optimize-tf-input-pipeline/ """
        # OPTIONAL: Could reshape into a 28×28 image and apply distortions
        # here.  Since we are not applying any distortions in this
        # example, and the next step expects the image to be flattened
        # into a vector, we don't bother.
        return image, label

    def normalize(self, image, label):
        """ https://sebastianwallkoetter.wordpress.com/2018/02/24/optimize-tf-input-pipeline/ """
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1. / 255) # – 0.5
        return image, label

    def get_dimensions( self ):
        return self.h, self.w, self.d

    def get_dataset(self, filename, set_type, batch):
        """ Create a dataset around the TFRecordd files.
            The function parse_tfr_elem() is to get a single example.
            This function creates a TFRecordDataset to map all examples to this function.
        """
        ignore_order = tf.data.Options()
        # Disable native order, increase speed
        ignore_order.experimental_deterministic = False
        dataset = tf.data.TFRecordDataset(filename)

        dataset = dataset.with_options(
            ignore_order
        )
        # The AUTOTUNE optimizer allows to automatically determine how many 
        # examples we can process in parallel. This can reduce GPU idle times.
        dataset = dataset.map(
            self.parse_tfr_elem, num_parallel_calls = AUTOTUNE
        )
        # Shuffle the data, set the batch size, and set repeat with no 
        # argument; this means to repeat endlessly.
        # This requires us to remember how many examples our dataser has.
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.repeat() if (set_type=='trainBase') else dataset

        # map takes a python function and applies it to every sample
        # https://sebastianwallkoetter.wordpress.com/2018/02/24/optimize-tf-input-pipeline/
        # dataset = dataset.map(self.decode)
        # dataset = dataset.map(self.augment)
        dataset = dataset.map(self.normalize) if ( set_type=='trainBase' or \
            set_type=='testBase' or \
                set_type=='valid' or set_type=='train_quant' or \
                    set_type=='test_quant') else dataset
        dataset = dataset.unbatch() if ( set_type=='train_quant' or \
            set_type=='test_quant' ) else dataset
        return dataset


if __name__=="__main__":
    """ Above we created a Dataset and mapped a data-generating 
        function to it. This function looks as a sanity check
        at one sample of the dataset.
        It returns two tensors -- for mnist (32,28,28,1) because 
        we took one batch of size 32 --, the second Tensor is of shape
        (32,) since we have 32 labels, one per example in our batch)
    """
    filename = os.path.join(DATA, "train_01-02-2021_04-26-24_PM.tfrecords")
    loadData = load_from_tfrecords()
    tfr_dataset = loadData.get_dataset(filename, "train")
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=plt.figaspect(1/2))
    for sample in tfr_dataset.take(1):
        frame = sample[0].numpy()
        print( frame[0,:,:,0] )
        plt.imshow( frame[0,:,:,0], cmap='gray' )
        print(f"In this dataset the MIN value is {sample[0].numpy().min()} ")
        print(f" and the MAX value is {sample[0].numpy().max()})")
        # ax.imshow(sample[0,:,:,0].numpy(), cmap='gray')
        # ax.set(title=train_labels[i])
        plt.show()


 