################################################
# Test-model using tfrecords data for training
#
#
# Reference: https://medium.com/swlh/using-tfrecords-to-train-a-cnn-on-mnist-aec141d65e3d
################################################
import os
from load_tfrecords import load_from_tfrecords

import tensorflow as tf

DATA='data/'            # Folder to save the tfrecords files to...
BATCH_SIZE = 32

def get_cnn():
  model = tf.keras.Sequential([
      
    tf.keras.layers.Conv2D(kernel_size=3, filters=16, padding='same', activation='relu', input_shape=[28,28, 1]),
    tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    
    tf.keras.layers.Conv2D(kernel_size=3, filters=64, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    
    tf.keras.layers.Conv2D(kernel_size=3, filters=128, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    
    tf.keras.layers.Conv2D(kernel_size=3, filters=256, padding='same', activation='relu'),
    
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10,'softmax')
  ])

  optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model

if __name__=="__main__":
    filename = os.path.join(DATA, "train_01-02-2021_04-26-24_PM.tfrecords")
    loadData = load_from_tfrecords()
    tfr_dataset = loadData.get_dataset(filename, "train")

    model = get_cnn()
    model.summary()

    model.fit(tfr_dataset, steps_per_epoch=60000//BATCH_SIZE, epochs=2)

    filename = os.path.join(DATA, "test_01-02-2021_04-26-24_PM.tfrecords")
    loadData = load_from_tfrecords()
    tfr_testdata = loadData.get_dataset(filename, "test")
    model.evaluate(tfr_testdata, )