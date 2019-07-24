import os
from random import choices

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from matplotlib.image import imread
from sklearn.utils import shuffle


def download_mnist():
    (x, y), (x_test, y_test) = mnist.load_data()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=10_000)

    print(x_val.shape, x_train.shape, x_test.shape)

    x_train = x_train.astype(np.uint8)
    x_val = x_val.astype(np.uint8)
    x_test = x_test.astype(np.uint8)
    #
    train_path = os.path.abspath('mnist/data/train_2.tfrecords')
    val_path = os.path.abspath('mnist/data/valid_2.tfrecords')
    test_path = os.path.abspath('mnist/data/test_2.tfrecords')
    #
    convert_to_records(x_train, y_train, train_path)
    convert_to_records(x_val, y_val, val_path)
    convert_to_records(x_test, y_test, test_path)


def download_not_mnist():
    base_dir = 'mnist/data/not_mnist'
    labels = 'ABCDEFGHIJ'

    class_num = 10
    num_by_class = 4000
    num = num_by_class * class_num
    dimensions = 784
    test_size = 10000

    x, y = np.zeros((num, dimensions), dtype='uint8'), np.zeros((num,), dtype='uint8')

    for i, label in enumerate(labels):
        class_files = os.listdir(os.path.join(base_dir, label))
        files = choices(class_files, k=num_by_class)
        images = [imread(os.path.join(base_dir, label, file)) for file in files]
        x[i*num_by_class:(i+1)*num_by_class] = 255*np.reshape(images, (-1, dimensions))
        y[i*num_by_class:(i+1)*num_by_class] = [i]*num_by_class
    x, y = shuffle(x, y)

    save_not_mnist(x, y, test_size)


def save_not_mnist(x, y, test_size):
    train_path = os.path.abspath('mnist/data/not_mnist_train.tfrecords')
    test_path = os.path.abspath('mnist/data/not_mnist_test.tfrecords')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    convert_to_records(x_train, y_train, train_path)
    convert_to_records(x_test, y_test, test_path)


def convert_to_records(x,y,path):
    print('writing to {}'.format(path))
    with tf.python_io.TFRecordWriter(path) as writer:
        for i in range(x.shape[0]):
            example = tf.train.Example(features = tf.train.Features(
                            feature =
                            {
                                'image/encoded':_bytes_feature(x[i].tostring()),
                                'image/class/label':_int64_feature(int(y[i]))
                            }
                                                                    )
                                      )
            writer.write(example.SerializeToString())
            if i % 5000 == 0:
                print('writing {}th image'.format(i))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    # download_mnist()
    download_not_mnist()
