import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2 as cv
import os


def show2img(img1, img2):
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.show()


def save_result(data, label, pred, path):
    grid = plt.GridSpec(2, 3, wspace=0.5, hspace=0.5)
    plt.subplot(grid[:2, :2])
    plt.imshow(data)
    plt.subplot(grid[0, 2])
    plt.imshow(label)
    plt.subplot(grid[1, 2])
    plt.imshow(pred)
    plt.savefig(path)


def _parse_function(example):
    features = {
        "data": tf.FixedLenFeature((), tf.string),
        "label": tf.FixedLenFeature((), tf.string)
    }
    parsed_features = tf.parse_single_example(example, features)
    data = tf.decode_raw(parsed_features['data'], tf.uint8)
    label = tf.decode_raw(parsed_features['label'], tf.uint8)
    return data, label


def tfr2dataset(path):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(_parse_function)
    return dataset


def load_batch(dataset, batchsize=16, epoch=100):
    dataset = dataset.batch(batchsize).repeat(epoch)
    iterator = dataset.make_one_shot_iterator()
    data, label = iterator.get_next()
    return data, label

def reform_data(data, label):
    data = tf.reshape(data, [-1, 512, 512, 3])
    _label = tf.reshape(label, [-1, 128, 128])
    # label = tf.one_hot(_label, 2, axis=2)

    return data, _label


# PATH = "dataset\\test.tfrecords"
# dataset = tfr2dataset(PATH)
# data, label = load_batch(dataset)
# with tf.Session() as sess:
#     label = tf.one_hot(label, 2)
#     data, label = sess.run((data, label))
#     data = np.reshape(data, [-1, 512, 512, 3])
#     label = np.array(label)
#     label = np.reshape(label, [-1, 128, 128, 2]).astype(np.uint8)
#
#     show2img(data[1], label[1, :, :, 1])
#
# pass
