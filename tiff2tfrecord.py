import numpy as np
import tensorflow as tf
import cv2 as cv
import os

isize = (512, 512)
lsize = (128, 128)

PATH = "E:\\SC\\segmentation"
CANCERPATH = PATH + '\\cancer'
NONCANCERPATH = PATH + '\\non_cancer'
LABELPATH = PATH + '\\label'


cancer_imgs_f = os.listdir(CANCERPATH)
cancer_imgs_f.sort()
cancer_labels_f = os.listdir(LABELPATH)
cancer_labels_f.sort()
noncancer_imgs_f = os.listdir(NONCANCERPATH)
noncancer_imgs_f.sort()
cwd = "dataset"

tr_writer = tf.python_io.TFRecordWriter(cwd +"\\train.tfrecords")
for i in range(len(cancer_imgs_f)-80):
    img = cv.imread(CANCERPATH+'\\'+cancer_imgs_f[i])
    label = cv.imread(LABELPATH+'\\'+cancer_labels_f[i])
    img = cv.resize(img, isize)
    label = np.sum(label, axis=2)
    label[label > 0] = 1
    label = label.astype(np.uint8)
    w, h, = label.shape[0], label.shape[1]
    mask = np.zeros([w+2, h+2], dtype=np.uint8)
    for sp in [(0, 0), (0, 2047), (2047, 0), (2047, 2047)]:
        if label[sp[0]][sp[1]] == 0:
            cv.floodFill(label, mask, sp, 1, 0, 0, cv.FLOODFILL_FIXED_RANGE)
    tmp = np.zeros([w, h], np.uint8)
    tmp[label == 0] = 1
    label = tmp
    label = cv.resize(label, lsize)
    img = np.array(img)
    label = np.array(label).astype(np.uint8)
    for j in range(4):
        img = np.rot90(img)
        label = np.rot90(label)
        features = tf.train.Features(
            feature = {
                "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
            }
        )
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        tr_writer.write(serialized)
    img = np.fliplr(img)
    label = np.fliplr(label)
    for j in range(4):
        img = np.rot90(img)
        label = np.rot90(label)
        features = tf.train.Features(
            feature = {
                "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
            }
        )
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        tr_writer.write(serialized)

for i in range(len(noncancer_imgs_f) - 20):
    img = cv.imread(NONCANCERPATH + '\\' + noncancer_imgs_f[i])
    img = cv.resize(img, isize)
    img = np.array(img)
    label = np.zeros(lsize, dtype=np.uint8)
    for j in range(4):
        img = np.rot90(img)
        features = tf.train.Features(
            feature = {
                "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
            }
        )
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        tr_writer.write(serialized)
    img = np.fliplr(img)
    for j in range(4):
        img = np.rot90(img)
        features = tf.train.Features(
            feature = {
                "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
            }
        )
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        tr_writer.write(serialized)

te_writer = tf.python_io.TFRecordWriter(cwd +"\\test.tfrecords")
for i in range(len(cancer_imgs_f)-80, len(cancer_imgs_f)):
    img = cv.imread(CANCERPATH + '\\' + cancer_imgs_f[i])
    label = cv.imread(LABELPATH + '\\' + cancer_labels_f[i])
    img = cv.resize(img, isize)
    label = np.sum(label, axis=2)
    label[label > 0] = 1
    label = label.astype(np.uint8)
    w, h, = label.shape[0], label.shape[1]
    mask = np.zeros([w + 2, h + 2], dtype=np.uint8)
    for sp in [(0, 0), (0, 2047), (2047, 0), (2047, 2047)]:
        if label[sp[0]][sp[1]] == 0:
            cv.floodFill(label, mask, sp, 1, 0, 0, cv.FLOODFILL_FIXED_RANGE)
    tmp = np.zeros([w, h], np.uint8)
    tmp[label == 0] = 1
    label = tmp
    label = cv.resize(label, lsize)
    img = np.array(img)
    label = np.array(label).astype(np.uint8)
    for j in range(4):
        img = np.rot90(img)
        label = np.rot90(label)
        features = tf.train.Features(
            feature={
                "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
            }
        )
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        te_writer.write(serialized)
    img = np.fliplr(img)
    label = np.fliplr(label)
    for j in range(4):
        img = np.rot90(img)
        label = np.rot90(label)
        features = tf.train.Features(
            feature={
                "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
            }
        )
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        te_writer.write(serialized)
for i in range(len(noncancer_imgs_f) - 20, len(noncancer_imgs_f)):
    img = cv.imread(NONCANCERPATH + '\\' + noncancer_imgs_f[i])
    img = cv.resize(img, isize)
    img = np.array(img)
    label = np.zeros(lsize, dtype=np.uint8)
    for j in range(4):
        img = np.rot90(img)
        features = tf.train.Features(
            feature={
                "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
            }
        )
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        te_writer.write(serialized)
    img = np.fliplr(img)
    for j in range(4):
        img = np.rot90(img)
        features = tf.train.Features(
            feature={
                "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
            }
        )
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        te_writer.write(serialized)

