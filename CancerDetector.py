import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
from model import Model

CKPT_PATH = "model\\model.ckpt"

def detector(path):
    origin = cv.imread(path)
    img = cv.resize(origin, (512, 512))
    data = np.array(img)
    data = np.reshape(data, (-1, 512, 512, 3))

    with tf.Session() as sess:
        model = Model(False)
        saver = tf.train.Saver()
        saver.restore(sess, CKPT_PATH)
        pred = sess.run(model.y_, feed_dict={model.x: data})
    pred = np.argmax(pred, axis=-1)
    pred = np.reshape(pred, (128, 128))
    pred = pred.astype(np.uint8)
    pred = cv.resize(pred, (2048, 2048))

    result = np.array(origin)
    result[:, :, 0] = origin[:, :, 0] + pred*24

    plt.rcParams['figure.dpi'] = 300
    plt.imshow(result)
    plt.show()

testpath = "E:\SC\segmentation\cancer\\00CBD45669DC06735ADC060D2C155C1E.tiff"
detector(testpath)