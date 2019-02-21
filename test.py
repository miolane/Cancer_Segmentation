import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import tfr2dataset, _parse_function, load_batch, show2img, save_result
from model import Model

plt.rcParams['savefig.dpi'] = 300

RECORD_PATH = "dataset\\test.tfrecords"
CKPT_PATH = "model\\model.ckpt"
RESULTPATH = "result"
BATCHSIZE = 8

testset = tfr2dataset(RECORD_PATH)
testset = testset.batch(BATCHSIZE).repeat(1)
iterator = tf.data.Iterator.from_structure(testset.output_types, testset.output_shapes)
nextbatch = iterator.get_next()
dataset_init = iterator.make_initializer(testset)

with tf.Session() as sess:
    sess.run(dataset_init)
    model = Model(False)
    saver = tf.train.Saver()
    saver.restore(sess, CKPT_PATH)

    i = 0
    try:
        while True:
            data, label = sess.run(nextbatch)
            data = np.reshape(data, [-1, 512, 512, 3])
            label = np.reshape(label, [-1, 128, 128])
            y_ = sess.run(model.y_, feed_dict={model.x: data, model.is_istrain: False})
            pred = np.argmax(y_, axis=-1)
            savepath = RESULTPATH + '/' + str(i)
            i += 1
            save_result(data[0], label[0], pred[0], savepath)
    except tf.errors.OutOfRangeError:
        print("Out of range")