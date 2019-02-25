import tensorflow as tf
import numpy as np
from utils import tfr2dataset
from model import Model


RECORD_PATH = "dataset\\train.tfrecords"
BATCHSIZE = 13
EPOCH = 100
BUFFERSIZE = 10400

CKPT_PATH = "model\\model.ckpt"
RESTORE = False

SUMMARY_PATH = "logs"

trainset = tfr2dataset(RECORD_PATH)
trainset = trainset.shuffle(buffer_size=BUFFERSIZE)
trainset = trainset.prefetch(buffer_size=BUFFERSIZE)
trainset = trainset.batch(BATCHSIZE).repeat(EPOCH)
iterator = tf.data.Iterator.from_structure(trainset.output_types, trainset.output_shapes)
nextbatch = iterator.get_next()
dataset_init = iterator.make_initializer(trainset)


step = 0
with tf.Session() as sess:
    sess.run(dataset_init)
    model = Model(True)
    tf.summary.scalar("Loss", model.loss)
    saver = tf.train.Saver()
    merge = tf.summary.merge_all()
    writer = tf.summary.FileWriter(SUMMARY_PATH, sess.graph)
    if RESTORE:
        saver.restore(sess, CKPT_PATH)
    else:
        sess.run(tf.global_variables_initializer())

    try:
        while True:
            data, label = sess.run(nextbatch)
            data = np.reshape(data, [-1, 512, 512, 3])
            label = np.reshape(label, [-1, 128, 128])
            sess.run(model.train_op,
                     feed_dict={model.x: data, model.y: label})
            summary = sess.run(merge, feed_dict={model.x: data, model.y: label})
            writer.add_summary(summary, step)
            step += 1
            if step % 800 == 0:
                print("Saving at %d"%step)
                saver.save(sess, CKPT_PATH)
                print("Done")
    except tf.errors.OutOfRangeError:
        saver.save(sess, CKPT_PATH)
        print("Finish training")

