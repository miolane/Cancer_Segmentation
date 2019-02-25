import tensorflow as tf
import tensorflow.contrib.slim as slim

def fcn(x, is_train):
    net = x
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        trainable=is_train,
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(5e-4),
                        kernel_size=[3, 3],
                        padding='SAME'):
        with slim.arg_scope([slim.conv2d],
                            stride=1,
                            normalizer_fn=slim.batch_norm):
            net = slim.repeat(net, 3, slim.conv2d, 16, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 256
            net = slim.repeat(net, 3, slim.conv2d, 32, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 128
            net = slim.repeat(net, 3, slim.conv2d, 64, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 64
            net = slim.repeat(net, 3, slim.conv2d, 256, scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')  # 32
            net = slim.repeat(net, 3, slim.conv2d, 1024, scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')  # 16

            net = slim.conv2d_transpose(net, num_outputs=512, stride=2, scope='deconv1')  # 32
            net = slim.repeat(net, 3, slim.conv2d, 512, scope='conv6')
            net = slim.conv2d_transpose(net, num_outputs=128, stride=2, scope='deconv2')  # 64
            net = slim.repeat(net, 3, slim.conv2d, 128, scope='conv7')
            net = slim.conv2d_transpose(net, num_outputs=32, stride=2, scope='deconv3')  # 128
            net = slim.repeat(net, 3, slim.conv2d, 32, scope='conv8')
            net = slim.conv2d(net, 2, scope='output', activation_fn=None, normalizer_fn=None)
    return net


class Model(object):
    def __init__(self, is_train):
        self.x = tf.placeholder(tf.float32, [None, 512, 512, 3])
        self.y = tf.placeholder(tf.int32, [None, 128, 128])
        self.fcn = fcn
        self.y_ = self.fcn(self.x, is_train)
        self.optimizer = tf.train.AdamOptimizer()
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_))
        self.train_op = self.optimizer.minimize(self.loss)
