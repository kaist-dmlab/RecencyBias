import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from network.ResNet.utils import *
from tensorpack import *

weight_decay = 0.0005

class ResNet(object):
    def __init__(self, res_n, image_shape, num_labels, scope='ResNet'):

        self.res_n = res_n
        self.image_shape = image_shape
        self.num_labels = num_labels
        self.scope = scope

        [height, width, channels] = image_shape
        train_batch_shape = [None, height, width, channels]

        with tf.variable_scope(self.scope):
            self.train_image_placeholder = tf.placeholder(
                tf.float32,
                shape=train_batch_shape,
                name='train_images'
            )
            self.train_label_placeholder = tf.placeholder(
                tf.int32,
                shape=[None, ],
                name='train_labels'
            )
            test_batch_shape = [None, height, width, channels]
            self.test_image_placeholder = tf.placeholder(
                tf.float32,
                shape=test_batch_shape,
                name='test_images'
            )
            self.test_label_placeholder = tf.placeholder(
                tf.int32,
                shape=[None, ],
                name='test_labels'
            )

    def build_network(self, images, is_training, reuse):
        if is_training:
            keep_prob = 0.7
            with tf.variable_scope(self.scope, reuse=reuse):
                # data augmentation

                if self.image_shape[2] == 1:
                    random_flip = lambda x: tf.image.random_flip_left_right(x, seed=0.5)
                    augmented_images = tf.map_fn(random_flip, images)
                else:
                    random_flip = lambda x: tf.image.random_flip_left_right(x, seed=0.5)
                    random_brightness = lambda x: tf.image.random_brightness(x, max_delta=0.5, seed=0.5)
                    random_hue = lambda x: tf.image.random_hue(x, 0.08, seed=0.5)
                    random_saturation = lambda x: tf.image.random_saturation(x, 0.5, 1.5, seed=0.5)

                    augmented_images = tf.map_fn(random_flip, images)
                    augmented_images = tf.map_fn(random_hue, augmented_images)
                    augmented_images = tf.map_fn(random_saturation, augmented_images)
                    augmented_images = tf.map_fn(random_brightness, augmented_images)
                    augmented_images = tf.image.random_contrast(augmented_images, lower=0.5, upper=1.5, seed=0.5)

                logits = self.inference(augmented_images,is_training, reuse, keep_prob)
        else:
            keep_prob = 0.7
            with tf.variable_scope(self.scope, reuse=reuse):
                logits = self.inference(images,is_training, reuse, keep_prob)

        return tf.nn.softmax(logits), logits

    def inference(self, images, is_training=True, reuse=False, keep_prob = 1.0):

        with tf.variable_scope("network", reuse=reuse):
            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = conv(images, channels=ch, kernel=3, stride=1, scope='conv')
            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, keep_prob=keep_prob, scope='resblock0_' + str(i))
            ########################################################################################################
            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, keep_prob=keep_prob, scope='resblock1_0')
            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, keep_prob=keep_prob, scope='resblock1_' + str(i))
            ########################################################################################################
            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, keep_prob=keep_prob, scope='resblock2_0')
            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, keep_prob=keep_prob, scope='resblock2_' + str(i))
            ########################################################################################################
            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, keep_prob=keep_prob, scope='resblock_3_0')
            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, keep_prob=keep_prob, scope='resblock_3_' + str(i))
            ########################################################################################################

            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)
            x = global_avg_pooling(x)
            logits = fully_conneted(x, units=self.num_labels, scope='logit')

        return logits

    def build_train_op(self, lr_boundaries, lr_values, optimizer_type):
        train_step = tf.Variable(initial_value=0, trainable=False)

        self.train_step = train_step

        prob, logits = self.build_network(self.train_image_placeholder, True, False)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.train_label_placeholder,
            logits=logits
        )

        prediction = tf.equal(tf.cast(tf.argmax(prob, axis=1), tf.int32), self.train_label_placeholder)
        prediction = tf.cast(prediction, tf.float32)

        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        self.train_loss = tf.reduce_mean(loss) + l2_loss * weight_decay
        self.train_accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        self.learning_rate = tf.train.piecewise_constant(train_step, lr_boundaries, lr_values)

        if optimizer_type == "momentum":
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9, use_nesterov=True)
        elif optimizer_type == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        train_vars = [x for x in tf.trainable_variables() if self.scope in x.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.train_loss, global_step=train_step, var_list=train_vars)

        return self.train_loss, self.train_accuracy, train_op, loss, prob

    def build_test_op(self):
        prob, logits = self.build_network(self.test_image_placeholder, False, True)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.test_label_placeholder,
            logits=logits
        )

        prediction = tf.equal(tf.cast(tf.argmax(prob, axis=1), tf.int32), self.test_label_placeholder)
        prediction = tf.cast(prediction, tf.float32)

        self.test_loss = tf.reduce_mean(loss)
        self.test_accuracy = tf.reduce_mean(prediction)

        # variance -> distance
        mean, variance = tf.nn.moments(prob, axes=[1])

        # distance = sign(prediction) * variance
        # sign function : y = 2*prediction - 1
        sign = tf.subtract(tf.scalar_mul(2.0, prediction), 1.0)
        distance = sign * tf.sqrt(variance)

        return self.test_loss, self.test_accuracy, loss, prob

