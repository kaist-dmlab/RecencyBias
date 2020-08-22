import os, sys, time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from reader import batch_patcher as patcher
from method import online_sampler as online_sampler
from network.DenseNet.DenseNet import *
from network.ResNet.ResNet import *

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.train_loss_op = None
        self.train_accuracy_op = None
        self.train_op = None
        self.train_xentropy_op = None
        self.train_prob_op = None
        self.test_loss_op = None
        self.test_accuracy_op = None
        self.test_xentropy_op = None
        self.test_prob_op = None

def training(sess, training_epochs, batch_size, train_batch_patcher, validation_batch_patcher, trainer, cur_epoch, method, start_time, sampler=None, training_log=None):

    for epoch in range(training_epochs):
        avg_mini_loss = 0.0
        avg_mini_acc = 0.0

        if method == "warm-up":
            for i in range(train_batch_patcher.num_iters_per_epoch):
                ids, images, labels = train_batch_patcher.get_next_random_mini_batch(batch_size)
                mini_loss, mini_acc, _, xentropy = sess.run([trainer.train_loss_op, trainer.train_accuracy_op, trainer.train_op, trainer.train_xentropy_op], feed_dict={trainer.model.train_image_placeholder: images, trainer.model.train_label_placeholder: labels})
                avg_mini_loss += mini_loss
                avg_mini_acc += mini_acc

        elif method == "online-batch":
            for i in range(train_batch_patcher.num_iters_per_epoch):
                ids, images, labels = train_batch_patcher.get_next_random_mini_batch(batch_size, table=sampler.prob_table.table, normalized=True)
                mini_loss, mini_acc, _, xentropy = sess.run([trainer.train_loss_op, trainer.train_accuracy_op, trainer.train_op, trainer.train_xentropy_op], feed_dict={trainer.model.train_image_placeholder: images, trainer.model.train_label_placeholder: labels})
                avg_mini_loss += mini_loss
                avg_mini_acc += mini_acc

        avg_mini_loss /= train_batch_patcher.num_iters_per_epoch
        avg_mini_acc /= train_batch_patcher.num_iters_per_epoch

        # Training
        avg_train_loss = 0.0
        avg_train_acc = 0.0
        for i in range(train_batch_patcher.num_iters_per_epoch):
            ids, images, labels = train_batch_patcher.get_init_mini_batch(i)
            train_loss, train_acc, train_xentropy = sess.run([trainer.test_loss_op, trainer.test_accuracy_op, trainer.test_xentropy_op], feed_dict={trainer.model.test_image_placeholder: images, trainer.model.test_label_placeholder: labels})
            sampler.async_update_prediction_matrix(ids, train_xentropy)
            avg_train_loss += train_loss
            avg_train_acc += train_acc
        avg_train_loss /= train_batch_patcher.num_iters_per_epoch
        avg_train_acc /= train_batch_patcher.num_iters_per_epoch

        # p_table update
        sampler.update_sampling_probability(epoch + cur_epoch + 1)

        # Validation
        avg_val_loss = 0.0
        avg_val_acc = 0.0
        for i in range(validation_batch_patcher.num_iters_per_epoch):
            ids, images, labels = validation_batch_patcher.get_init_mini_batch(i)
            val_loss, val_acc = sess.run([trainer.test_loss_op, trainer.test_accuracy_op], feed_dict={trainer.model.test_image_placeholder: images, trainer.model.test_label_placeholder: labels})
            avg_val_loss += val_loss
            avg_val_acc += val_acc
        avg_val_loss /= validation_batch_patcher.num_iters_per_epoch
        avg_val_acc /= validation_batch_patcher.num_iters_per_epoch

        # training log
        cur_lr = sess.run(trainer.model.learning_rate)
        print((epoch + cur_epoch + 1), ", ", int(time.time() - start_time), ", ", cur_lr, ", ", avg_mini_loss, ", ", (1.0 - avg_mini_acc), ", ", avg_train_loss, ", ", (1.0 - avg_train_acc), ", ", avg_val_loss, ", ", (1.0 - avg_val_acc))
        if training_log is not None:
            training_log.append(str(epoch + cur_epoch + 1) + ", " + str(int(time.time() - start_time)) + ", " + str(cur_lr) + ", " + str(avg_mini_loss) + ", " + str(1.0 - avg_mini_acc) + ", " + str(avg_train_loss) + ", " + str(1.0 - avg_train_acc) + ", " + str(avg_val_loss) + ", " + str(1.0 - avg_val_acc))


def online_batch(gpu_id, input_reader, model_type, total_epochs, batch_size, lr_boundaries, lr_values, optimizer_type, warm_up, s_es, epochs, log_dir="log", restore=False, pre_trained=10):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    method = "online-batch"

    # log list
    training_log = []
    #training_log.append("epoch, time(s), learning rate, minibatch loss, minibatch error, training loss, training error, test loss, test error")
    start_time = time.time()

    num_train_images = input_reader.num_train_images
    num_test_images = input_reader.num_val_images
    num_label = input_reader.num_classes
    image_shape = [input_reader.resize_height, input_reader.resize_width, input_reader.depth]

    # batch pathcer
    train_batch_patcher = patcher.BatchPatcher(num_train_images, batch_size, num_label)
    test_batch_patcher = patcher.BatchPatcher(num_test_images, batch_size, num_label)

    # online self label correcter
    sampler = online_sampler.Sampler(num_train_images, num_label, s_es, epochs)

    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.visible_device_list = str(gpu_id)
    config.gpu_options.allow_growth = True
    graph = tf.Graph()

    with graph.as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            with tf.Session(config = config) as sess:
                train_ids, train_images, train_labels = input_reader.data_read(batch_size, train = True, normalize = False)
                test_ids, test_images, test_labels = input_reader.data_read(batch_size, train = False, normalize = False)

                if model_type == "DenseNet-25-12":
                    model = DenseNet(25, 12, image_shape, num_label)
                elif model_type == "ResNet-50":
                    model = ResNet(50, image_shape, num_label)

                # register training operations on Trainer class
                trainer = Trainer(model)
                trainer.train_loss_op, trainer.train_accuracy_op, trainer.train_op, trainer.train_xentropy_op, trainer.train_prob_op = model.build_train_op(lr_boundaries, lr_values, optimizer_type)
                trainer.test_loss_op, trainer.test_accuracy_op, trainer.test_xentropy_op, trainer.test_prob_op = model.build_test_op()
                trainer.init_op = tf.global_variables_initializer()

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord = coord)

                # load data set in main memory
                train_batch_patcher.bulk_load_in_memory(sess, train_ids, train_images, train_labels)
                test_batch_patcher.bulk_load_in_memory(sess, test_ids, test_images, test_labels)

                ######################## main methodology for training #######################
                # 1. classify clean samples using prediction certainty

                # init params
                if restore:
                    start_time = time.time()
                    saver = tf.train.Saver()
                    file_dir = "init_weight/" + input_reader.dataset_name + "/" + model_type + "_" + optimizer_type + "_lr=" + str(init_lr) + "_e=" + str(pre_trained) + "/"
                    minus_start_time = 0
                    with open(file_dir + "log.csv") as f:
                        for line in f:
                            print(line, end="")
                            training_log.append(line)
                            minus_start_time = line.split(",")[1]
                    start_time = start_time - float(minus_start_time)
                    saver.restore(sess, file_dir + "model.ckpt")

                else:
                    pre_trained = 0
                    sess.run(trainer.init_op)

                # warm-up
                print("warm_up phase")
                training(sess, warm_up, batch_size, train_batch_patcher, test_batch_patcher, trainer, pre_trained, "warm-up", start_time, sampler=sampler, training_log=training_log)

                print("online-batch phase")
                # self online correction mechanism
                training(sess, total_epochs-warm_up-pre_trained, batch_size, train_batch_patcher, test_batch_patcher, trainer, pre_trained + warm_up, method, start_time, sampler=sampler, training_log=training_log)

                ##############################################################################

                coord.request_stop()
                coord.join(threads)
                sess.close()

    f = open(log_dir + "/log.csv", "w")
    for text in training_log:
        f.write(text + "\n")
    f.close()


