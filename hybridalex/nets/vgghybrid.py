from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .common import ModelBuilder
from .vggcommon import vgg_part_conv, vgg_inference, vgg_loss, vgg_eval


def original(images, labels, num_classes, total_num_examples, devices=None, is_train=True):
    """Build inference"""
    if devices is None:
        devices = [None]

    with tf.device(devices[0]):
        builder = ModelBuilder()
        net, logits, total_loss = vgg_inference(builder, images, labels, num_classes)

        if not is_train:
            return vgg_eval(net, labels)

        global_step = builder.ensure_global_step()
        # Compute gradients
        opt = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = opt.minimize(total_loss, global_step=global_step)

    return net, logits, total_loss, train_op, global_step


def ndev_data(images, labels, num_classes, total_num_examples, devices, is_train=True):
    """Build inference, data parallelism"""
    # use the last device in list as variable device
    devices = devices[:]
    builder = ModelBuilder(devices.pop())

    if not is_train:
        with tf.variable_scope('model'):
            prob = vgg_inference(builder, images, labels, num_classes)[0]
        return vgg_eval(prob, labels)

    global_step = builder.ensure_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=0.01)

    # construct each replica
    replica_grads = []
    with tf.device(builder.variable_device()):
        image_slices = tf.split(images, len(devices), 0)
        label_slices = tf.split(labels, len(devices), 0)
    with tf.variable_scope('model') as vsp:
        # we only want scope for variables but not operations
        with tf.name_scope(''):
            for idx in range(len(devices)):
                dev = devices[idx]
                with tf.name_scope('tower_{}'.format(idx)) as scope:
                    with tf.device(dev):
                        prob, logits, total_loss = vgg_inference(builder, image_slices[idx],
                                                                 label_slices[idx], num_classes,
                                                                 scope)
                        # calculate gradients for batch in this replica
                        grads = opt.compute_gradients(total_loss)

                replica_grads.append(grads)
                # reuse variable for next replica
                vsp.reuse_variables()

    # average gradients across replica
    with tf.device(builder.variable_device()):
        grads = builder.average_gradients(replica_grads)
        apply_grads_op = opt.apply_gradients(grads, global_step=global_step)

        train_op = tf.group(apply_grads_op, name='train')

    # simply return prob, logits, total_loss from the last replica for simple evaluation
    return prob, logits, total_loss, train_op, global_step
