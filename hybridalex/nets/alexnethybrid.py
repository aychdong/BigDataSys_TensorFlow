from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .common import ModelBuilder
from .alexnetcommon import alexnet_inference, alexnet_part_conv, alexnet_loss, alexnet_eval
from ..optimizers.momentumhybrid import HybridMomentumOptimizer


def original(images, labels, num_classes, total_num_examples, devices=None, is_train=True):
    """Build inference"""
    if devices is None:
        devices = [None]

    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })

    def train(total_loss, global_step, total_num_steps):
        """Build train operations"""
        # Compute gradients
        with tf.control_dependencies([total_loss]):
            opt = configure_optimizer(global_step, total_num_steps)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        with tf.control_dependencies([apply_gradient_op]):
            return tf.no_op(name='train')

    with tf.device(devices[0]):
        builder = ModelBuilder()
        net, logits, total_loss = alexnet_inference(builder, images, labels, num_classes)

        if not is_train:
            return alexnet_eval(net, labels)

        global_step = builder.ensure_global_step()
        train_op = train(total_loss, global_step, total_num_examples)
    return net, logits, total_loss, train_op, global_step


def ndev_data(images, labels, num_classes, total_num_examples, devices, is_train=True):
    """Build inference, data parallelism"""
    # use the last device in list as variable device
    devices = devices[:]
    builder = ModelBuilder(devices.pop())

    if not is_train:
        with tf.variable_scope('model'):
            prob = alexnet_inference(builder, images, labels, num_classes)[0]
        return alexnet_eval(prob, labels)

    # configure optimizer
    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })

    with tf.device(builder.variable_device()):
        global_step = builder.ensure_global_step()
    opt = configure_optimizer(global_step, total_num_examples)

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
                        prob, logits, total_loss = alexnet_inference(builder, image_slices[idx],
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
