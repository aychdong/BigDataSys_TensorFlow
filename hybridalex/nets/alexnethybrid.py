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
    # Your code starts here...
    devices_list = devices[:]

    # Use first device as the variable device. 
    builder = ModelBuilder(devices_list[0])
    devices_list = devices_list[1:]

    # If is_not_train, return alexnet_eval()
    if not is_train:
        net, _, _ = alexnet_inference(builder, images, labels, num_classes)
        return alexnet_eval(net, labels)

    #Create global steps on the parameter server node. You can use the same method that the single-machine program uses to perform this step.
    with tf.device(builder.variable_device()):
        global_step = builder.ensure_global_step()

    #Configure the optimizer using the global step created and the total examples number user passed into ndev_data.
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
    opt = configure_optimizer(global_step, total_num_examples)

    #Construct graph replica by splitting the original tensors into sub tensors. You need to create replica for both images and labels. Take a look at how tf.split works.
    grads_list = []
    with tf.device(builder.variable_device()):
        image_list = tf.split(0, len(devices_list), images)
        label_list = tf.split(0, len(devices_list), labels)
    with tf.variable_scope('model') as var_scope:
        with tf.name_scope(''):
            for i in range(len(devices)):
                dev = devices[i]
                with tf.name_scope('tower_{}'.format(i)) as name_scope:
                    with tf.device(dev):
                        #building up network layers by calling alexnet_inference/vgg_inference using the sub tensors you created in Step 3
                        net, logits, total_loss = alexnet_inference(builder, image_slices[i],label_slices[i], num_classes,name_scope)
                        # calculate gradients for batch in this replica
                        tmp_grads = opt.compute_gradients(total_loss)

                grads_list.append(tmp_grads)
                # reuse variable for next replica
                var_scope.reuse_variables()

    #On the parameter server node, calculate the average accross the gradients you collect. You should call average_gradents method definded in the ModuleBuilder class for this step. This method will transfer gradients from the worker nodes to the parameter server.
    with tf.device(builder.variable_device()):
        gradients = builder.average_gradients(grads_list)
        apply_gradient_op = opt.apply_gradients(gradients, global_step=global_step)
        train_op = tf.group(apply_gradient_op, name='train')

    #Finally, return the same parameters as the single-machine version code does.
    return net, logits, total_loss, train_op, global_step
