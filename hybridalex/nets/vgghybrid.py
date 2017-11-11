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
    # Your code starts here...
    devices_list = devices[:]

    # Use first device as the variable device. 
    builder = ModelBuilder(devices_list[0])
    devices_list = devices_list[1:]

    # If is_not_train, return alexnet_eval()
    if not is_train:
        net, _, _ = vgg_inference(builder, images, labels, num_classes)
        return vgg_eval(net, labels)


    global_step = builder.ensure_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=0.01)

    # construct each replica
    grads_list = []
    with tf.device(builder.variable_device()):
        image_list = tf.split(images, len(devices_list), 0)
        label_list = tf.split(labels, len(devices_list), 0)
    with tf.variable_scope('model') as var_scope:
        with tf.name_scope(''):
            for i in range(len(devices_list)):
                dev = devices_list[i]
                with tf.name_scope('tower_{}'.format(i)) as name_scope:
                    with tf.device(dev):
                        #building up network layers by calling alexnet_inference/vgg_inference using the sub tensors you created in Step 3
                        net, logits, total_loss = vgg_inference(builder, image_list[i], label_list[i], num_classes, name_scope)
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
