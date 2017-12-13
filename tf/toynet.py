import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import tensorflow as tf

def build(input_tensor, num_class=4, trainable=True, debug=True):

    net = input_tensor
    if debug: print 'input tensor:', input_tensor.shape

    filters = 32
    num_modules = 5
    with tf.variable_scope('conv'):
        for step in xrange(5):
            stride = 2
            if step: stride = 1
            net = slim.conv2d(inputs        = net,        # input tensor
                              num_outputs   = filters,    # number of filters (neurons) = # of output feature maps
                              kernel_size   = [3,3],      # kernel size
                              stride        = stride,     # stride size
                              trainable     = trainable,  # train or inference
                              activation_fn = tf.nn.relu, # relu
                              scope         = 'conv%da_conv' % step)

            net = slim.conv2d(inputs        = net,        # input tensor
                              num_outputs   = filters,    # number of filters (neurons) = # of output feature maps
                              kernel_size   = [3,3],      # kernel size
                              stride        = 1,          # stride size
                              trainable     = trainable,  # train or inference
                              activation_fn = tf.nn.relu, # relu
                              scope         = 'conv%db_conv' % step)
            if (step+1) < num_modules:
                net = slim.max_pool2d(inputs      = net,    # input tensor
                                      kernel_size = [2,2],  # kernel size
                                      stride      = 2,      # stride size
                                      scope       = 'conv%d_pool' % step)

            else:
                net = tf.layers.average_pooling2d(inputs = net,
                                                  pool_size = [net.get_shape()[-2].value,net.get_shape()[-3].value],
                                                  strides = 1,
                                                  padding = 'valid',
                                                  name = 'conv%d_pool' % step)
            filters *= 2

            if debug: print 'After step',step,'shape',net.shape

    with tf.variable_scope('final'):
        net = slim.flatten(net, scope='flatten')

        if debug: print 'After flattening', net.shape

        net = slim.fully_connected(net, int(num_class), scope='final_fc')

        if debug: print 'After final_fc', net.shape

    return net

# script unit test
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [2,512,512,1])
    net = build(x)

    import sys
    if 'save' in sys.argv:
        # Create a session
        sess = tf.InteractiveSession()
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # Create a summary writer handle + save graph
        writer=tf.summary.FileWriter('toynet_graph')
        writer.add_graph(sess.graph)
