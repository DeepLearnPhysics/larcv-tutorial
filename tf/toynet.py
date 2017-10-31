import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import tensorflow as tf

def build(input_tensor, num_class=4, trainable=True, debug=True):

    net = input_tensor
    if debug: print 'input tensor:', input_tensor.shape

    filters = 32

    with tf.variable_scope('conv'):
        for step in xrange(5):

            net = slim.conv2d(inputs      = net,       # input tensor
                              num_outputs = filters,   # number of filters (neurons) = # of output feature maps
                              kernel_size = [3,3],     # kernel size
                              stride      = 1,         # stride size
                              trainable   = trainable, # train or inference
                              scope       = 'conv%d_conv' % step)
            
            net = slim.max_pool2d(inputs      = net,    # input tensor
                                  kernel_size = [3,3],  # kernel size
                                  stride      = 2,      # stride size
                                  scope       = 'conv%d_pool' % step)
            
            filters *= 2

            if debug: print 'After step',step,'shape',net.shape

    with tf.variable_scope('fc'):
        net = slim.flatten(net, scope='flatten')
        if debug: print 'After flattening', net.shape
        
        net = slim.fully_connected(net, 4096, scope='fc1')
        if debug: print 'After fc1', net.shape
        if trainable:
            net = slim.dropout(net, keep_prob=0.5, is_training=trainable, scope='fc1_dropout')
            
        net = slim.fully_connected(net, 4096, scope='fc2')
        if debug: print 'After fc2', net.shape
        if trainable:
            net = slim.dropout(net, keep_prob=0.5, is_training=trainable, scope='fc2_dropout')
            
        net = slim.fully_connected(net, int(num_class), scope='fc_final')
            
        if debug: print 'After fc_final', net.shape

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
