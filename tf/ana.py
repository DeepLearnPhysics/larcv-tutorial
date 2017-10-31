from larcv import larcv
from larcv.dataloader2 import larcv_threadio
import toynet
import numpy as np
import tensorflow as tf
import os,sys,time

# constants
IO_CONFIG  = 'io.cfg'
BATCH_SIZE = 10
LOGDIR     = 'log'
ITERATIONS = 20

SNAPSHOT   = '/home/cadams/larcv-tutorial/tf/toynet-1599'

#
# Step 0: IO
#
train_io = larcv_threadio()        # create io interface 
train_io_cfg = {'filler_name' : 'TrainIO',
                'verbosity'   : 0, 
                'filler_cfg'  : IO_CONFIG}
train_io.configure(train_io_cfg)   # configure
train_io.start_manager(BATCH_SIZE) # start read thread
time.sleep(2)
# retrieve data dimensions to define network later
train_io.next()
dim_data  = train_io.fetch_data('image').dim()
dim_label = train_io.fetch_data('label').dim()

#
# Step 1: Network (same as before!)
#
data_tensor    = tf.placeholder(tf.float32, [None, dim_data[1] * dim_data[2] * dim_data[3]], name='image')
label_tensor   = tf.placeholder(tf.float32, [None, dim_label[1]], name='label')
data_tensor_2d = tf.reshape(data_tensor, [-1,dim_data[1],dim_data[2],dim_data[3]],name='image_reshape')
# Let's keep 10 random set of images in the log
tf.summary.image('input',data_tensor_2d,10)
# build net
net = toynet.build(input_tensor=data_tensor_2d, num_class=dim_label[1], trainable=True, debug=False)
net = tf.nn.softmax(net)
# Define accuracy
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(net,1), tf.argmax(label_tensor,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # tf.summary.scalar('accuracy', accuracy)
# Define loss + backprop as training step
# We aren't training, so there is no need to include the train step:
with tf.name_scope('train'):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor, logits=net))
  # tf.summary.scalar('cross_entropy',cross_entropy)
  # train_step = tf.train.RMSPropOptimizer(0.0003).minimize(cross_entropy)  

#
# Step 2: weight saver & summary writer
# Different from before!
#
# Create a session
sess = tf.InteractiveSession()
# Initialize variables
# Instead of running a global variable initializer, we intialize the weights from
# the file we had
saver = tf.train.Saver()
saver.restore(sess, SNAPSHOT)

verbose = True
#  
# Step 3: Run training loop
#
for i in range(ITERATIONS):

  batch_data  = train_io.fetch_data('image').data()
  batch_label = train_io.fetch_data('label').data()

  feed_dict = { data_tensor  : batch_data,
                label_tensor : batch_label }

  train_io.next()

  _output = sess.run(net, feed_dict=feed_dict)

  # The output is the softmax output of the network, 
  # so we can use argmax to see what the network thinks is most likely

  if verbose:
      selection = np.argmax(_output, axis=1)
      true_answer = np.argmax(batch_label, axis=1)
    
      for n, t in zip(selection, true_answer):
            print "Network selected category {}, true answer was {}.".format(n, t)


# inform log directory
print
print 'Run `tensorboard --logdir=%s` in terminal to see the results.' % LOGDIR
