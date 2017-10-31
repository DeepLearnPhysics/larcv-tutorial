from larcv import larcv
from larcv.dataloader2 import larcv_threadio
import toynet
import numpy as np
import tensorflow as tf
import os,sys,time

# constants
IO_CONFIG  = 'io.cfg'
BATCH_SIZE = 50
LOGDIR     = 'log'
ITERATIONS = 20000
SAVE_SUMMARY = 20
SAVE_WEIGHTS = 200

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
# Step 1: Network
#
data_tensor    = tf.placeholder(tf.float32, [None, dim_data[1] * dim_data[2] * dim_data[3]], name='image')
label_tensor   = tf.placeholder(tf.float32, [None, dim_label[1]], name='label')
data_tensor_2d = tf.reshape(data_tensor, [-1,dim_data[1],dim_data[2],dim_data[3]],name='image_reshape')
# Let's keep 10 random set of images in the log
tf.summary.image('input',data_tensor_2d,10)
# build net
net = toynet.build(input_tensor=data_tensor_2d, num_class=dim_label[1], trainable=True, debug=False)
# Define accuracy
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(net,1), tf.argmax(label_tensor,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
# Define loss + backprop as training step
with tf.name_scope('train'):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor, logits=net))
  tf.summary.scalar('cross_entropy',cross_entropy)
  train_step = tf.train.RMSPropOptimizer(0.3).minimize(cross_entropy)  

#
# Step 2: weight saver & summary writer
#
# Create a bandle of summary
merged_summary=tf.summary.merge_all()
# Create a session
sess = tf.InteractiveSession()
# Initialize variables
# sess.run(tf.global_variables_initializer())
# Create a summary writer handle
writer=tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)
# Create weights saver
saver = tf.train.Saver()

#  
# Step 3: Run training loop
#
for i in range(ITERATIONS):

  batch_data  = train_io.fetch_data('image').data()
  batch_label = train_io.fetch_data('label').data()

  feed_dict = { data_tensor  : batch_data,
                label_tensor : batch_label }

  train_io.next()

  loss, acc, _ = sess.run([cross_entropy, accuracy, train_step], feed_dict=feed_dict)

  sys.stdout.write('Training in progress @ step %d loss %g accuracy %g          \r' % (i,loss,acc))
  sys.stdout.flush()

  if (i+1)%SAVE_SUMMARY == 0:
    s = sess.run(merged_summary, feed_dict=feed_dict)
    writer.add_summary(s,i)

  if (i+1)%SAVE_WEIGHTS == 0:
    ssf_path = saver.save(sess,'toynet',global_step=i)
    print 'saved @',ssf_path

# inform log directory
print
print 'Run `tensorboard --logdir=%s` in terminal to see the results.' % LOGDIR
