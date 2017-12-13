from larcv import larcv
from larcv.dataloader2 import larcv_threadio
import toynet
#import resnet as toynet
import numpy as np
import tensorflow as tf
import os,sys,time
#
# Configurations
#
TRAIN_IO_CONFIG  = 'io_train.cfg'
TEST_IO_CONFIG   = 'io_test.cfg'
TRAIN_BATCH_SIZE = 50
TEST_BATCH_SIZE  = 100
LOGDIR = 'log'
ITERATIONS = 10000
SAVE_SUMMARY = 20
SAVE_WEIGHTS = 100

# Check log directory is empty
train_logdir = os.path.join(LOGDIR,'train')
test_logdir  = os.path.join(LOGDIR,'test')
if not os.path.isdir(train_logdir): os.makedirs(train_logdir)
if not os.path.isdir(test_logdir):  os.makedirs(test_logdir)
if len(os.listdir(train_logdir)) or len(os.listdir(test_logdir)):
  sys.stderr.write('Error: train or test log dir not empty...\n')
  raise OSError

#
# Step 0: IO
#
train_io = larcv_threadio()        # create io interface 
train_io_cfg = {'filler_name' : 'TrainIO',
                'verbosity'   : 0, 
                'filler_cfg'  : TRAIN_IO_CONFIG}
train_io.configure(train_io_cfg)   # configure
train_io.start_manager(TRAIN_BATCH_SIZE) # start read thread
time.sleep(2)
# retrieve data dimensions to define network later
train_io.next()
dim_data  = train_io.fetch_data('train_image').dim()
dim_label = train_io.fetch_data('train_label').dim()

test_io = larcv_threadio()        # create io interface
test_io_cfg = {'filler_name' : 'TestIO',
               'verbosity'   : 0,
               'filler_cfg'  : TEST_IO_CONFIG}
test_io.configure(test_io_cfg)   # configure
test_io.start_manager(TEST_BATCH_SIZE) # start read thread
time.sleep(2)
# retrieve data dimensions to define network later
test_io.next()

#
# Step 1: Network
#
data_tensor    = tf.placeholder(tf.float32, [None, dim_data[1] * dim_data[2] * dim_data[3]], name='image')
label_tensor   = tf.placeholder(tf.float32, [None, dim_label[1]], name='label')
data_tensor_2d = tf.reshape(data_tensor, [-1,dim_data[1],dim_data[2],dim_data[3]],name='image_reshape')
# Let's keep 10 random set of images in the log
tf.summary.image('input',data_tensor_2d,10)
# build net
net = toynet.build(input_tensor=data_tensor_2d, num_class=dim_label[1], trainable=True, debug=True)
# Define accuracy
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(net,1), tf.argmax(label_tensor,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
# Define loss + backprop as training step
with tf.name_scope('train'):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor, logits=net))
  tf.summary.scalar('cross_entropy',cross_entropy)
  train_step = tf.train.RMSPropOptimizer(0.0001).minimize(cross_entropy)  
  #train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)  

#
# Step 2: weight saver & summary writer
#
# Create a bandle of summary
merged_summary=tf.summary.merge_all()
# Create a session
sess = tf.InteractiveSession()
# Initialize variables
sess.run(tf.global_variables_initializer())
# Create a summary writer handle
writer_train=tf.summary.FileWriter(train_logdir)
writer_train.add_graph(sess.graph)
writer_test=tf.summary.FileWriter(test_logdir)
writer_test.add_graph(sess.graph)
# Create weights saver
saver = tf.train.Saver()

#  
# Step 3: Run training loop
#
for i in range(ITERATIONS):

  train_data  = train_io.fetch_data('train_image').data()
  train_label = train_io.fetch_data('train_label').data()

  feed_dict = { data_tensor  : train_data,
                label_tensor : train_label }

  loss, acc, _ = sess.run([cross_entropy, accuracy, train_step], feed_dict=feed_dict)
  sys.stdout.write('Training in progress @ step %d loss %g accuracy %g          \n' % (i,loss,acc))
  sys.stdout.flush()

  if (i+1)%SAVE_SUMMARY == 0:
    # Save train log
    #s = sess.run(merged_summary, feed_dict=feed_dict)
    #writer_train.add_summary(s,i)

    # Calculate & save test log
    test_data  = test_io.fetch_data('test_image').data()
    test_label = test_io.fetch_data('test_label').data()
    feed_dict  = { data_tensor  : test_data,
                   label_tensor : test_label }
    loss, acc = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
    sys.stdout.write('Testing in progress @ step %d loss %g accuracy %g          \n' % (i,loss,acc))
    sys.stdout.flush()
    #s = sess.run(merged_summary, feed_dict=feed_dict)
    #writer_test.add_summary(s,i)

    test_io.next()

  train_io.next()

  if (i+1)%SAVE_WEIGHTS == 0:
    ssf_path = saver.save(sess,'weights/toynet',global_step=i)
    print 'saved @',ssf_path

# inform log directory
print
print 'Run `tensorboard --logdir=%s` in terminal to see the results.' % LOGDIR
train_io.reset()
