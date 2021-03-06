"""
Stanford CS 231A Computer Vision
Owen Wang, Natalie Ng, Matthew Kim

Same as the one taken from |mnist_sample_cnn.py|, but tweaked to fit
GazeCapture's iTracker CNN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# Specific to Gazelle.
import pickle # may be deprecated soon
from gazelle_utils import *

tf.logging.set_verbosity(tf.logging.ERROR)

# For debugging. This prints.
# THIS SHIT BROKEN
def tfprint(tensor, message=None):
  if message is not None: print(message)
  with tf.Session() as sess:
    a = sess.run([tensor])
    print(a)
            


# The learning rate specified as a parameter when running
#   gazecapture_cnn.py; this is used in |cnn_model_fn|.

LEARNRATE = 0.00001  # Defaults to 0.00001 (10^-5).
MINIBATCH_SIZE = 20
NUM_SAMPLES = None
BATCH_GSTEP = 0

TRAIN_DATA = None
TRAIN_HOG = None
TRAIN_LABEL = None


def cnn_model_fn(feature_cols, labels, mode):
  global LEARNRATE

  features = feature_cols['data']
  hog_input = feature_cols['hog']
  # features = Tensor("Print:0", shape=(?, 144, 144, 3, 4), dtype=float32)
  print ("cnn_model_fn was called, in mode '%s'" % mode)
  #print ("features, hog, labels size:")
  #tfprint(tf.shape(features))
  #tfprint(tf.shape(hog_input))
  #tfprint(tf.shape(labels))

  # Input Layer
  # we have 4 inputs in the order: right eye, left eye, face, face grid (bound by dim #4 of value 4)
  #   Each one is [batch_size, width, height, channels] = num_batches x 144 x 144 x 3.

  R_eye = tf.squeeze(tf.slice(features, [0,0,0,0,0], [-1, 144, 144, 3, 1]), axis=4)
  L_eye = tf.squeeze(tf.slice(features, [0,0,0,0,1], [-1, 144, 144, 3, 1]), axis=4)
  face  = tf.squeeze(tf.slice(features, [0,0,0,0,2], [-1, 144, 144, 3, 1]), axis=4)
  fgrid = tf.squeeze(tf.slice(features, [0,0,0,0,3], [-1, 144, 144, 3, 1]), axis=4)
  # Tensor("Print_1:0", shape=(?, 1, 144, 144, 3), dtype=float32)
  
  # Convolving the Eyes and Face
  # ============================
  # Convolutional Layer #1
  # Input Tensor Shape: [batch_size, 144, 144, 3]
  # Output Tensor Shape: [batch_size, 144, 144, 48]
  conv_ER1 = tf.layers.conv2d(
      inputs=R_eye,
      filters=48,
      kernel_size=[11,11],
      padding="same",
      activation=tf.nn.relu)
  conv_EL1 = tf.layers.conv2d(
      inputs=L_eye,
      filters=48,
      kernel_size=[11,11],
      padding="same",
      activation=tf.nn.relu)
  conv_F1  = tf.layers.conv2d(
      inputs=face,
      filters=48,
      kernel_size=[11,11],
      padding="same",
      activation=tf.nn.relu)
  # Tensor("Print:0", shape=(?, 144, 144, 48), dtype=float32)

  # Pooling Layer 1
  # Input Tensor Shape: [batch_size, 144, 144, 48]
  # Output Tensor Shape: [batch_size, 72, 72, 48]
  conv_ER1_pooled = tf.layers.max_pooling2d(
      inputs=conv_ER1,
      pool_size=[2,2],
      strides=2)
  conv_EL1_pooled = tf.layers.max_pooling2d(
      inputs=conv_EL1,
      pool_size=[2,2],
      strides=2)
  conv_F1_pooled = tf.layers.max_pooling2d(
      inputs=conv_F1,
      pool_size=[2,2],
      strides=2)

  # Convolutional Layer #2
  # Input Tensor Shape: [batch_size, 72, 72, 48]
  # Output Tensor Shape: [batch_size, 72, 72, 128]
  conv_ER2 = tf.layers.conv2d(
      inputs=conv_ER1_pooled,
      filters=128,
      kernel_size=[5,5],
      padding="same",
      activation=tf.nn.relu)
  conv_EL2 = tf.layers.conv2d(
      inputs=conv_EL1_pooled,
      filters=128,
      kernel_size=[5,5],
      padding="same",
      activation=tf.nn.relu)
  conv_F2  = tf.layers.conv2d(
      inputs=conv_F1_pooled,
      filters=128,
      kernel_size=[5,5],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #3
  # Input Tensor Shape: [batch_size, 72, 72, 128]
  # Output Tensor Shape: [batch_size, 72, 72, 192]
  conv_ER3 = tf.layers.conv2d(
      inputs=conv_ER2,
      filters=192,
      kernel_size=[3,3],
      padding="same",
      activation=tf.nn.relu)
  conv_EL3 = tf.layers.conv2d(
      inputs=conv_EL2,
      filters=192,
      kernel_size=[3,3],
      padding="same",
      activation=tf.nn.relu)
  conv_F3  = tf.layers.conv2d(
      inputs=conv_F2,
      filters=192,
      kernel_size=[3,3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer 2
  # Input Tensor Shape: [batch_size, 72, 72, 192]
  # Output Tensor Shape: [batch_size, 36, 36, 192]
  conv_ER3_pooled = tf.layers.max_pooling2d(
      inputs=conv_ER3,
      pool_size=[2,2],
      strides=2)
  conv_EL3_pooled = tf.layers.max_pooling2d(
      inputs=conv_EL3,
      pool_size=[2,2],
      strides=2)
  conv_F3_pooled  = tf.layers.max_pooling2d(
      inputs=conv_F3,
      pool_size=[2,2],
      strides=2)

  # Convolutional Layer #4
  # Input Tensor Shape: [batch_size, 36, 36, 192]
  # Output Tensor Shape: [batch_size, 36, 36, 64]
  conv_ER4 = tf.layers.conv2d(
      inputs=conv_ER3_pooled,
      filters=64,
      kernel_size=[1,1],
      padding="same",
      activation=tf.nn.relu)
  conv_EL4 = tf.layers.conv2d(
      inputs=conv_EL3_pooled,
      filters=64,
      kernel_size=[1,1],
      padding="same",
      activation=tf.nn.relu)
  conv_F4  = tf.layers.conv2d(
      inputs=conv_F3_pooled,
      filters=64,
      kernel_size=[1,1],
      padding="same",
      activation=tf.nn.relu)

  # Convolving the HOG of the Face
  # ==============================
  # input: [batch_size, 11, 11, 36]
  # output: [batch_size, 11, 11, 64]
  conv_HOG = tf.layers.conv2d(
      inputs=hog_input,
      filters=64,
      kernel_size=[3,3],
      padding="same",
      activation=tf.nn.relu)

  # ----------------------------------
  

  # Dense Layer Eyes (together): 128 units
  ER_flat = tf.reshape(conv_ER4, [-1, 36 * 36 * 64])
  EL_flat = tf.reshape(conv_EL4, [-1, 36 * 36 * 64])
  eye_flat = tf.concat([ER_flat, EL_flat], axis=1)
  dense_eyes = tf.layers.dense(inputs=eye_flat, units=128, activation=tf.nn.relu)

  # Face dense layer 1: 128 units, dense layer 2: 64 units
  F_flat = tf.reshape(conv_F4, [-1, 36 * 36 * 64])
  dense_face1 = tf.layers.dense(inputs=F_flat, units=128, activation=tf.nn.relu)
  dense_face2 = tf.layers.dense(inputs=dense_face1, units=64, activation=tf.nn.relu)

  # Face-grid dense layer 1: 256 units, dense layer 2: 128 units
  fgrid_mask = tf.slice(fgrid, [0,0,0,0], [-1, 144, 144, 1])
  fgrid_pooled = tf.layers.max_pooling2d(
      inputs=fgrid_mask,
      pool_size=[2,2],
      strides=2)
  fgrid_flat = tf.reshape(fgrid_pooled, [-1, 72 * 72])
  dense_fgrid1 = tf.layers.dense(inputs=fgrid_flat, units=128, activation=tf.nn.relu)
  dense_fgrid2 = tf.layers.dense(inputs=dense_fgrid1, units=64, activation=tf.nn.relu)

  # HOG dense layer 1: 256 units, layer 2: 64 units
  hog_flat = tf.reshape(conv_HOG, [-1, 11 * 11 * 64])
  dense_hog1 = tf.layers.dense(inputs=hog_flat, units=256, activation=tf.nn.relu)
  dense_hog2 = tf.layers.dense(inputs=dense_hog1, units=64, activation=tf.nn.relu)
    
  # Final Dense Layers
  #combined_flat = tf.concat([dense_eyes, dense_face2, dense_fgrid2, dense_hog2], axis=1)
  combined_flat = tf.concat([dense_eyes, dense_face2, dense_fgrid2], axis=1)
  # combined_flat shape: [batch_size, 128 + 64 + 64 + 64]
  dense_final = tf.layers.dense(inputs=combined_flat, units=128, activation=tf.nn.relu)
  xy_output = tf.layers.dense(inputs=dense_final, units=2)
  # Tensor("Print:0", shape=(?, 2), dtype=float32)

  
  # -------------------------------------------------

  loss = None
  train_op = None

  # 1. Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    loss = tf.sqrt(tf.losses.mean_squared_error(labels=labels, predictions=xy_output))
    
  # 2. Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=LEARNRATE,
        optimizer="SGD")

  # 3. Generate Predictions
  # Remember, |xy_output| returns a [batch_size, 2] Tensor.
  predictions = {
      "loss": tf.Print(loss, [loss], name="loss_tensor"),
      "coords delta": tf.subtract(xy_output, labels, name="delta_tensor")
      #"x difference": tf.slice(tf.subtract(xy_output, labels), [0,0], [-1,1], name="xdiff_tensor"),
      #"y difference": tf.slice(tf.subtract(xy_output, labels), [0,1], [-1,1], name="ydiff_tensor")
  }

  # Done: Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


##############################
# Helper.
def dataw(num): return CNN_DATA_ROOT + "data" + str(num) + '.npy'
def labelw(num): return CNN_DATA_ROOT + "XYArray" + str(num) + '.npy'
def hogw(num): return CNN_DATA_ROOT + "hog" + str(num) + '.npy'


###############################

# Load input data globally, so we can implement mini-batches.
#    This is only run once at the beginning of main().

def load_training_batch_data_globally(train_id):
    train_data_filename = dataw(train_id)
    train_hog_filename = hogw(train_id)
    train_label_filename = labelw(train_id)
    
    global TRAIN_DATA
    global TRAIN_HOG
    global TRAIN_LABEL
    TRAIN_DATA  = np.load(train_data_filename)
    TRAIN_HOG   = np.load(train_hog_filename)
    TRAIN_LABEL = np.load(train_label_filename)

    # return the number of images in the training batch
    return TRAIN_DATA.shape[0]


###############################

"""
function|gazelle_input_fn|:
    Supplies our x, y.
    This allows for SKCompat compatibility, and also gives us a
    scope to run HOG and supply hog features into the cnn_model_fn.
"""
def gazelle_input_fn(data, hog, label, mode):
    global BATCH_GSTEP
    global MINIBATCH_SIZE
    global NUM_SAMPLES
    
    assert NUM_SAMPLES is not None
    chopstart = int(BATCH_GSTEP * MINIBATCH_SIZE)
    chopend = min(chopstart + int(MINIBATCH_SIZE), NUM_SAMPLES)

    data = data[chopstart:chopend, :,:,:,:]
    hog = hog[chopstart:chopend, :,:,:]
    label = label[chopstart:chopend, :]

    print ("input_fn called, returned data segment [%s, %s)" % (chopstart, chopend))

    
    feature_cols = {"data": tf.convert_to_tensor(data, dtype=tf.float32),
                    "hog" : tf.convert_to_tensor(hog, dtype=tf.float32) }

    BATCH_GSTEP += 1
    return feature_cols, tf.convert_to_tensor(label, dtype=tf.float32)


##############################

def main(argv):
  """ argv = 'gazecapture_cnn.py', [id of train], [id of test/eval], learnrate (optional) """
  global MINIBATCH_SIZE  
  global LEARNRATE
  global TRAIN_DATA
  global TRAIN_HOG
  global TRAIN_LABEL
  global NUM_SAMPLES
  
  mode = argv[1] # this parameter is 'train', 'validation' or 'test'
  assert mode == 'test' or mode == 'train' or mode == 'validation'
  print ("Mode has been detected as '%s'" % mode)

  train_id, eval_id = argv[2:4]
  if len(argv) > 4:
        LEARNRATE = float(argv[4])
        print ("Learn rate has been set to", LEARNRATE)

  # Load all the training inputs for a given batch into global variables.
  if mode == 'train':
    assert train_id != 0
    num = load_training_batch_data_globally(train_id)
    print ("Detected training batch %s w.size %s" % (train_id, num))
  else:
    assert eval_id != 0
    eval_data  = np.load(dataw(eval_id))
    eval_hog   = np.load(hogw(eval_id))
    eval_label = np.load(labelw(eval_id))
    num = eval_data.shape[0]
    print ("Detected evaluation (test/vali.) batch %s w.size %s" % (eval_id, num))
  NUM_SAMPLES = int(num)


  num_steps = int(num / MINIBATCH_SIZE) + 1
  print ("minibatch size %s -> dynamically calculated steps %s" % (MINIBATCH_SIZE, num_steps))
  

  # --------------------------------------------------------
  
  # Set up logging for when the CNN trains
  tensors_to_log = { "loss": "loss_tensor" }
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log,
      every_n_iter=1)

  # Create the Estimator, which encompasses training and evaluation
  gazelle_estimator = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="../tmp/gazelle_conv_model")

  # Train the model.
  if mode == 'train':
    gazelle_estimator.fit(
        input_fn=lambda: gazelle_input_fn(TRAIN_DATA, TRAIN_HOG, TRAIN_LABEL, mode=mode),
        steps=num_steps,
        monitors=[logging_hook])
  else: # 'validation' or 'test'
      metrics = { # Configure the accuracy metric for test (eval) / validation
          "Gazelle prediction mean abs. x/y direction error":
              learn.MetricSpec(
                  metric_fn=tf.metrics.mean_absolute_error, prediction_key="coords delta")
      }
      # Evaluate the model and print results
      eval_results = gazelle_estimator.evaluate(
          input_fn=lambda: gazelle_input_fn(eval_data, eval_hog, eval_label, mode=mode),
          steps=num_steps,
          metrics=metrics)
      print(eval_results)


if __name__ == "__main__":
  with tf.Session() as sess:
    with tf.device("/gpu:0"):
      tf.app.run()
