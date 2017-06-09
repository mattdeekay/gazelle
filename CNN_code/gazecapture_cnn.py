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

tf.logging.set_verbosity(tf.logging.INFO)

# For debugging
def tfprint(name, tensor):
  print(name)
  a = tf.Print(tensor, [tensor])
  print(a)


# The learning rate specified as a parameter when running
#   gazecapture_cnn.py; this is used in |cnn_model_fn|.
# Defaults to 0.00001 (10^-5).
LEARNRATE = 0.00001

def cnn_model_fn(feature_cols, labels, mode):
  global learnrate
  features = feature_cols['data']
  hog_input = feature_cols['hog']
  # features = Tensor("Print:0", shape=(?, 144, 144, 3, 4), dtype=float32)
  print ("cnn_model_fn was called! feature size:")
  tfprint("features", tf.shape(features))

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
  # Output Tensor Shape: [batch_size, 144, 144, 96]
  conv_ER1 = tf.layers.conv2d(
      inputs=R_eye,
      filters=96,
      kernel_size=[11,11],
      padding="same",
      activation=tf.nn.relu)
  conv_EL1 = tf.layers.conv2d(
      inputs=L_eye,
      filters=96,
      kernel_size=[11,11],
      padding="same",
      activation=tf.nn.relu)
  conv_F1  = tf.layers.conv2d(
      inputs=face,
      filters=96,
      kernel_size=[11,11],
      padding="same",
      activation=tf.nn.relu)
  # Tensor("Print:0", shape=(?, 144, 144, 96), dtype=float32)

  # Pooling Layer 1
  # Input Tensor Shape: [batch_size, 144, 144, 96]
  # Output Tensor Shape: [batch_size, 72, 72, 96]
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
  # Input Tensor Shape: [batch_size, 72, 72, 96]
  # Output Tensor Shape: [batch_size, 72, 72, 256]
  conv_ER2 = tf.layers.conv2d(
      inputs=conv_ER1_pooled,
      filters=256,
      kernel_size=[5,5],
      padding="same",
      activation=tf.nn.relu)
  conv_EL2 = tf.layers.conv2d(
      inputs=conv_EL1_pooled,
      filters=256,
      kernel_size=[5,5],
      padding="same",
      activation=tf.nn.relu)
  conv_F2  = tf.layers.conv2d(
      inputs=conv_F1_pooled,
      filters=256,
      kernel_size=[5,5],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #3
  # Input Tensor Shape: [batch_size, 72, 72, 256]
  # Output Tensor Shape: [batch_size, 72, 72, 384]
  conv_ER3 = tf.layers.conv2d(
      inputs=conv_ER2,
      filters=384,
      kernel_size=[3,3],
      padding="same",
      activation=tf.nn.relu)
  conv_EL3 = tf.layers.conv2d(
      inputs=conv_EL2,
      filters=384,
      kernel_size=[3,3],
      padding="same",
      activation=tf.nn.relu)
  conv_F3  = tf.layers.conv2d(
      inputs=conv_F2,
      filters=384,
      kernel_size=[3,3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer 2
  # Input Tensor Shape: [batch_size, 72, 72, 384]
  # Output Tensor Shape: [batch_size, 36, 36, 384]
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
  # Input Tensor Shape: [batch_size, 36, 36, 384]
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
        # learning_rate=0.001 # This was original, when training against X/YPts it needs to be smaller below
        learning_rate=LEARNRATE,
        optimizer="SGD")
        #decay_rate=tf.??? Can try to use tf.train.exponential_decay

  # 3. Generate Predictions
  # Remember, |xy_output| returns a [batch_size, 2] Tensor.
  # What is supposed to go in here? Unsure, may be wrong. Do we need anything else in the dictionary?
  predictions = {
      "loss": tf.Print(loss, [loss], name="loss_tensor"),
      "coords delta": tf.subtract(xy_output, labels, name="delta_tensor"),
      "x difference": tf.slice(tf.subtract(xy_output, labels), [0,0], [-1,1], name="xdiff_tensor"),
      "y difference": tf.slice(tf.subtract(xy_output, labels), [0,1], [-1,1], name="ydiff_tensor")
  }

  # Done: Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

###############################

"""
function|gazelle_input_fn|:
    Supplies our x, y.
    This allows for SKCompat compatibility, and also gives us a
    scope to run HOG and supply hog features into the cnn_model_fn.
"""
def gazelle_input_fn(data_name, hog_name, eval_name):
    print ("input_fn was called! this loads everything")
    data = np.load(data_name)[:10,:,:,:,:]
    data_hog = np.load(hog_name)[:10,:,:,:]
    labels = np.load(eval_name)[:10,:]
    
    #mono_face = hog.as_monochrome(data[:,:,:,:,2]) # shape (N, 144,144)
    #data_hog = hog.compute_hog_features(mono_face, pic=12, cib=2, nbins=9)

    feature_cols = {"data": tf.convert_to_tensor(data, dtype=tf.float32),
                    "hog" : tf.convert_to_tensor(data_hog, dtype=tf.float32) }
    return feature_cols, tf.convert_to_tensor(labels, dtype=tf.float32)



##############################
# Helper.
def dataw(num): return CNN_DATA_ROOT + "data" + str(num) + '.npy'
def labelw(num): return CNN_DATA_ROOT + "XYArray" + str(num) + '.npy'
def hogw(num): return CNN_DATA_ROOT + "hog" + str(num) + '.npy'


##############################

def main(argv):
  """ argv = 'gazecapture_cnn.py', [id of train], [id of test/eval], learnrate (optional) """
  train_id, eval_id = argv[1:3]
  global LEARNRATE

  train_data_filename = dataw(train_id)
  train_hog_filename = hogw(train_id)
  train_labels_filename = labelw(train_id)
  eval_data_filename = dataw(eval_id)
  eval_hog_filename = hogw(eval_id)
  eval_labels_filename = labelw(eval_id)
  if len(argv) > 3: LEARNRATE = float(argv[3])

  # Load all the inputs for a given batch into global variables.
  #
    
    
  # Create the Estimator, which encompasses training and evaluation
  gazelle_estimator = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="../tmp/gazelle_conv_model")

  # Set up logging for when the CNN trains
  tensors_to_log = { "loss": "loss_tensor" }
                     #"x diff": "xdiff_tensor",
                     #"y diff": "ydiff_tensor"
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log,
      every_n_iter=5)

  # Train the model.
  gazelle_estimator.fit(
      input_fn=lambda: gazelle_input_fn(train_data_filename, train_hog_filename, train_labels_filename),
      steps=3, # At every step, does it randomly pull out 4 samples from the 364? Can test this tomorrow
      monitors=[logging_hook])

  # Make our own GC accuracy metric
  # Configure the accuracy metric for evaluation
  metrics = {
      "Gazelle prediction mean abs. error":
          learn.MetricSpec(
              metric_fn=tf.metrics.mean_absolute_error, prediction_key="coords delta")
  }

  # Evaluate the model and print results
  eval_results = gazelle_estimator.evaluate(
      input_fn=lambda: gazelle_input_fn(eval_data_filename, eval_hog_filename, eval_labels_filename),
      metrics=metrics)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
