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
import pickle
from gazelle_utils import *

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):


  # GC Input Layer
  # |features| is a 5-D Tensor, (num_batches, input_idx, 144, 144, 3)
  # the last 3 dimensions are a 144x144 color image, 3 channels
  # we have 4 inputs in the order: right eye, left eye, face, face grid.
  #   Each one must be [batch_size, width, height, channels] = num_batches x 144 x 144 x 3.

  # Unpacking the 4 inputs.
  # tf.slice(tensor, start, lengths)
  R_eye = tf.squeeze( tf.slice(features, [0,0,0,0,0], [-1, 1, 144, 144, 3])) # shape (num, 144,144,3)
  L_eye = tf.squeeze( tf.slice(features, [0,1,0,0,0], [-1, 1, 144, 144, 3]))
  face  = tf.squeeze( tf.slice(features, [0,2,0,0,0], [-1, 1, 144, 144, 3]))
  fgrid = tf.squeeze( tf.slice(features, [0,3,0,0,0], [-1, 1, 144, 144, 3]))


  # Convolutional Layer #1
  # [Nvm, now added Relu again]
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


  # Add a pooling layer


  # Convolutional Layer #2
  # Input Tensor Shape: [batch_size, 144, 144, 96]
  # Output Tensor Shape: [batch_size, 144, 144, 256]
  conv_ER2 = tf.layers.conv2d(
      inputs=conv_ER1,
      filters=256,
      kernel_size=[5,5],
      padding="same",
      activation=tf.nn.relu)
  conv_EL2 = tf.layers.conv2d(
      inputs=conv_EL1,
      filters=256,
      kernel_size=[5,5],
      padding="same",
      activation=tf.nn.relu)
  conv_F2  = tf.layers.conv2d(
      inputs=conv_F1,
      filters=256,
      kernel_size=[5,5],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #3
  # Input Tensor Shape: [batch_size, 144, 144, 256]
  # Output Tensor Shape: [batch_size, 144, 144, 384]
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

  # Add a pooling layer


  # Convolutional Layer #4
  # Input Tensor Shape: [batch_size, 144, 144, 384]
  # Output Tensor Shape: [batch_size, 144, 144, 64]
  conv_ER4 = tf.layers.conv2d(
      inputs=conv_ER3,
      filters=64,
      kernel_size=[1,1],
      padding="same",
      activation=tf.nn.relu)
  conv_EL4 = tf.layers.conv2d(
      inputs=conv_EL3,
      filters=64,
      kernel_size=[1,1],
      padding="same",
      activation=tf.nn.relu)
  conv_F4  = tf.layers.conv2d(
      inputs=conv_F3,
      filters=64,
      kernel_size=[1,1],
      padding="same",
      activation=tf.nn.relu)


  # ----------------------------------

  # Dense Layers: Eyes
  # Flatten tensors into a batch of vectors, then feed to dense layer
  # For each (of the 2 eyes):
  #   Input Tensor Shape (flatten): [batch_size, 144, 144, 64]
  #   Output Tensor Shape (flatten): [batch_size, 144 * 144 * 64]
  # Concatenate:
  #   Final Output Tensor Shape: [batch_size, 144 * 144 * 64 * 2]

  # Dense Layer Eyes: 128 units
  ER_flat = tf.reshape(conv_ER4, [-1, 144 * 144 * 64])
  EL_flat = tf.reshape(conv_EL4, [-1, 144 * 144 * 64])
  eye_flat = tf.concat([ER_flat, EL_flat], axis=1) # concatenate the two along axis=1
  dense_eyes = tf.layers.dense(inputs=eye_flat, units=128, activation=tf.nn.relu)

  # Dense Layers: Face. 128, 64
  F_flat = tf.reshape(conv_F4, [-1, 144 * 144 * 64])
  # Dense Layer Face 1: 128 units
  dense_face1 = tf.layers.dense(inputs=F_flat, units=128, activation=tf.nn.relu)
  # Dense Layer Face 2: 64 units
  dense_face2 = tf.layers.dense(inputs=dense_face1, units=64, activation=tf.nn.relu)

  # Dense Layers: Face Grid boolean mask
  #   fully-connected layers reading the face grid that specifices face location in image
  fgrid_mask = tf.squeeze( tf.slice(fgrid, [0,0,0,0], [-1, 144, 144, 1])) # shape [batch_size, 144,144]
  fgrid_flat = tf.reshape(fgrid_mask, [-1, 144 * 144])
  # Dense Layer Face-grid 1: 256 units
  dense_fgrid1 = tf.layers.dense(inputs=fgrid_flat, units=256, activation=tf.nn.relu)
  # Dense Layer Face-grid 1: 128 units
  dense_fgrid2 = tf.layers.dense(inputs=dense_fgrid1, units=128, activation=tf.nn.relu)


  # Final Dense Layers
  #   concatenate tensors: eyes, face, face-grid.
  combined_flat = tf.concat([dense_eyes, dense_face2, dense_fgrid2], axis=1) # shape (batch_size, ?N_features)
  dense_final = tf.layers.dense(inputs=combined_flat, units=128, activation=tf.nn.relu)
  xy_output = tf.layers.dense(inputs=dense_final, units=2) # (batch_size, 2) <- should be that.
  
  # Debugging:
  # print(xy_output)

  
  # -------------------------------------------------

  loss = None
  train_op = None

  # 1. Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    loss = tf.losses.mean_square_error(labels=labels, predictions=xy_output)

  # 2. Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")
        #decay_rate=tf.??? Can try to use tf.train.exponential_decay

  # 3. Generate Predictions
  # Remember, |xy_output| returns a [batch_size, 2] Tensor.
  # What to do here? Unsure, may be wrong. Do we need anything else in the dictionary?
  predictions = {
      "coordinates": xy_output,
      "squared diff": tf.squared_difference(labels, xy_output,
                                            name="squared_diff_tensor")
  }

  # Done: Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


###############################


def main(unused_argv):
  # We are testing the Gazelle CNN on the *tiny* dataset right now!
  # This means: train_data_tiny, train_labels_tiny, eval_data_tiny, eval_labels_tiny
  # Load training and eval data from GazeCapture dataset
  train_data = pickle.load(open(CNN_DATA_ROOT + 'train_data_tiny.pkl', 'rb'))
  train_labels = pickle.load(open(CNN_DATA_ROOT + 'train_labels_tiny.pkl', 'rb'))
  eval_data = pickle.load(open(CNN_DATA_ROOT + 'eval_data_tiny.pkl', 'rb'))
  eval_labels = pickle.load(open(CNN_DATA_ROOT + 'eval_labels_tiny.pkl', 'rb'))

  # Create the Estimator
  gazelle_estimator = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/gazelle_convnet_model")

  # Set up logging for when the CNN trains
  # Log the values in the tensor named under predictions "squared_diff_tensor"
  #   with label "coords sq.diff loss"
  tensors_to_log = {"coords sq.diff loss": "squared_diff_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=3)

  # Train the model
  gazelle_estimator.fit(
      x=train_data,
      y=train_labels,
      batch_size=3,
      steps=8,
      monitors=[logging_hook])

  # Make our own GC accuracy metric
  # Configure the accuracy metric for evaluation
  metrics = {
      "Gazelle prediction accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.mean_absolute_error, prediction_key="coordinates")
  }

  # Evaluate the model and print results
  eval_results = gazelle_estimator.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
