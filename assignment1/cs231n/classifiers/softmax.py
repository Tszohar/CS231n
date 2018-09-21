import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, native implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  dW = np.random.randn(3073, 500) * 0.0001
  loss_vec = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  step_size=0.01
  #Forward
  # multiply data by a matrix in order to create scores for each class
  f1 = X.dot(W)
  f1 -= np.max(f1)

  softmax = np.zeros_like(f1)
  for image_idx in range(f1.shape[0]):
    for class_idx in range(f1.shape[1]):
      #softmax[image_idx, class_idx] = np.exp(f1[image_idx, class_idx]) / np.sum(np.exp(f1.shape[1]), axis=1)
      softmax[image_idx, class_idx] = np.exp(f1[image_idx, class_idx]) / np.sum(np.exp(f1.shape[1]))
  #for i in range(y.shape[0]):
    # i = 0, 1, 2, 3
  #for i in y:
    # i = y[0], y[1], y[2]

  for i in range(y.shape[0]):
    loss_vec[i] = -np.log(softmax[i, y[i]])

  loss_vec = reg * (1/2 * loss_vec**2)

  loss = np.mean(loss_vec)
  # mean of Li over all training examples together with a regularization term R(W)


  # Backward
  for image_idx in range(f1.shape[0]):
    for class_idx in range(f1.shape[1]):
      if ( class_idx == y[class_idx]):
        dW[image_idx, class_idx] = np.exp(f1[image_idx, class_idx]) / np.sum(np.exp(f1.shape[1])) - 1 + ( reg * dW[image_idx, class_idx])
      else:
        dW[image_idx, class_idx] = 1 / np.sum(np.exp(f1.shape[1])) + ( reg * dW[image_idx, class_idx])




  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################


  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #f1=W.dot(X)
  #f1 -= np.max(f1)
  #softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
  """
   Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
   it for the linear classifier. These are the same steps as we used for the
   SVM, but condensed to a single function.
   """
  # Load the raw CIFAR-10 data
  cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

  # subsample the data
  mask = list(range(num_training, num_training + num_validation))
  X_val = X_train[mask]
  y_val = y_train[mask]
  mask = list(range(num_training))
  X_train = X_train[mask]
  y_train = y_train[mask]
  mask = list(range(num_test))
  X_test = X_test[mask]
  y_test = y_test[mask]
  mask = np.random.choice(num_training, num_dev, replace=False)
  X_dev = X_train[mask]
  y_dev = y_train[mask]

  # Preprocessing: reshape the image data into rows
  X_train = np.reshape(X_train, (X_train.shape[0], -1))
  X_val = np.reshape(X_val, (X_val.shape[0], -1))
  X_test = np.reshape(X_test, (X_test.shape[0], -1))
  X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

  # Normalize the data: subtract the mean image
  mean_image = np.mean(X_train, axis=0)
  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image
  X_dev -= mean_image

  # add bias dimension and transform into columns
  X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
  X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
  X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
  X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

  return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

if __name__ == "__main__":
  # from __future__ import print_function

  import os
  import random
  import numpy as np
  from cs231n.data_utils import load_CIFAR10
  # from cs231n.classifiers.softmax import softmax_loss_naive
  import matplotlib.pyplot as plt
  import time
  os.chdir("../../")

  # Load the raw CIFAR-10 data
  cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

  # Invoke the above function to get our data.
  X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()

  print('Train data shape: ', X_train.shape)
  print('Train labels shape: ', y_train.shape)
  print('Validation data shape: ', X_val.shape)
  print('Validation labels shape: ', y_val.shape)
  print('Test data shape: ', X_test.shape)
  print('Test labels shape: ', y_test.shape)
  print('dev data shape: ', X_dev.shape)
  print('dev labels shape: ', y_dev.shape)


  # Generate a random softmax weight matrix and use it to compute the loss.
  W = np.random.randn(3073, 10) * 0.0001
  loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

  # As a rough sanity check, our loss should be something close to -log(0.1).
  print('loss: %f' % loss)
  print('sanity check: %f' % (-np.log(0.1)))