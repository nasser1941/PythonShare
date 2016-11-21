'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import cv2
import os
import numpy as np
import json
from pprint import pprint
from Fish import FishClass
from random import shuffle
import copy

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
"""
finalx = np.empty((0,65536), int)
dirAddress = 'train/train/BET/resize'
for filename in os.listdir(dirAddress):
        img = cv2.imread(os.path.join(dirAddress, filename))
        if img==None:
          continue
        height, width, channels = img.shape
        if (height>0 and width>0):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgarray = np.asarray(gray_image).flatten()
            #finalx = finalx.append(imgarray)
            finalx = np.append(finalx, np.array([imgarray]), axis=0)


with open('train/train/ANNOTATION/BET.json') as data_file:
    data = json.load(data_file)
    stringdata = data[0]
    print (stringdata)
 """


def find_between( s, first, last ):
    try:
        start = s.rfind( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

jsonFile = open('train/train/ANNOTATION/BET.json', 'r')
values = json.load(jsonFile)
jsonFile.close()

fishList = []
for i in range (len(values)):
    fish = FishClass()
    items = values[i]
    fish.imageName = find_between(items['filename'], "/", ".")

    dirAddress = 'train/train/BET/resize'
    fish.fishType = 'BET'
    for filename in os.listdir(dirAddress):
        img = cv2.imread(os.path.join(dirAddress, filename))
        if img == None:
            continue
        if(fish.imageName == filename):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fish.fishPixels = np.asarray(gray_image).flatten()
            break


    for criteria in items['annotations']:
        if (criteria['class']=='fish'):
            for key, value in criteria.iteritems():
                fish.fish_X = criteria['x']
                fish.fish_Y = criteria['y']
                fish.fish_H = criteria['height']
                fish.fish_W = criteria['width']
        elif (criteria['class']=='non_fish'):
            for key, value in criteria.iteritems():
                fish.nonfish_X = criteria['x']
                fish.nonfish_Y = criteria['y']
                fish.nonfish_H = criteria['height']
                fish.nonfish_W = criteria['width']

        elif (criteria['class'] == 'head'):
            for key, value in criteria.iteritems():
                fish.head_X = criteria['x']
                fish.head_Y = criteria['y']

        elif (criteria['class'] == 'tail'):
            for key, value in criteria.iteritems():
                fish.tail_X = criteria['x']
                fish.tail_Y = criteria['y']

        elif (criteria['class'] == 'up_fin'):
            for key, value in criteria.iteritems():
                fish.upfin_X = criteria['x']
                fish.upfin_Y = criteria['y']

        elif (criteria['class'] == 'low_fin'):
            for key, value in criteria.iteritems():
                fish.lowfin_X = criteria['x']
                fish.lowfin_Y = criteria['y']

    fishList.append(fish)

fishListTrain = copy.copy(fishList[:160])
print (len(fishListTrain))
fishListTest = copy.copy(fishList[160:])
print (len(fishListTest))
# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 65536 # (img shape: 256*256)
n_classes = 8 # total out puts - x, y, hight, width, tailx, taily, headx, heady
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs (becuse of two pooling layer), 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        shuffle(fishListTrain)
        finalx = np.empty((0, 65536), int)
        finaly = np.empty((0, 8), int)
        for i in range(batch_size):
            finalx = np.append(finalx, np.array([fishList[i].fishPixels]), axis=0)
            oneY = [fishList[i].fish_X, fishList[i].fish_Y, fishList[i].fish_H, fishList[i].fish_W, fishList[i].head_X, fishList[i].head_Y, fishList[i].tail_X, fishList[i].tail_Y]
            finaly = np.append(finaly, np.array([oneY]), axis=0)









        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))
