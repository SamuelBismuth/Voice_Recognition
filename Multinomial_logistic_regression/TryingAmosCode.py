from __future__ import print_function

import numpy as np
import tensorflow as tf


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_input = 1024  # Data input (array size)
n_classes = 5  # Total classes (0-4 languages)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['h2']) + biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def test(data):
    x=0
    while(x<len(data)):
        
        data[x].mfcc = data[x].mfcc.flatten()
        if(data[x].accent==0):
            data[x].accent=[1, 0, 0, 0, 0]
        elif(data[x].accent==1):
            data[x].accent=[0, 1, 0, 0, 0]
        elif(data[x].accent==2):
            data[x].accent=[0, 0, 1, 0, 0]
        elif(data[x].accent==3):
            data[x].accent=[0, 0, 0, 1, 0]
        elif(data[x].accent==4):
            data[x].accent=[0, 0, 0, 0, 1]
        if (len(data[x].mfcc)!=299*13):
             del data[x] 
        else:
            x=x+1
     

    train_dataset = np.array([data[i].mfcc for i in range(int(len(data) * 0.70))])
    train_labels = np.array([data[i].accent for i in range(int(len(data) * 0.70))])
    test_dataset = np.array([data[i].mfcc for i in range(int(len(data) * 0.70), int(len(data)))]).astype(np.float32)
    test_labels = np.array([data[i].accent for i in range(int(len(data) * 0.70), int(len(data)))])

    print(train_dataset.shape)

    x = tf.placeholder(tf.float32, shape=[None, 299*13])
    y_ = tf.placeholder(tf.float32, shape=[None, 5])

    W_conv1 = tf.Variable(tf.truncated_normal([10, 10, 1, 32], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    x_image = tf.reshape(x, [-1, 299, 13, 1])  # if we had RGB, we would have 3 channels

    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 13, 13, 1], strides=[1, 13, 13, 1], padding='VALID')

    W_conv2 = tf.Variable(tf.truncated_normal([10, 10, 32, 64], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    print(h_pool1)

    #h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    #h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 13, 23, 1], strides=[1, 13, 23, 1], padding='SAME')

    h_pool2_flat = tf.reshape(h_pool1, [-1, 1 * 23 * 32])
    W_fc1 = tf.Variable(tf.truncated_normal([23 * 1 * 32, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    print("hui")
    W_fc2 = tf.Variable(tf.truncated_normal([1024, 5], stddev=0.1))
    print("hui2")

    b_fc2 = tf.Variable(tf.constant(0.1, shape=[5]))

    print("hui3")

    y_conv = multilayer_perceptron(h_fc1_drop)


    print("hui4")

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    print("hui5")

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # uses moving averages momentum
    print("hui6")

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    print("hui7")

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("hui8")

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    print("hello")

    for i in range(400):
        batch = next_batch(128, train_dataset, train_labels)
        if i % 100 == 0:
            print("hello2")
            print(train_labels[0])
            print(batch[0].shape)
            print(batch[1].shape)
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={x: test_dataset, y_: test_labels, keep_prob: 1.0}))



