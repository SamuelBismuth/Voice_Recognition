from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# Parameters
learning_rate = 0.05
batch_size = 600
num_steps = 60

# Network Parameters
rnn_layer = 512
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_input = 12960  # Data input (array size)
n_classes = 5  # Total classes (0-4 languages)


# initialize a tensorflow graph
graph = tf.Graph()


# utility function to calculate accuracy
def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.leaky_relu(tf.matmul(x, weights['h1']) + biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.leaky_relu(tf.matmul(layer_1, weights['h2']) + biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def softmax(data):
    train_dataset = np.array([data[i].mfcc for i in range(int(len(data) * 0.70))])
    train_labels = np.array([data[i].accent for i in range(int(len(data) * 0.70))])
    test_dataset = np.array([data[i].mfcc for i in range(int(len(data) * 0.70), int(len(data)))]).astype(np.float32)
    test_labels = np.array([data[i].accent for i in range(int(len(data) * 0.70), int(len(data)))])

    def recurrent_neural_network_model():
        # reshape to [1, n_input]
        x = tf.reshape(X, [-1, n_input])

        # Generate a n_input-element sequence of inputs
        # (eg. [had] [a] [general] -> [20] [6] [33])
        x = tf.split(x, n_input, 1)

        # 1-layer LSTM with n_hidden units.
        rnn_cell = rnn.LSTMCell(rnn_layer)

        # generate prediction
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # there are n_input outputs but
        # we only want the last output
        return tf.matmul(outputs[-1], weights['hrnn']) + biases['brnn']

    with graph.as_default():
        # tf Graph input
        X = tf.placeholder("float", [None, n_input])
        Y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        weights = {
            'hrnn' : tf.Variable(tf.random_normal([rnn_layer, n_classes])),
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        biases = {
            'brnn' : tf.Variable(tf.random_normal([n_classes])),
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        logits = recurrent_neural_network_model()
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            for step in range(1, num_steps + 1):
                batch_x = train_dataset
                batch_y = test_dataset
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % 10 == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    print("hello1")
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y})
                    print("hello2")
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

            print("Optimization Finished!")
            print("Testing Accuracy:", \
                  sess.run(accuracy, feed_dict={X: test_dataset, Y: test_labels}))
