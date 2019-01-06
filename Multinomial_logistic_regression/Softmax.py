from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# Parameters
learning_rate = 0.05
batch_size = 600
num_steps = 150

# Network Parameters
n_hidden = 26
n_hidden_1 = 26  # 1st layer number of neurons
n_hidden_2 = 26  # 2nd layer number of neurons
n_input = 3887  # Data input (array size)
n_classes = 5  # Total classes (0-4 languages)

# initialize a tensorflow graph
graph = tf.Graph()



# utility function to calculate accuracy
def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu


def softmax(data):
    print("1")
    x = 0
    while (x < len(data)):
        data[x].mfcc = data[x].mfcc.flatten()
        if (data[x].accent == 0):
            data[x].accent = [1, 0, 0, 0, 0]
        elif (data[x].accent == 1):
            data[x].accent = [0, 1, 0, 0, 0]
        elif (data[x].accent == 2):
            data[x].accent = [0, 0, 1, 0, 0]
        elif (data[x].accent == 3):
            data[x].accent = [0, 0, 0, 1, 0]
        elif (data[x].accent == 4):
            data[x].accent = [0, 0, 0, 0, 1]
        if (len(data[x].mfcc) != 299 * 13):
            del data[x]
        else:
            x = x + 1

    print("2")


    train_dataset = np.array([data[i].mfcc for i in range(int(len(data) * 0.2))])
    train_labels = np.array([data[i].accent for i in range(int(len(data) * 0.2))])
    test_dataset = np.array([data[i].mfcc for i in range(int(len(data) * 0.70), int(len(data)))]).astype(np.float32)
    test_labels = np.array([data[i].accent for i in range(int(len(data) * 0.70), int(len(data)))])

    print("3")


    with graph.as_default():
        # tf Graph input
        X = tf.placeholder("float", [None, n_input])
        Y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        weights = {
            'rnn': tf.Variable(tf.random_normal([n_hidden, n_input])),
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        biases = {
            'brnn': tf.Variable(tf.random_normal([n_input])),
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

        def RNN(x):
            print("5")
            # reshape to [1, n_input]
            x = tf.reshape(x, [-1, n_input])
            print("6")

            # Generate a n_input-element sequence of inputs
            # (eg. [had] [a] [general] -> [20] [6] [33])
            x = tf.split(x, n_input, 1)

            print("7")

            # 1-layer LSTM with n_hidden units.
            rnn_cell = rnn.LSTMCell(n_hidden)

            print("8")

            # generate prediction
            outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
            print("9")

            # there are n_input outputs but
            # we only want the last output
            return tf.matmul(outputs[-1], weights['rnn']) + biases['brnn']

        print("4")

        # Construct model
        lstm = RNN(X)
        print("10")

        logits = multilayer_perceptron(lstm)

        print("11")

        # Inputs
        tf_test_dataset = tf.constant(test_dataset)

        print("12")

        # Define loss and optimizer

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        print("13")

        beta = 0.01
        loss_regularizer = tf.add_n([tf.nn.l2_loss(weights[v]) for v in weights]) * beta
        print("14")

        loss = tf.reduce_mean(loss + loss_regularizer)
        print("15")


        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        print("16")

        train_op = optimizer.minimize(loss)

        print("17")


        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        print("18")

        test_prediction = tf.nn.softmax(multilayer_perceptron(tf_test_dataset))
        print("18")

        cost_history = np.empty(shape=[1], dtype=float)
        print("19")


        with tf.Session(graph=graph) as session:
            # initialize weights and biases
            tf.global_variables_initializer().run()
            print("Initialized")
            for step in range(num_steps):
                # pick a randomized offset
                offset = np.random.randint(0, train_labels.shape[0] - batch_size - 1)
                print("20")

                # Generate a minibatch.
                batch_data = train_dataset[offset:(offset + batch_size), :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                print("21")

                # Prepare the feed dict
                feed_dict = {X: batch_data,
                             Y: batch_labels}
                print("22")

                # run one step of computation
                _, l, predictions = session.run([train_op, loss, train_prediction],
                                                feed_dict=feed_dict)
                print("23")

                cost_history = np.append(cost_history, session.run(loss, feed_dict=feed_dict))
                print("24")

                if step % 10 == 0:
                    print("Minibatch loss at step {0}: {1}".format(step, l))
                    print("Minibatch accuracy: {:.1f}%".format(
                        accuracy(predictions, batch_labels)))

            print("\nTest accuracy: {:.1f}%".format(
                accuracy(test_prediction.eval(), test_labels)))

            plt.plot(range(len(cost_history)), cost_history)
            plt.axis([0, num_steps, 0, np.max(cost_history)])
            plt.show()


