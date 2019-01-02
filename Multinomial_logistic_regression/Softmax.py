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

        # Construct model
        #logits = multilayer_perceptron(X, weights, biases)

        print("layers done")

        # Inputs
        tf_test_dataset = tf.constant(test_dataset)
        print("here1")


        # Define loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        print("here2")
        beta = 0.01
        loss_regularizer = tf.add_n([tf.nn.l2_loss(weights[v]) for v in weights]) * beta
        loss = tf.reduce_mean(loss + loss_regularizer)

        print("here3")


        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #train_op = optimizer.minimize(loss)

        print("here4")

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(multilayer_perceptron(tf_test_dataset, weights, biases))
        cost_history = np.empty(shape=[1], dtype=float)

        print("here5")

        with tf.Session(graph=graph) as session:
            # initialize weights and biases
            tf.global_variables_initializer().run()
            print("Initialized")
            for step in range(num_steps):
                # pick a randomized offset
                offset = np.random.randint(0, train_labels.shape[0] - batch_size - 1)
                print("here6")

                # Generate a minibatch.
                batch_data = train_dataset[offset:(offset + batch_size), :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                print("here7")

                # Prepare the feed dict
                feed_dict = {X: batch_data,
                             Y: batch_labels}
                print("here8")

                # run one step of computation
                _, l, predictions = session.run([loss, train_prediction],
                                                feed_dict=feed_dict)
                print("here9")
                cost_history = np.append(cost_history, session.run(loss, feed_dict=feed_dict))
                print("here10")
                if step % 10 == 0:
                    print("Minibatch loss at step {0}: {1}".format(step, l))
                    print("Minibatch accuracy: {:.1f}%".format(
                        accuracy(predictions, batch_labels)))

            print("\nTest accuracy: {:.1f}%".format(
                accuracy(test_prediction.eval(), test_labels)))

            plt.plot(range(len(cost_history)), cost_history)
            plt.axis([0, num_steps, 0, np.max(cost_history)])
            plt.show()
