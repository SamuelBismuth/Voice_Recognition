# https://www.geeksforgeeks.org/softmax-regression-using-tensorflow/
from random import shuffle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# number of features - n_input
num_features = 6450
# number of target labels - n_classes
num_labels = 5
# learning rate (alpha)
learning_rate = 0.05
# batch size
batch_size = 200
# number of epochs
num_steps = 1000
# initialize a tensorflow graph
graph = tf.Graph()


# utility function to calculate accuracy
def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu


def softmax(data):
    train_dataset = np.array([data[i].mfcc for i in range(int(len(data)*0.70))])
    train_labels = np.array([data[i].accent for i in range(int(len(data)*0.70))])
    test_dataset = np.array([data[i].mfcc for i in range(int(len(data)*0.70), int(len(data)))]).astype(np.float32)
    test_labels = np.array([data[i].accent for i in range(int(len(data)*0.70), int(len(data)))])

    with graph.as_default():
        """ 
        defining all the nodes 
        """
        # Parameters
        
        
        display_step = 1
        
        # Network Parameters
        n_hidden_1 = 256 # 1st layer number of neurons
        n_hidden_2 = 256 # 2nd layer number of neurons
        
        
        
        # tf Graph input
        X = tf.placeholder("float", [batch_size, num_features])
        Y = tf.placeholder("float", [batch_size, num_labels])
        
        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([num_features, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, num_labels]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([num_labels]))
        }
        
        
        # Create model
        def multilayer_perceptron(x):
            # Hidden fully connected layer with 256 neurons
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            # Hidden fully connected layer with 256 neurons
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            # Output fully connected layer with a neuron for each class
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            return out_layer
        
        # Construct model
        logits = multilayer_perceptron(X)
          
        # Inputs
        tf_test_dataset = tf.constant(test_dataset)




        
        # Define loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        regularizer = tf.nn.l2_loss(weights)
        loss = tf.reduce_mean(loss + 0.01 * regularizer)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)

       

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(multilayer_perceptron(tf_test_dataset))
        cost_history = np.empty(shape=[1], dtype=float)

        with tf.Session(graph=graph) as session:
            # initialize weights and biases
            tf.global_variables_initializer().run()
            print("Initialized")
            for step in range(num_steps):
                # pick a randomized offset
                offset = np.random.randint(0, train_labels.shape[0] - batch_size - 1)

                # Generate a minibatch.
                batch_data = train_dataset[offset:(offset + batch_size), :]
                batch_labels = train_labels[offset:(offset + batch_size), :]

                # Prepare the feed dict
                feed_dict = {X: batch_data,
                             Y: batch_labels}

                # run one step of computation
                _, l, predictions = session.run([train_op, loss, train_prediction],
                                                feed_dict=feed_dict)
                cost_history = np.append(cost_history, session.run(loss, feed_dict=feed_dict))
                if step % 200 == 0:
                    print("Minibatch loss at step {0}: {1}".format(step, l))
                    print("Minibatch accuracy: {:.1f}%".format(
                        accuracy(predictions, batch_labels)))

            print("\nTest accuracy: {:.1f}%".format(
                accuracy(test_prediction.eval(), test_labels)))

            plt.plot(range(len(cost_history)), cost_history)
            plt.axis([0, num_steps, 0, np.max(cost_history)])
            plt.show()
