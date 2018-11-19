# https://www.geeksforgeeks.org/softmax-regression-using-tensorflow/

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# number of features
num_features = 6450
# number of target labels
num_labels = 5
# learning rate (alpha)
learning_rate = 0.05
# batch size
batch_size = 200
# number of epochs
num_steps = 10000
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

        # Inputs
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights = tf.Variable(tf.truncated_normal([num_features, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf_train_labels, logits=logits))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
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
                feed_dict = {tf_train_dataset: batch_data,
                             tf_train_labels: batch_labels}

                # run one step of computation
                _, l, predictions = session.run([optimizer, loss, train_prediction],
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
