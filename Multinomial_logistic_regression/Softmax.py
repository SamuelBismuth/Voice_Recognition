from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# Parameters

#unrolled through 28 time steps
time_steps=60
#hidden LSTM units
num_units=128
learning_rate = 0.05
batch_size = 600
num_steps = 100

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_input = 216  # Data input (array size)
n_classes = 5  # Total classes (0-4 languages)

# initialize a tensorflow graph
graph = tf.Graph()


# utility function to calculate accuracy
def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu


def softmax(data):
    train_dataset = np.array([data[i].mfcc for i in range(int(len(data) * 0.70))])
    train_labels = np.array([data[i].accent for i in range(int(len(data) * 0.70))])
    test_dataset = np.array([data[i].mfcc for i in range(int(len(data) * 0.70), int(len(data)))]).astype(np.float32)
    test_labels = np.array([data[i].accent for i in range(int(len(data) * 0.70), int(len(data)))])

    with graph.as_default():
        # tf Graph input
        X = tf.placeholder("float", [None,time_steps, n_input])
        Y = tf.placeholder("float", [None, n_classes])
        
        #weights and biases of appropriate shape to accomplish above task
        out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
        out_bias=tf.Variable(tf.random_normal([n_classes]))
        
        input=tf.unstack(X ,time_steps,1)
        
        #defining the network
        lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
        outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")
        
        #converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
        prediction=tf.matmul(outputs[-1],out_weights)+out_bias

        """# Store layers weight & bias
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

        # Construct model
        logits = multilayer_perceptron(X)"""

        # Inputs
        tf_test_dataset = tf.constant(test_dataset)

        # Define loss and optimizer
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=Y))
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        #beta = 0.01
        #loss_regularizer = tf.add_n([tf.nn.l2_loss(weights[v]) for v in weights]) * beta
        #loss = tf.reduce_mean(loss + loss_regularizer)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)
        
        #model evaluation
        correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        """ # Predictions for the training, validation, and test data.
        
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(multilayer_perceptron(tf_test_dataset))"""
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
                
                batch_data=batch_data.reshape((batch_size,time_steps,n_input))

                # Prepare the feed dict
                feed_dict = {X: batch_data,
                             Y: batch_labels}

                # run one step of computation
               # _, l, predictions = session.run([train_op, loss, correct_prediction],feed_dict=feed_dict)
                session.run(train_op, feed_dict=feed_dict)
                cost_history = np.append(cost_history, session.run(loss, feed_dict=feed_dict))
                if step % 10 == 0:
                    acc=session.run(accuracy,feed_dict=feed_dict)
                    los=session.run(loss,feed_dict=feed_dict)
                    print("Minibatch loss at step {0}: {1}".format(step, los))
                    print("Accuracy ",acc)

            #print("\nTest accuracy: {:.1f}%".format(accuracy(test_prediction.eval(), test_labels)))
            print("Testing Accuracy:", session.run(accuracy, feed_dict={X: test_dataset.reshape((-1, time_steps, n_input)), Y: test_labels}))
            plt.plot(range(len(cost_history)), cost_history)
            plt.axis([0, num_steps, 0, np.max(cost_history)])
            plt.show()
