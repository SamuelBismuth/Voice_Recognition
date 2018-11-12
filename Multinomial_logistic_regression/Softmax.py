#https://www.geeksforgeeks.org/softmax-regression-using-tensorflow/
#https://en.wikipedia.org/wiki/Multinomial_logistic_regression

#https://www.altoros.com/blog/using-logistic-and-softmax-regression-with-tensorflow/
#http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/

# usa = [1, 0, 0, 0, 0]
# uk = [0, 1, 0, 0, 0]
# ussr = [0, 0, 1, 0, 0]
# france = [0, 0, 0, 1, 0]
# israel = [0, 0, 0, 0, 1]

import tensorflow as tf
import numpy as np

categories = 5
features = 1


def softmax(data):
    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, categories])
    W = tf.Variable(tf.zeros([features, categories]))
    b = tf.Variable(tf.zeros([categories]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    loss = -tf.reduce_mean(y_*tf.log(y))
    update = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
    # Type: <class 'numpy.ndarray'>
    data_x = [data[i].mfcc for i in range(len(data))]
    data_y = np.array([data[i].accent for i in range(len(data))])
    print(data_x)
    print(data_y)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(0, 10000):
        sess.run(update, feed_dict={x: data_x, y_: data_y})
    print('Prediction for: 60"' + ': "', sess.run(y, feed_dict={x: [data[60].mfcc]}))