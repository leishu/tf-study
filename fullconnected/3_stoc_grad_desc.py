# stochastic gradient descent training instead, which is much faster.
# 随机梯度下降，更快

import tensorflow as tf
from dataInit import DataInit
import numpy as np

data = DataInit()

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data.getDataSet()

batch_size = 128

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    # 输入数据
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, data.image_size * data.image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, data.num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights = tf.Variable(
        tf.truncated_normal([data.image_size * data.image_size, data.num_labels]))
    biases = tf.Variable(tf.zeros([data.num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        # 在训练数据中选择一个已被随机化的偏移量
        # 可以使用更好的随机化
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        # 小批量
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        # 输入是字典结构的数据
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))





# Initialized
# Minibatch loss at step 0: 16.854244
# Minibatch accuracy: 18.8%
# Validation accuracy: 17.9%
# Minibatch loss at step 500: 1.271446
# Minibatch accuracy: 78.9%
# Validation accuracy: 75.0%
# Minibatch loss at step 1000: 1.299051
# Minibatch accuracy: 72.7%
# Validation accuracy: 75.7%
# Minibatch loss at step 1500: 1.333678
# Minibatch accuracy: 78.9%
# Validation accuracy: 77.0%
# Minibatch loss at step 2000: 0.996773
# Minibatch accuracy: 81.2%
# Validation accuracy: 77.1%
# Minibatch loss at step 2500: 1.010918
# Minibatch accuracy: 79.7%
# Validation accuracy: 77.5%
# Minibatch loss at step 3000: 1.107841
# Minibatch accuracy: 79.7%
# Validation accuracy: 77.6%
# Test accuracy: 85.4%