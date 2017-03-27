

import tensorflow as tf
from dataInit import DataInit
import numpy as np

data = DataInit()

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data.getDataSet()

batch_size = 128
hidden1_units = 2048


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def feedforward(dataset):
    h1 = tf.nn.relu(tf.matmul(dataset, weights1) + biases1)
    return tf.matmul(h1, weights2) + biases2



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
    weights1 = tf.Variable(
        tf.truncated_normal([data.image_size * data.image_size, hidden1_units]))
    biases1 = tf.Variable(tf.zeros([hidden1_units]))



    weights2 = tf.Variable(
        tf.truncated_normal([hidden1_units, data.num_labels]))
    biases2 = tf.Variable(tf.zeros([data.num_labels]))

    # Training computation.
    logits = feedforward(tf_train_dataset)


    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(feedforward(tf_valid_dataset))
    test_prediction = tf.nn.softmax(feedforward(tf_test_dataset))


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












# hidden1_units = 512
# Minibatch loss at step 0: 220.128464
# Minibatch accuracy: 7.8%
# Validation accuracy: 29.3%
# Minibatch loss at step 500: 4.893038
# Minibatch accuracy: 79.7%
# Validation accuracy: 77.8%
# Minibatch loss at step 1000: 6.220913
# Minibatch accuracy: 79.7%
# Validation accuracy: 78.3%
# Minibatch loss at step 1500: 3.095931
# Minibatch accuracy: 78.1%
# Validation accuracy: 79.0%
# Minibatch loss at step 2000: 2.105818
# Minibatch accuracy: 83.6%
# Validation accuracy: 78.8%
# Minibatch loss at step 2500: 3.824628
# Minibatch accuracy: 85.9%
# Validation accuracy: 78.5%
# Minibatch loss at step 3000: 2.640835
# Minibatch accuracy: 82.0%
# Validation accuracy: 76.7%
# Test accuracy: 83.9%


# hidden1_units = 1024
# Minibatch loss at step 0: 412.173889
# Minibatch accuracy: 5.5%
# Validation accuracy: 35.0%
# Minibatch loss at step 500: 13.804764
# Minibatch accuracy: 79.7%
# Validation accuracy: 79.3%
# Minibatch loss at step 1000: 12.009517
# Minibatch accuracy: 79.7%
# Validation accuracy: 80.2%
# Minibatch loss at step 1500: 6.690843
# Minibatch accuracy: 82.8%
# Validation accuracy: 81.5%
# Minibatch loss at step 2000: 5.911739
# Minibatch accuracy: 83.6%
# Validation accuracy: 81.4%
# Minibatch loss at step 2500: 7.510405
# Minibatch accuracy: 87.5%
# Validation accuracy: 81.0%
# Minibatch loss at step 3000: 3.826010
# Minibatch accuracy: 82.0%
# Validation accuracy: 79.7%
# Test accuracy: 86.5%


# hidden1_units = 2048
# Minibatch loss at step 0: 509.393555
# Minibatch accuracy: 10.2%
# Validation accuracy: 29.4%
# Minibatch loss at step 500: 18.893696
# Minibatch accuracy: 85.2%
# Validation accuracy: 81.0%
# Minibatch loss at step 1000: 22.493233
# Minibatch accuracy: 82.8%
# Validation accuracy: 82.0%
# Minibatch loss at step 1500: 7.828497
# Minibatch accuracy: 82.0%
# Validation accuracy: 82.1%
# Minibatch loss at step 2000: 9.114884
# Minibatch accuracy: 84.4%
# Validation accuracy: 83.1%
# Minibatch loss at step 2500: 14.965179
# Minibatch accuracy: 85.9%
# Validation accuracy: 82.0%
# Minibatch loss at step 3000: 6.086237
# Minibatch accuracy: 82.8%
# Validation accuracy: 83.6%
# Test accuracy: 90.2%