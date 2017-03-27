

import tensorflow as tf
from dataInit import DataInit
import numpy as np

data = DataInit()

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data.getDataSet()

batch_size = 128
hidden1_units = 512


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
    # L2 regularization
    beta = 0.001
    loss += beta * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2))

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
# Minibatch loss at step 0: 405.541412
# Minibatch accuracy: 17.2 %
# Validation accuracy: 28.0 %
# Minibatch loss at step 500: 96.010368
# Minibatch accuracy: 82.8 %
# Validation accuracy: 77.7 %
# Minibatch loss at step 1000: 57.378365
# Minibatch accuracy: 79.7 % Validation accuracy: 80.9 %
# Minibatch loss at step 1500: 34.350525
# Minibatch accuracy: 82.0 %
# Validation accuracy: 81.9 % Minibatch loss at step 2000: 20.934284
# Minibatch accuracy: 86.7 %
# Validation accuracy: 84.0 %
# Minibatch loss at step 2500: 12.963234
# Minibatch accuracy: 86.7 %
# Validation accuracy: 85.2 %
# Minibatch loss at step 3000: 7.943158
# Minibatch accuracy: 89.8 % Validation accuracy: 85.9 %
# Test accuracy: 92.1 %


# hidden1_units = 1024
# Minibatch loss at step 0: 677.916504
# Minibatch accuracy: 11.7 %
# Validation accuracy: 28.2 %
# Minibatch loss at step 500: 191.958740
# Minibatch accuracy: 80.5 %
# Validation accuracy: 80.6 %
# Minibatch loss at step 1000: 116.306618
# Minibatch accuracy: 80.5 %
# Validation accuracy: 80.7 %
# Minibatch loss at step 1500: 68.702820
# Minibatch accuracy: 83.6 %
# Validation accuracy: 83.2 %
# Minibatch loss at step 2000: 41.535564
# Minibatch accuracy: 85.9 %
# Validation accuracy: 84.4 %
# Minibatch loss at step 2500: 25.357819
# Minibatch accuracy: 86.7 %
# Validation accuracy: 85.4 %
# Minibatch loss at step 3000: 15.447125
# Minibatch accuracy: 89.1 %
# Validation accuracy: 86.6 %
# Test accuracy: 92.5 %


# hidden1_units = 2048
# Minibatch loss at step 0: 1105.501465
# Minibatch accuracy: 5.5%
# Validation accuracy: 32.4%
# Minibatch loss at step 500: 386.388000
# Minibatch accuracy: 85.2%
# Validation accuracy: 80.5%
# Minibatch loss at step 1000: 234.160522
# Minibatch accuracy: 82.8%
# Validation accuracy: 82.0%
# Minibatch loss at step 1500: 137.453400
# Minibatch accuracy: 84.4%
# Validation accuracy: 83.6%
# Minibatch loss at step 2000: 83.199440
# Minibatch accuracy: 87.5%
# Validation accuracy: 84.8%
# Minibatch loss at step 2500: 50.457291
# Minibatch accuracy: 87.5%
# Validation accuracy: 85.2%
# Minibatch loss at step 3000: 30.491129
# Minibatch accuracy: 89.8%
# Validation accuracy: 86.8%
# Test accuracy: 93.1%