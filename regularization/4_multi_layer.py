# Try to get the best performance you can using a multi-layer model!
# The best reported test accuracy using a deep network is 97.1%.
#
# One avenue you can explore is to add multiple layers.
#
# Another one is to use learning rate decay:
#
# global_step = tf.Variable(0)  # count the number of steps taken.
# learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


# 使用多层网络，能获得更好的性能，使用深层网络的最佳报告测试精度为97.1％。


import tensorflow as tf
from dataInit import DataInit
import numpy as np

data = DataInit()



batch_size = 128
hidden1_units = 1024
hidden2_units = 512

num_steps = 30001

keep_prob = 0.5

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data.getDataSet()


def feedforward(dataset):
    h1 = tf.nn.relu(tf.matmul(dataset, weights1) + biases1)
    h2 = tf.nn.relu(tf.matmul(h1, weights2) + biases2)
    return tf.matmul(h2, weights3) + biases3



def feedforwarddropout(dataset):
    h1 = tf.nn.relu(tf.matmul(dataset, weights1) + biases1)
    #d1 = tf.nn.dropout(h1, keep_prob)
    h2 = tf.nn.relu(tf.matmul(h1, weights2) + biases2)
    d2 = tf.nn.dropout(h2, keep_prob)
    return tf.matmul(d2, weights3) + biases3


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])




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
        tf.truncated_normal([hidden1_units, hidden2_units]))
    biases2 = tf.Variable(tf.zeros([hidden2_units]))


    weights3 = tf.Variable(
        tf.truncated_normal([hidden2_units, data.num_labels]))
    biases3 = tf.Variable(tf.zeros([data.num_labels]))

    # Training computation.
    logits = feedforwarddropout(tf_train_dataset)
    logits = tf.nn.softmax(logits)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    # L2 regularization
    beta = 0.001
    loss += beta * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3))


    # Optimizer.
    # optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    global_step = tf.Variable(0)  # count the number of steps taken.
    starter_learning_rate = 0.5
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               10000, 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = logits
    valid_prediction = tf.nn.softmax(feedforward(tf_valid_dataset))
    test_prediction = tf.nn.softmax(feedforward(tf_test_dataset))




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
        if (step % 2000 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# learning rate decay:
# Minibatch loss at step 0: 517.622375
# Minibatch accuracy: 9.4%
# Validation accuracy: 12.0%
# Minibatch loss at step 2000: 72.497810
# Minibatch accuracy: 38.3%
# Validation accuracy: 37.7%
# Minibatch loss at step 4000: 11.578331
# Minibatch accuracy: 68.0%
# Validation accuracy: 65.4%
# Minibatch loss at step 6000: 3.132052
# Minibatch accuracy: 73.4%
# Validation accuracy: 76.6%
# Minibatch loss at step 8000: 1.873861
# Minibatch accuracy: 82.8%
# Validation accuracy: 77.5%
# Minibatch loss at step 10000: 1.675801
# Minibatch accuracy: 87.5%
# Validation accuracy: 85.7%
# Minibatch loss at step 12000: 1.654357
# Minibatch accuracy: 86.7%
# Validation accuracy: 86.4%
# Minibatch loss at step 14000: 1.636801
# Minibatch accuracy: 88.3%
# Validation accuracy: 86.5%
# Minibatch loss at step 16000: 1.665716
# Minibatch accuracy: 85.2%
# Validation accuracy: 86.8%
# Minibatch loss at step 18000: 1.657981
# Minibatch accuracy: 86.7%
# Validation accuracy: 86.8%
# Minibatch loss at step 20000: 1.654783
# Minibatch accuracy: 87.5%
# Validation accuracy: 86.6%
# Minibatch loss at step 22000: 1.625729
# Minibatch accuracy: 89.8%
# Validation accuracy: 86.9%
# Minibatch loss at step 24000: 1.631351
# Minibatch accuracy: 91.4%
# Validation accuracy: 87.3%
# Minibatch loss at step 26000: 1.619883
# Minibatch accuracy: 91.4%
# Validation accuracy: 87.1%
# Minibatch loss at step 28000: 1.659107
# Minibatch accuracy: 85.2%
# Validation accuracy: 87.0%
# Minibatch loss at step 30000: 1.613097
# Minibatch accuracy: 92.2%
# Validation accuracy: 87.4%
# Test accuracy: 93.3%



#
# Minibatch loss at step 0: 516.611938
# Minibatch accuracy: 6.2%
# Validation accuracy: 7.3%
# Minibatch loss at step 2000: 72.557068
# Minibatch accuracy: 20.3%
# Validation accuracy: 25.8%
# Minibatch loss at step 4000: 11.819049
# Minibatch accuracy: 45.3%
# Validation accuracy: 42.5%
# Minibatch loss at step 6000: 3.142283
# Minibatch accuracy: 75.0%
# Validation accuracy: 75.3%
# Minibatch loss at step 8000: 1.894289
# Minibatch accuracy: 81.2%
# Validation accuracy: 76.8%
# Minibatch loss at step 10000: 1.771114
# Minibatch accuracy: 76.6%
# Validation accuracy: 76.9%
# Minibatch loss at step 12000: 1.671385
# Minibatch accuracy: 85.2%
# Validation accuracy: 85.4%
# Minibatch loss at step 14000: 1.662133
# Minibatch accuracy: 85.9%
# Validation accuracy: 85.9%
# Minibatch loss at step 16000: 1.689631
# Minibatch accuracy: 85.2%
# Validation accuracy: 85.5%
# Minibatch loss at step 18000: 1.668524
# Minibatch accuracy: 85.2%
# Validation accuracy: 85.8%
# Minibatch loss at step 20000: 1.658547
# Minibatch accuracy: 84.4%
# Validation accuracy: 85.8%
# Minibatch loss at step 22000: 1.643137
# Minibatch accuracy: 87.5%
# Validation accuracy: 85.9%
# Minibatch loss at step 24000: 1.642132
# Minibatch accuracy: 89.1%
# Validation accuracy: 86.2%
# Minibatch loss at step 26000: 1.635869
# Minibatch accuracy: 89.8%
# Validation accuracy: 86.1%
# Minibatch loss at step 28000: 1.667284
# Minibatch accuracy: 84.4%
# Validation accuracy: 86.0%
# Minibatch loss at step 30000: 1.625820
# Minibatch accuracy: 88.3%
# Validation accuracy: 86.2%
# Test accuracy: 92.8%