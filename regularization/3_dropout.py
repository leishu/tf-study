
# Introduce Dropout on the hidden layer of the neural network.
# Remember: Dropout should only be introduced during training, not evaluation,
# otherwise your evaluation results would be stochastic as well.
# TensorFlow provides `nn.dropout()` for that, but you have to make sure it's only inserted during training.
# What happens to our extreme overfitting case?

# 在隐藏层引入Dropout
# 只能在训练时Dropout，不能在评估时Dropout。否则，评估结果也将是随机的


import tensorflow as tf
from dataInit import DataInit
import numpy as np

data = DataInit()



batch_size = 128
hidden1_units = 1024


keep_prob = 0.5

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data.getDataSet()

train_dataset = train_dataset[:5000]
train_labels = train_labels[:5000]


def feedforward(dataset):
    h1 = tf.nn.relu(tf.matmul(dataset, weights1) + biases1)
    return tf.matmul(h1, weights2) + biases2


def feedforwarddropout(dataset):
    h1 = tf.nn.relu(tf.matmul(dataset, weights1) + biases1)
    d1 = tf.nn.dropout(h1, keep_prob)
    return tf.matmul(d1, weights2) + biases2

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
        tf.truncated_normal([hidden1_units, data.num_labels]))
    biases2 = tf.Variable(tf.zeros([data.num_labels]))

    # Training computation.
    logits = feedforwarddropout(tf_train_dataset)


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


# keep_prob = 0.2
# Minibatch loss at step 0: 1081.487549
# Minibatch accuracy: 11.7%
# Validation accuracy: 28.2%
# Minibatch loss at step 500: 320.488586
# Minibatch accuracy: 62.5%
# Validation accuracy: 78.9%
# Minibatch loss at step 1000: 174.102264
# Minibatch accuracy: 68.8%
# Validation accuracy: 77.6%
# Minibatch loss at step 1500: 104.005257
# Minibatch accuracy: 67.2%
# Validation accuracy: 73.2%
# Minibatch loss at step 2000: 69.788475
# Minibatch accuracy: 69.5%
# Validation accuracy: 75.3%
# Minibatch loss at step 2500: 46.164005
# Minibatch accuracy: 76.6%
# Validation accuracy: 77.9%
# Minibatch loss at step 3000: 24.066160
# Minibatch accuracy: 82.8%
# Validation accuracy: 79.6%
# Test accuracy: 87.0%



# keep_prob = 0.5
# Minibatch loss at step 0: 850.412903
# Minibatch accuracy: 3.9%
# Validation accuracy: 26.5%
# Minibatch loss at step 500: 195.286026
# Minibatch accuracy: 90.6%
# Validation accuracy: 81.2%
# Minibatch loss at step 1000: 117.972626
# Minibatch accuracy: 97.7%
# Validation accuracy: 81.1%
# Minibatch loss at step 1500: 71.762550
# Minibatch accuracy: 95.3%
# Validation accuracy: 82.2%
# Minibatch loss at step 2000: 43.424957
# Minibatch accuracy: 95.3%
# Validation accuracy: 82.0%
# Minibatch loss at step 2500: 26.444960
# Minibatch accuracy: 98.4%
# Validation accuracy: 82.3%
# Minibatch loss at step 3000: 15.995116
# Minibatch accuracy: 98.4%
# Validation accuracy: 82.6%
# Test accuracy: 90.2%


# keep_prob = 0.8
# Minibatch loss at step 0: 689.250549
# Minibatch accuracy: 9.4%
# Validation accuracy: 32.5%
# Minibatch loss at step 500: 192.104355
# Minibatch accuracy: 94.5%
# Validation accuracy: 79.9%
# Minibatch loss at step 1000: 115.776680
# Minibatch accuracy: 100.0%
# Validation accuracy: 81.1%
# Minibatch loss at step 1500: 70.250534
# Minibatch accuracy: 100.0%
# Validation accuracy: 81.1%
# Minibatch loss at step 2000: 42.621563
# Minibatch accuracy: 99.2%
# Validation accuracy: 81.6%
# Minibatch loss at step 2500: 26.081749
# Minibatch accuracy: 99.2%
# Validation accuracy: 81.6%
# Minibatch loss at step 3000: 15.678505
# Minibatch accuracy: 100.0%
# Validation accuracy: 82.1%
# Test accuracy: 89.2%