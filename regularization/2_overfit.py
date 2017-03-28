import tensorflow as tf
from dataInit import DataInit
import numpy as np

data = DataInit()

# Let's demonstrate an extreme case of overfitting.
# Restrict your training data to just a few batches. What happens?




batch_size = 128
hidden1_units = 1024


train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data.getDataSet()

train_dataset = train_dataset[:5000]
train_labels = train_labels[:5000]


def feedforward(dataset):
    h1 = tf.nn.relu(tf.matmul(dataset, weights1) + biases1)
    return tf.matmul(h1, weights2) + biases2



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



# Minibatch loss at step 0: 760.376709
# Minibatch accuracy: 10.9%
# Validation accuracy: 33.6%
# Minibatch loss at step 500: 190.085205
# Minibatch accuracy: 98.4%
# Validation accuracy: 79.7%
# Minibatch loss at step 1000: 115.315346
# Minibatch accuracy: 99.2%
# Validation accuracy: 79.3%
# Minibatch loss at step 1500: 69.836143
# Minibatch accuracy: 100.0%
# Validation accuracy: 79.7%
# Minibatch loss at step 2000: 42.344410
# Minibatch accuracy: 100.0%
# Validation accuracy: 80.1%
# Minibatch loss at step 2500: 25.779739
# Minibatch accuracy: 99.2%
# Validation accuracy: 79.2%
# Minibatch loss at step 3000: 15.591178
# Minibatch accuracy: 100.0%
# Validation accuracy: 81.0%
# Test accuracy: 88.4%