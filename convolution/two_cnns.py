# Let's build a small network with two convolutional layers,
# followed by one fully connected layer.
# Convolutional networks are more expensive computationally,
# so we'll limit its depth and number of fully connected nodes.

# 生成小型网络：两层卷积，然后一个全连接层
# 卷积网络的计算更昂贵，所以，我们限制它的深度和全连接节点数量

import tensorflow as tf
from prepare import image_size, num_channels, num_labels
from prepare import train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
from prepare import accuracy


batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # 5, 5, 1, 16
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    # 5, 5, 16, 16
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    # 7*7*16=784, 64
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    # 16, 10
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases


    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))



num_steps = 1001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))



# Minibatch loss at step 0: 3.208451
# Minibatch accuracy: 25.0%
# Validation accuracy: 10.0%
# Minibatch loss at step 50: 2.083748
# Minibatch accuracy: 18.8%
# Validation accuracy: 35.2%
# Minibatch loss at step 100: 0.646165
# Minibatch accuracy: 87.5%
# Validation accuracy: 64.1%
# Minibatch loss at step 150: 0.610930
# Minibatch accuracy: 87.5%
# Validation accuracy: 74.2%
# Minibatch loss at step 200: 0.505966
# Minibatch accuracy: 81.2%
# Validation accuracy: 73.6%
# Minibatch loss at step 250: 0.994628
# Minibatch accuracy: 62.5%
# Validation accuracy: 76.9%
# Minibatch loss at step 300: 0.665928
# Minibatch accuracy: 81.2%
# Validation accuracy: 75.4%
# Minibatch loss at step 350: 0.457973
# Minibatch accuracy: 93.8%
# Validation accuracy: 79.1%
# Minibatch loss at step 400: 0.451236
# Minibatch accuracy: 87.5%
# Validation accuracy: 79.2%
# Minibatch loss at step 450: 1.006736
# Minibatch accuracy: 62.5%
# Validation accuracy: 79.5%
# Minibatch loss at step 500: 0.692601
# Minibatch accuracy: 75.0%
# Validation accuracy: 80.2%
# Minibatch loss at step 550: 0.379068
# Minibatch accuracy: 87.5%
# Validation accuracy: 80.8%
# Minibatch loss at step 600: 1.015194
# Minibatch accuracy: 68.8%
# Validation accuracy: 80.8%
# Minibatch loss at step 650: 0.764558
# Minibatch accuracy: 75.0%
# Validation accuracy: 78.9%
# Minibatch loss at step 700: 0.846055
# Minibatch accuracy: 68.8%
# Validation accuracy: 81.4%
# Minibatch loss at step 750: 0.182785
# Minibatch accuracy: 93.8%
# Validation accuracy: 81.9%
# Minibatch loss at step 800: 0.460340
# Minibatch accuracy: 87.5%
# Validation accuracy: 80.6%
# Minibatch loss at step 850: 0.728638
# Minibatch accuracy: 68.8%
# Validation accuracy: 81.8%
# Minibatch loss at step 900: 0.736923
# Minibatch accuracy: 68.8%
# Validation accuracy: 81.8%
# Minibatch loss at step 950: 0.574336
# Minibatch accuracy: 81.2%
# Validation accuracy: 82.4%
# Minibatch loss at step 1000: 0.374335
# Minibatch accuracy: 87.5%
# Validation accuracy: 81.7%
# Test accuracy: 88.9%