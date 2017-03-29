# The convolutional model above uses convolutions with stride 2 to reduce the dimensionality.
# Replace the strides by a max pooling operation (nn.max_pool()) of stride 2 and kernel size 2.
# 第一个卷积层使用stride=2来减少维度
# 通过最大池化，使用stride=2和kernel=2替换strides


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

        tf.nn.max_pool(hidden, padding='SAME', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

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



# Minibatch loss at step 0: 2.914891
# Minibatch accuracy: 0.0%
# Validation accuracy: 9.5%
# Minibatch loss at step 50: 1.739102
# Minibatch accuracy: 43.8%
# Validation accuracy: 39.2%
# Minibatch loss at step 100: 0.453460
# Minibatch accuracy: 93.8%
# Validation accuracy: 71.7%
# Minibatch loss at step 150: 0.541777
# Minibatch accuracy: 87.5%
# Validation accuracy: 76.3%
# Minibatch loss at step 200: 0.571164
# Minibatch accuracy: 81.2%
# Validation accuracy: 74.3%
# Minibatch loss at step 250: 1.081659
# Minibatch accuracy: 62.5%
# Validation accuracy: 77.2%
# Minibatch loss at step 300: 0.851963
# Minibatch accuracy: 81.2%
# Validation accuracy: 77.2%
# Minibatch loss at step 350: 0.347110
# Minibatch accuracy: 87.5%
# Validation accuracy: 79.4%
# Minibatch loss at step 400: 0.379048
# Minibatch accuracy: 93.8%
# Validation accuracy: 80.0%
# Minibatch loss at step 450: 1.120907
# Minibatch accuracy: 62.5%
# Validation accuracy: 80.9%
# Minibatch loss at step 500: 0.642104
# Minibatch accuracy: 81.2%
# Validation accuracy: 80.2%
# Minibatch loss at step 550: 0.430961
# Minibatch accuracy: 81.2%
# Validation accuracy: 80.4%
# Minibatch loss at step 600: 1.064621
# Minibatch accuracy: 68.8%
# Validation accuracy: 81.1%
# Minibatch loss at step 650: 0.802496
# Minibatch accuracy: 75.0%
# Validation accuracy: 80.1%
# Minibatch loss at step 700: 0.889794
# Minibatch accuracy: 68.8%
# Validation accuracy: 81.4%
# Minibatch loss at step 750: 0.126830
# Minibatch accuracy: 100.0%
# Validation accuracy: 82.4%
# Minibatch loss at step 800: 0.520708
# Minibatch accuracy: 87.5%
# Validation accuracy: 81.0%
# Minibatch loss at step 850: 0.734428
# Minibatch accuracy: 62.5%
# Validation accuracy: 82.5%
# Minibatch loss at step 900: 0.751345
# Minibatch accuracy: 68.8%
# Validation accuracy: 82.1%
# Minibatch loss at step 950: 0.748958
# Minibatch accuracy: 75.0%
# Validation accuracy: 82.5%
# Minibatch loss at step 1000: 0.461151
# Minibatch accuracy: 87.5%
# Validation accuracy: 82.4%
# Test accuracy: 89.0%