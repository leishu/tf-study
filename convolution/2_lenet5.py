# Try to get the best performance you can using a convolutional net.
# Look for example at the classic LeNet5 architecture,
# adding Dropout, and/or adding learning rate decay.


import tensorflow as tf
from prepare import image_size, num_channels, num_labels
from prepare import train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
from prepare import accuracy

batch_size = 128
patch_size = 5
max_pool_size1 = 2  # NxN window for 1st max pool layer
max_pool_size2 = 2  # NxN window for 2nd max pool layer
conv1_features = image_size + 1 - patch_size
conv2_features = conv1_features * 2
fully_connected_size1 = 120
target_size = 10




graph = tf.Graph()

keep_prob = 0.5

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # 5, 5, 1, 24
    conv1_weight = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, conv1_features], stddev=0.1, dtype=tf.float32))
    conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))

    # 5, 5, 24, 48
    conv2_weight = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, conv1_features, conv2_features], stddev=0.1, dtype=tf.float32))
    conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

    # fully connected variables
    resulting_width = image_size // (max_pool_size1 * max_pool_size2)
    resulting_height = image_size // (max_pool_size1 * max_pool_size2)

    # 2352
    full1_input_size = resulting_width * resulting_height * conv2_features


    full1_weight = tf.Variable(tf.truncated_normal(
        [full1_input_size, fully_connected_size1], stddev=0.1, dtype=tf.float32))
    full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))

    # 2352, 10
    full2_weight = tf.Variable(tf.truncated_normal(
        [fully_connected_size1, target_size], stddev=0.1, dtype=tf.float32))
    full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))


    # Model.
    def model(data):
        # First Conv-ReLU-MaxPool Layer
        conv1 = tf.nn.conv2d(data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
        max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1],
                                   strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')

        # Second Conv-ReLU-MaxPool Layer
        conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
        max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1],
                                   strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')

        # Transform Output into a 1xN layer for next fully connected layer
        final_conv_shape = max_pool2.get_shape().as_list()
        final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
        flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

        # First Fully Connected Layer
        fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))

        # Second Fully Connected Layer
        final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)

        return final_model_output


    # Training computation.
    logits = model(tf_train_dataset)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    # L2 regularization
    beta = 0.001
    loss += beta * (tf.nn.l2_loss(conv1_weight) + tf.nn.l2_loss(conv2_weight) +
                    tf.nn.l2_loss(full1_weight) + tf.nn.l2_loss(full2_weight))

    # Optimizer.
    global_step = tf.Variable(0)  # count the number of steps taken.
    starter_learning_rate = 0.5
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               10000, 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

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


# fully_connected_size1 = 100
# Minibatch loss at step 0: 3.683372
# Minibatch accuracy: 10.2%
# Validation accuracy: 10.6%
# Minibatch loss at step 50: 2.450181
# Minibatch accuracy: 57.8%
# Validation accuracy: 64.7%
# Minibatch loss at step 100: 1.474915
# Minibatch accuracy: 82.0%
# Validation accuracy: 82.5%
# Minibatch loss at step 150: 1.411926
# Minibatch accuracy: 86.7%
# Validation accuracy: 84.4%
# Minibatch loss at step 200: 1.345138
# Minibatch accuracy: 85.2%
# Validation accuracy: 84.7%
# Minibatch loss at step 250: 1.204821
# Minibatch accuracy: 91.4%
# Validation accuracy: 85.6%
# Minibatch loss at step 300: 1.307194
# Minibatch accuracy: 85.9%
# Validation accuracy: 85.4%
# Minibatch loss at step 350: 1.191712
# Minibatch accuracy: 89.8%
# Validation accuracy: 86.0%
# Minibatch loss at step 400: 1.518077
# Minibatch accuracy: 78.9%
# Validation accuracy: 85.8%
# Minibatch loss at step 450: 0.959132
# Minibatch accuracy: 93.8%
# Validation accuracy: 86.8%
# Minibatch loss at step 500: 1.014024
# Minibatch accuracy: 89.8%
# Validation accuracy: 86.7%
# Minibatch loss at step 550: 1.036407
# Minibatch accuracy: 86.7%
# Validation accuracy: 87.6%
# Minibatch loss at step 600: 1.148993
# Minibatch accuracy: 83.6%
# Validation accuracy: 87.3%
# Minibatch loss at step 650: 1.134534
# Minibatch accuracy: 83.6%
# Validation accuracy: 87.6%
# Minibatch loss at step 700: 0.893124
# Minibatch accuracy: 90.6%
# Validation accuracy: 87.7%
# Minibatch loss at step 750: 0.903596
# Minibatch accuracy: 87.5%
# Validation accuracy: 88.0%
# Minibatch loss at step 800: 0.969631
# Minibatch accuracy: 85.9%
# Validation accuracy: 88.3%
# Minibatch loss at step 850: 0.812270
# Minibatch accuracy: 88.3%
# Validation accuracy: 88.3%
# Minibatch loss at step 900: 0.907242
# Minibatch accuracy: 86.7%
# Validation accuracy: 88.1%
# Minibatch loss at step 950: 0.740381
# Minibatch accuracy: 91.4%
# Validation accuracy: 88.6%
# Minibatch loss at step 1000: 0.836618
# Minibatch accuracy: 87.5%
# Validation accuracy: 87.6%
# Test accuracy: 93.6%



# fully_connected_size1 = 80
# Minibatch loss at step 0: 3.383838
# Minibatch accuracy: 5.5%
# Validation accuracy: 12.2%
# Minibatch loss at step 50: 2.677838
# Minibatch accuracy: 39.8%
# Validation accuracy: 45.4%
# Minibatch loss at step 100: 1.859754
# Minibatch accuracy: 71.9%
# Validation accuracy: 65.4%
# Minibatch loss at step 150: 1.337420
# Minibatch accuracy: 83.6%
# Validation accuracy: 82.7%
# Minibatch loss at step 200: 1.299586
# Minibatch accuracy: 85.2%
# Validation accuracy: 83.6%
# Minibatch loss at step 250: 1.153738
# Minibatch accuracy: 89.1%
# Validation accuracy: 84.9%
# Minibatch loss at step 300: 1.232265
# Minibatch accuracy: 86.7%
# Validation accuracy: 84.0%
# Minibatch loss at step 350: 1.107319
# Minibatch accuracy: 86.7%
# Validation accuracy: 85.7%
# Minibatch loss at step 400: 1.261146
# Minibatch accuracy: 78.9%
# Validation accuracy: 85.8%
# Minibatch loss at step 450: 0.901378
# Minibatch accuracy: 91.4%
# Validation accuracy: 86.7%
# Minibatch loss at step 500: 0.909978
# Minibatch accuracy: 88.3%
# Validation accuracy: 86.5%
# Minibatch loss at step 550: 0.964818
# Minibatch accuracy: 87.5%
# Validation accuracy: 87.3%
# Minibatch loss at step 600: 1.065246
# Minibatch accuracy: 85.2%
# Validation accuracy: 87.5%
# Minibatch loss at step 650: 1.080757
# Minibatch accuracy: 83.6%
# Validation accuracy: 87.3%
# Minibatch loss at step 700: 0.840729
# Minibatch accuracy: 88.3%
# Validation accuracy: 88.0%
# Minibatch loss at step 750: 0.810744
# Minibatch accuracy: 89.1%
# Validation accuracy: 86.0%
# Minibatch loss at step 800: 0.942629
# Minibatch accuracy: 86.7%
# Validation accuracy: 88.0%
# Minibatch loss at step 850: 0.737187
# Minibatch accuracy: 89.1%
# Validation accuracy: 88.3%
# Minibatch loss at step 900: 0.895308
# Minibatch accuracy: 82.8%
# Validation accuracy: 88.2%
# Minibatch loss at step 950: 0.704891
# Minibatch accuracy: 89.8%
# Validation accuracy: 88.8%
# Minibatch loss at step 1000: 0.754470
# Minibatch accuracy: 85.9%
# Validation accuracy: 88.2%
# Test accuracy: 94.2%

# fully_connected_size1 = 120
# Minibatch loss at step 0: 4.162920
# Minibatch accuracy: 8.6%
# Validation accuracy: 10.0%
# Minibatch loss at step 50: 3.226353
# Minibatch accuracy: 38.3%
# Validation accuracy: 29.6%
# Minibatch loss at step 100: 2.611044
# Minibatch accuracy: 55.5%
# Validation accuracy: 58.1%
# Minibatch loss at step 150: 1.787150
# Minibatch accuracy: 78.1%
# Validation accuracy: 79.9%
# Minibatch loss at step 200: 1.636102
# Minibatch accuracy: 80.5%
# Validation accuracy: 82.5%
# Minibatch loss at step 250: 1.419644
# Minibatch accuracy: 89.8%
# Validation accuracy: 84.0%
# Minibatch loss at step 300: 1.557748
# Minibatch accuracy: 84.4%
# Validation accuracy: 83.4%
# Minibatch loss at step 350: 1.452234
# Minibatch accuracy: 82.8%
# Validation accuracy: 83.2%
# Minibatch loss at step 400: 1.524742
# Minibatch accuracy: 78.9%
# Validation accuracy: 85.3%
# Minibatch loss at step 450: 1.169768
# Minibatch accuracy: 93.0%
# Validation accuracy: 85.9%
# Minibatch loss at step 500: 1.211287
# Minibatch accuracy: 89.1%
# Validation accuracy: 86.1%
# Minibatch loss at step 550: 1.166994
# Minibatch accuracy: 89.1%
# Validation accuracy: 86.2%
# Minibatch loss at step 600: 1.356285
# Minibatch accuracy: 83.6%
# Validation accuracy: 86.7%
# Minibatch loss at step 650: 1.300179
# Minibatch accuracy: 82.8%
# Validation accuracy: 86.9%
# Minibatch loss at step 700: 1.037493
# Minibatch accuracy: 89.8%
# Validation accuracy: 87.4%
# Minibatch loss at step 750: 1.085287
# Minibatch accuracy: 87.5%
# Validation accuracy: 87.1%
# Minibatch loss at step 800: 1.176449
# Minibatch accuracy: 85.2%
# Validation accuracy: 86.2%
# Minibatch loss at step 850: 0.950196
# Minibatch accuracy: 86.7%
# Validation accuracy: 87.6%
# Minibatch loss at step 900: 1.057981
# Minibatch accuracy: 85.9%
# Validation accuracy: 88.2%
# Minibatch loss at step 950: 0.846578
# Minibatch accuracy: 92.2%
# Validation accuracy: 87.9%
# Minibatch loss at step 1000: 0.870837
# Minibatch accuracy: 89.8%
# Validation accuracy: 87.5%
# Test accuracy: 93.3%