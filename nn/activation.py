# 带有激活函数的门

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

batch_size = 50
# 学习速率
learning_rate = 0.01

graph = tf.Graph()

with graph.as_default():
    tf.set_random_seed(5)

    a1 = tf.Variable(tf.random_normal(shape=[1, 1]))
    b1 = tf.Variable(tf.random_uniform(shape=[1, 1]))
    a2 = tf.Variable(tf.random_normal(shape=[1, 1]))
    b2 = tf.Variable(tf.random_uniform(shape=[1, 1]))

    x = np.random.normal(2, 0.1, 500)
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1))
    relu_activation = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

    loss1 = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75)))
    loss2 = tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))

    # 优化器
    optimizer1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss1)
    optimizer2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss2)

print('\nOptimizing Sigmoid AND Relu Output to 0.75')
loss_vec_sigmoid = []
loss_vec_relu = []
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    for i in range(500):
        rand_indices = np.random.choice(len(x), size=batch_size)
        x_vals = np.transpose([x[rand_indices]])

        _, l1, sig = session.run([optimizer1, loss1, sigmoid_activation], feed_dict={x_data: x_vals})
        _, l2, relu = session.run([optimizer2, loss2, relu_activation], feed_dict={x_data: x_vals})

        loss_vec_sigmoid.append(l1)
        loss_vec_relu.append(l2)

        if i % 50 == 0:
            print('sigmoid = %s relu = %s' % (np.mean(sig), np.mean(relu)))

# Plot the loss
plt.plot(loss_vec_sigmoid, 'k-', label='Sigmoid Activation')
plt.plot(loss_vec_relu, 'r--', label='Relu Activation')
plt.ylim([0, 1.0])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
