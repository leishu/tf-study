# Implementing Gates
# ----------------------------------
#
# 一个门由一个variable和一个placeholder组成
# 我们将要求TensorFlow根据我们的损失函数改变variable

import tensorflow as tf
from tensorflow.python.framework import ops


# 学习速率
learning_rate = 0.01

graph = tf.Graph()

with graph.as_default():
    # ----------------------------------
    #   f(x) = a * x
    #
    #  a --
    #      |
    #      |---- (multiply) --> output
    #  x --|
    #

    a = tf.Variable(tf.constant(4.))
    x_val = 5.
    x_data = tf.placeholder(dtype=tf.float32)

    multiplication = tf.multiply(a, x_data)

    # 损失函数：输出和50的差的平方
    loss = tf.square(tf.subtract(multiplication, 50.))

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, name="optimizer")

with tf.Session(graph=graph) as session:

    tf.global_variables_initializer().run()
    print("Initialized")

    print('Optimizing a Multiplication Gate Output to 50.')
    for i in range(10):
        _, a_val, multi = session.run([optimizer, a, multiplication], feed_dict={x_data: x_val})

        print('%s * %s = %s' % (a_val, x_val, multi))


with graph.as_default():
    #----------------------------------
    # Create a nested gate:
    #   f(x) = a * x + b
    #
    #  a --
    #      |
    #      |-- (multiply)--
    #  x --|              |
    #                     |-- (add) --> output
    #                 b --|
    #
    #
    a = tf.Variable(tf.constant(1.))
    b = tf.Variable(tf.constant(1.))
    x_val = 5.
    x_data = tf.placeholder(dtype=tf.float32)

    two_gate = tf.add(tf.multiply(a, x_data), b)

    # 损失函数：输出和50的差的平方
    loss = tf.square(tf.subtract(two_gate, 50.))

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, name="optimizer")

with tf.Session(graph=graph) as session:

    tf.global_variables_initializer().run()
    print("Initialized")

    print('\nOptimizing Two Gate Output to 50.')
    for i in range(10):
        _, a_val, b_val, two = session.run([optimizer, a, b, two_gate], feed_dict={x_data: x_val})

        print('%s * %s + %s = %s' % (a_val, x_val, b_val, two))