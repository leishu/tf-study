import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

graph = tf.Graph()

# 我们将增加下列样本数据：
# x-data: 100个随机样本，服从正态分布 ~ N(1, 0.1)
# target: 100个等于10的值
# 我们将符合该模型：x-data * A = target
# 理论上，A = 10

# 增加数据
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)

with graph.as_default():
    x_data = tf.placeholder(shape=[1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[1], dtype=tf.float32)

    # 增加变量 （参数A）
    A = tf.Variable(tf.random_normal(shape=[1]))

    # 图的操作
    my_output = tf.multiply(x_data, A)

    # 增加L2（squared hinge loss）损失函数
    loss = tf.square(my_output - y_target)

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(loss)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    # 循环
    for i in range(100):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        _, l, pred = session.run([optimizer, loss, A], feed_dict={x_data: rand_x, y_target: rand_y})
        if (i + 1) % 10 == 0:
            print('Step #%s: A = %s' % (i + 1, pred))
            print('Loss = %s ' % (l))
