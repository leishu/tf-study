import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

graph = tf.Graph()

# 我们将增加下列样本数据：
# x-data: 50个随机样本，服从正态分布 ~ N(-1, 1).50个随机样本，服从正态分布 ~ N(1, 1)
# target: 50个等于0的值,50个等于1的值
# 我们将符合二元分类模型：sigmoid(x+A) < 0.5 -> 0 else 1
# 理论上，A应该是 -(mean1 + mean2)/2

x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))


with graph.as_default():
    x_data = tf.placeholder(shape=[1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[1], dtype=tf.float32)

    # 增加变量 （参数A）
    A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

    # 图的操作
    # 想要增加sigmoid(x + A)计算，sigmoid部分在损失函数内
    my_output = tf.add(x_data, A)

    # 增加分类损失函数（交叉熵）
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target)

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(loss)


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    # 循环
    for i in range(1400):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        _, l, pred = session.run([optimizer, loss, A], feed_dict={x_data: rand_x, y_target: rand_y})
        if (i + 1) % 50 == 0:
            print('Step #%s: A = %s' % (i + 1, pred))
            print('Loss = %s ' % (l))

    # 预测
    predictions = []
    for i in range(len(x_vals)):
        x_val = [x_vals[i]]
        prediction = session.run(tf.round(tf.sigmoid(my_output)), feed_dict={x_data: x_val})
        predictions.append(prediction[0])

    accuracy = sum(x == y for x, y in zip(predictions, y_vals)) / 100.
    print('Ending Accuracy = %s ' % (np.round(accuracy, 2)))