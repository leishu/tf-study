# 线性回归：TensorFlow方法
# ----------------------------------
#
# 怎样使用TensorFlow解决线性回归问题。
# y = Ax + b
#
# 我们将使用iris数据：
#  y = 萼片长
#  x = 花瓣宽

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

graph = tf.Graph()

# 加载数据
# iris.data = [(萼片 Length, 萼片 Width, 花瓣 Length, 花瓣 Width)]
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# 批大小
batch_size = 25
# 学习速率
learning_rate = 0.02
# 训练次数
iterations = 100
# 损失值
loss_vec = []

with graph.as_default():
    # 初始化placeholders
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # 变量
    A = tf.Variable(tf.random_normal(shape=[1, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    # 运算模型
    model_output = tf.add(tf.matmul(x_data, A), b)

    # 损失函数
    loss_l2 = tf.reduce_mean(tf.square(y_target - model_output))

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_l2)


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    # 循环
    for i in range(iterations):
        rand_index = np.random.choice(len(x_vals), size=batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        _, loss, slope, y_intercept = session.run([optimizer, loss_l2, A, b],
                                                  feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(loss)
        if (i + 1) % 10 == 0:
            print('Step #%s slope = %s y_intercept = %s' % (i + 1, slope, y_intercept))
            print('Loss = ' + str(loss))

# 拟合线
best_fit = []
for i in x_vals:
    best_fit.append(slope[0] * i + y_intercept[0])

# 画图
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

# 损失图
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.show()
