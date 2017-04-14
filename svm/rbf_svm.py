# 线性核
# K(x1, x2) = t(x1) * x2
#
# 高斯核 (RBF):
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)

# x = (x1, x2, x3, x4)
# y = (y1, y2, y3, y4)
# f(x) = (x1x1, x1x2, x1x3, x1x4, x2x1, x2x2, x2x3, x2x4, x3x1, x3x2, x3x3, x3x4, x4x1, x4x2, x4x3, x4x4)
# f(y)亦然；
# 核函数 K(x, y) = (x, y)^2.
# 比如：
# x = (1, 2, 3, 4)
# y = (5, 6, 7, 8). 那么：
# f(x) = (1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8,12, 16)
# f(y) = (25, 30, 35, 40, 30, 36, 42, 48, 35, 42, 49, 56, 40, 48, 56, 64)
# f(x), f(y) = 25+60+105+160+60+144+252+384+105+252+441+672+160+384+672+1024 = 4900
# K(x, y) = (5+12+21+32)^2 = 70^2 = 4900.


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

graph = tf.Graph()

# 批大小
batch_size = 350
# 生成非线性数据
(x_vals, y_vals) = datasets.make_circles(n_samples=batch_size, factor=.5, noise=.1)
y_vals = np.array([1 if y == 1 else -1 for y in y_vals])


# 学习速率
learning_rate = 0.002
# 训练次数
iterations = 1000
loss_vec = []
batch_accuracy = []

with graph.as_default():
    # 初始化placeholders
    x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    # 变量
    A = tf.Variable(tf.random_normal(shape=[1, batch_size]))

    gamma = tf.constant(-50.0)


    # RBF核
    def rbf(x, y):
        # 行平方和
        rA = tf.reshape(tf.reduce_sum(tf.square(x), 1), [-1, 1])
        rB = tf.reshape(tf.reduce_sum(tf.square(y), 1), [-1, 1])
        # rA - 2 × x * y + rB
        sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x, tf.transpose(y)))),
                         tf.transpose(rB))
        return tf.exp(tf.multiply(gamma, tf.abs(sq_dist)))


    my_kernel = rbf(x_data, x_data)
    pred_kernel = rbf(x_data, prediction_grid)

    # 计算SVM模型
    model_output = tf.matmul(A, my_kernel)

    first_term = tf.reduce_sum(A)
    b_vec_cross = tf.matmul(tf.transpose(A), A)
    y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
    second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
    loss = tf.negative(tf.subtract(first_term, second_term))

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), A), pred_kernel)
    prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    # 循环
    for i in range(iterations):
        rand_index = np.random.choice(len(x_vals), size=batch_size)
        rand_x = x_vals[rand_index]
        rand_y = np.transpose([y_vals[rand_index]])

        _, l, acc = session.run([optimizer, loss, accuracy],
                                feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})

        loss_vec.append(l)
        batch_accuracy.append(acc)

        if (i + 1) % 100 == 0:
            print('Step #' + str(i + 1))
            print('Loss = ' + str(l))

    # 增加画点的网格
    x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
    y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    [grid_predictions] = session.run(prediction, feed_dict={x_data: rand_x,
                                                            y_target: rand_y,
                                                            prediction_grid: grid_points})
    grid_predictions = grid_predictions.reshape(xx.shape)


class1_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class1_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class2_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == -1]
class2_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == -1]
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='Class 1')
plt.plot(class2_x, class2_y, 'kx', label='Class -1')
plt.title('Gaussian SVM Results')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.ylim([-1.5, 1.5])
plt.xlim([-1.5, 1.5])
plt.show()

plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
