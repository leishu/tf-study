# 多类非线性SVM
# ----------------------------------
#
# 高斯核，多类iris数据集(Sepal: 萼片, Petal: 花瓣)
#
# 高斯核:
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
#
# X : (Sepal Length, Petal Width)
# Y: (I. setosa, I. virginica, I. versicolor) (3 classes)
#
# 基本思路：引入额外的维度
# 根据最大margin或者是到边界的距离做分类

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
import os
from tensorflow.python.framework import ops

# 加载数据(Sepal: 萼片, Petal: 花瓣)
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals1 = np.array([1 if y == 0 else -1 for y in iris.target])
y_vals2 = np.array([1 if y == 1 else -1 for y in iris.target])
y_vals3 = np.array([1 if y == 2 else -1 for y in iris.target])
y_vals = np.array([y_vals1, y_vals2, y_vals3])
class1_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 0]
class1_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 0]
class2_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 1]
class2_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 1]
class3_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 2]
class3_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 2]

graph = tf.Graph()

# 批大小
batch_size = 50
# 学习速率
learning_rate = 0.01
# 训练次数
iterations = 300
loss_vec = []
batch_accuracy = []

log_path = "tensorboard/"

with graph.as_default():
    # 初始化placeholders
    x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="X")
    y_target = tf.placeholder(shape=[3, None], dtype=tf.float32, name="Y")
    prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="pred_grid")

    # 增加SVM变量
    A = tf.Variable(tf.random_normal(shape=[3, batch_size]), name="A")

    gamma = tf.constant(-10.0)

    # reshape/batch multiplication
    def reshape_matmul(mat):
        v1 = tf.expand_dims(mat, 1)
        v2 = tf.reshape(v1, [3, batch_size, 1])
        return tf.matmul(v2, v1)


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

    first_term = tf.reduce_sum(A, name="term-1")
    a_vec_cross = tf.matmul(tf.transpose(A), A, name="a-cross")
    y_target_cross = reshape_matmul(y_target)

    second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(a_vec_cross, y_target_cross)), name="term-1")
    loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)), name="loss")

    prediction_output = tf.matmul(tf.multiply(y_target, A), pred_kernel)
    prediction = tf.arg_max(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target, 0)), tf.float32), name="accuracy")
    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, name="optimizer")


    summary_writer = tf.summary.FileWriter(log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with tf.name_scope('Loss'):
        tf.summary.histogram('Histogram_Errors', loss)
        tf.summary.histogram('Histogram_pred', prediction)

    merged = tf.summary.merge_all()




with tf.Session(graph=graph) as session:

    tf.global_variables_initializer().run()
    print("Initialized")

    train_writer = tf.summary.FileWriter(log_path, session.graph)

    # 循环
    for i in range(iterations):
        rand_index = np.random.choice(len(x_vals), size=batch_size)
        rand_x = x_vals[rand_index]
        rand_y = y_vals[:, rand_index]

        summary, _, l, acc = session.run([merged, optimizer, loss, accuracy],
                                feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})
        loss_vec.append(l)

        batch_accuracy.append(acc)

        train_writer.add_summary(summary, i)

        if (i + 1) % 25 == 0:
            print('Step #' + str(i + 1))
            print('Loss = ' + str(l))

    # Create a mesh to plot points in
    x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
    y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_predictions = session.run(prediction, feed_dict={x_data: rand_x,
                                                          y_target: rand_y,
                                                          prediction_grid: grid_points})
    grid_predictions = grid_predictions.reshape(xx.shape)

# Plot points and grid
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='I. setosa')
plt.plot(class2_x, class2_y, 'kx', label='I. versicolor')
plt.plot(class3_x, class3_y, 'gv', label='I. virginica')
plt.title('Gaussian SVM Results on Iris Data')
plt.xlabel('Pedal Length')
plt.ylabel('Sepal Width')
plt.legend(loc='lower right')
plt.ylim([-0.5, 3.0])
plt.xlim([3.5, 8.5])
plt.show()

# Plot batch accuracy
plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
