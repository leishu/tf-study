import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops


# 加载数据(Sepal: 萼片, Petal: 花瓣)
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])
class1_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class1_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class2_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == -1]
class2_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == -1]

graph = tf.Graph()

# 批大小
batch_size = 150
# 学习速率
learning_rate = 0.01
# 训练次数
iterations = 300
loss_vec = []
batch_accuracy = []

with graph.as_default():
    # 初始化placeholders
    x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)

    # 增加SVM变量
    A = tf.Variable(tf.random_normal(shape=[1, batch_size]))

    gamma = tf.constant(-25.0)

    # RBF核
    def rbf(x, y):
        # 行平方和
        rA = tf.reshape(tf.reduce_sum(tf.square(x), 1), [-1, 1])
        rB = tf.reshape(tf.reduce_sum(tf.square(y), 1), [-1, 1])
        # rA - 2 × x * y + rB
        sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x, tf.transpose(y)))),
                         tf.transpose(rB))
        return tf.exp(tf.multiply(gamma, tf.abs(sq_dist)))

    # my_kernel的两种计算方法，第一种，loss线性下降
    #sq_dists = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
    #my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

    my_kernel = rbf(x_data, x_data)
    pred_kernel = rbf(x_data, prediction_grid)

    # 计算SVM模型
    first_term = tf.reduce_sum(A)

    b_vec_cross = tf.matmul(tf.transpose(A), A)
    y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
    second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
    loss = tf.negative(tf.subtract(first_term, second_term))

    prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), A), pred_kernel)
    prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

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

        if (i + 1) % 75 == 0:
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


plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='I. setosa')
plt.plot(class2_x, class2_y, 'kx', label='Non setosa')
plt.title('Gaussian SVM Results on Iris Data')
plt.xlabel('Pedal Length')
plt.ylabel('Sepal Width')
plt.legend(loc='lower right')
plt.ylim([-0.5, 3.0])
plt.xlim([3.5, 8.5])
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