# 线性支持向量机: Soft Margin
# hard版本要求每个点都能被正确区分。Soft Margin可以容忍少量噪声数据
# ----------------------------------
#
# 使用iris数据
#  x1 = 萼片长
#  x2 = 花瓣宽
# Class 1 : I. setosa
# Class -1: not I. setosa
#
# 我们知道x和y是线性可分的


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

graph = tf.Graph()

# 加载数据
# iris.data = [(萼片 Length, 萼片 Width, 花瓣 Length, 花瓣 Width)]
iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])

# 划分训练/测试集
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# 批大小
batch_size = 100
# 学习速率
learning_rate = 0.01
# 训练次数
iterations = 500

loss_vec = []
train_accuracy = []
test_accuracy = []

with graph.as_default():
    # 初始化placeholders
    x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # 线性回归变量
    A = tf.Variable(tf.random_normal(shape=[2, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    # 运算模型
    model_output = tf.subtract(tf.matmul(x_data, A), b)

    # L2 'norm'，向量A长度的平方
    l2_norm = tf.reduce_sum(tf.square(A))

    # 损失函数
    # Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
    # L2 正则化参数, alpha
    alpha = tf.constant([0.01])
    # hinge损失
    classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
    loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

    # 预测函数
    prediction = tf.sign(model_output)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    # 循环
    for i in range(iterations):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])

        _, l, slope, y_intercept = session.run([optimizer, loss, A, b], feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(l)

        train_acc_temp = session.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
        train_accuracy.append(train_acc_temp)

        test_acc_temp = session.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_accuracy.append(test_acc_temp)

        if (i + 1) % 100 == 0:
            print('Step #%s slope = %s y_intercept = %s' % (i + 1, slope, y_intercept))
            print('Loss = ' + str(l))


# 抽取系数
[[a1], [a2]] = slope
[[b]] = y_intercept
slope = -a2 / a1
y_intercept = b / a1

# 抽取 x1， x2
x1_vals = [d[1] for d in x_vals]

# 拟合线
best_fit = []
for i in x1_vals:
    best_fit.append(slope * i + y_intercept)

# 分离 I. setosa
setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]
not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == -1]


plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()


plt.plot(train_accuracy, 'k-', label='Training Accuracy')
plt.plot(test_accuracy, 'r--', label='Test Accuracy')
plt.title('Train and Test Set Accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
