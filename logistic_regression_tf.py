# 逻辑回归
# ----------------------------------
#
# 使用TensorFlow解决logistic回归问题
# y = sigmoid(Ax + b)
#
# 我们将使用出生时的低体重数据
#  y = 0 or 1 = 低体重
#  x = 人口统计学和病史资料

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
from tensorflow.python.framework import ops

graph = tf.Graph()

birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')[5:]
birth_header = [x for x in birth_data[0].split(' ') if len(x) >= 1]
birth_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
# 目标变量
y_vals = np.array([x[1] for x in birth_data])
# 预测因子 (不是id, 出生体重)
x_vals = np.array([x[2:9] for x in birth_data])

# 把数据分为训练集（80%）和测试集（20%）
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


# 按列标准化 (min-max)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)


x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# 批大小
batch_size = 25
# 学习速率
learning_rate = 0.02
# 训练次数
iterations = 1500

loss_vec = []
train_acc = []
test_acc = []

with graph.as_default():
    # 初始化placeholders
    x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # 增加回归变量
    A = tf.Variable(tf.random_normal(shape=[7, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    # 运算模型
    model_output = tf.add(tf.matmul(x_data, A), b)

    # 损失函数 (交叉熵)
    loss_ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_ce)
    # 实际预测
    prediction = tf.round(tf.sigmoid(model_output))
    predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    # 循环
    for i in range(iterations):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        _, loss = session.run([optimizer, loss_ce], feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(loss)

        temp_acc_train = session.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
        train_acc.append(temp_acc_train)
        temp_acc_test = session.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_acc.append(temp_acc_test)
        if (i + 1) % 300 == 0:
            print('Loss = ' + str(loss))

# 损失图
plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

# 训练和测试准确度
plt.plot(train_acc, 'k-', label='Train Set Accuracy')
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
