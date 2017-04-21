# 使用多层神经网络提升逻辑回归
# ----------------------------------
#
# y = sigmoid(A3 * sigmoid(A2* sigmoid(A1*x + b1) + b2) + b3)
# 我们将使用出生低体重数据：
#  y = 0 or 1 = 出生低体重
#  x = 人口统计和医疗历史数据

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
import os

birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')[5:]
birth_header = [x for x in birth_data[0].split(' ') if len(x) >= 1]
birth_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]

y_vals = np.array([x[1] for x in birth_data])
x_vals = np.array([x[2:9] for x in birth_data])

# 分割训练/测试集
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)


x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

batch_size = 90
# 学习速率
learning_rate = 0.002

log_path = "tensorboard/"

graph = tf.Graph()

with graph.as_default():
    x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)


    def init_variable(shape):
        return tf.Variable(tf.random_normal(shape=shape))


    # 逻辑层
    def logistic(input_layer, multiplication_weight, bias_weight, activation=True):
        linear_layer = tf.add(tf.matmul(input_layer, multiplication_weight), bias_weight)
        # 激活留到最后，在损失函数内实现sigmoid
        if activation:
            return tf.nn.sigmoid(linear_layer)
        else:
            return linear_layer


    # 第一个逻辑层 (7个输入到14个隐藏节点)
    A1 = init_variable(shape=[7, 14])
    b1 = init_variable(shape=[14])
    logistic_layer1 = logistic(x_data, A1, b1)

    # 第二个逻辑层 (14个隐藏输入到5个隐藏节点)
    A2 = init_variable(shape=[14, 5])
    b2 = init_variable(shape=[5])
    logistic_layer2 = logistic(logistic_layer1, A2, b2)

    # 最终输出层 (5个隐藏节点到一个输出)
    A3 = init_variable(shape=[5, 1])
    b3 = init_variable(shape=[1])
    final_output = logistic(logistic_layer2, A3, b3, activation=False)

    # 损失函数 (Cross Entropy loss)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_output, labels=y_target))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # 实际预测
    prediction = tf.round(tf.nn.sigmoid(final_output))
    predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)


    summary_writer = tf.summary.FileWriter(log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with tf.name_scope('Loss'):
        tf.summary.histogram('Histogram_Errors', loss)
        tf.summary.histogram('Histogram_pred', prediction)

    merged = tf.summary.merge_all()



loss_vec = []
train_acc = []
test_acc = []

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    train_writer = tf.summary.FileWriter(log_path, session.graph)

    for i in range(1500):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])

        _, l = session.run([optimizer, loss], feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(l)

        summary, temp_acc_train = session.run([merged, accuracy],
                                     feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})

        train_acc.append(temp_acc_train)
        train_writer.add_summary(summary, i)

        temp_acc_test = session.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_acc.append(temp_acc_test)
        if (i + 1) % 150 == 0:
            print('Loss = %s' % l)

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

# Plot train and test accuracy
plt.plot(train_acc, 'k-', label='Train Set Accuracy')
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
