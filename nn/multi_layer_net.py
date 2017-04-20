import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests

birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')[5:]
birth_header = [x for x in birth_data[0].split(' ') if len(x) >= 1]
birth_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]

# 抽取 y-target (birth weight)
y_vals = np.array([x[10] for x in birth_data])

# 过滤感兴趣的特征
cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI', 'FTV']
x_vals = np.array(
    [[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data])

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

# 学习速率
learning_rate = 0.05
batch_size = 50

graph = tf.Graph()

with graph.as_default():
    def init_weight(shape, st_dev):
        weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
        return weight


    def init_bias(shape, st_dev):
        bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
        return bias


    x_data = tf.placeholder(shape=[None, 8], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)


    # 全连接层
    def fully_connected(input_layer, weights, biases):
        layer = tf.add(tf.matmul(input_layer, weights), biases)
        return tf.nn.relu(layer)


    # 第一层(50 hidden nodes)
    weight_1 = init_weight(shape=[8, 25], st_dev=10.0)
    bias_1 = init_bias(shape=[25], st_dev=10.0)
    layer_1 = fully_connected(x_data, weight_1, bias_1)

    # 第二层(25 hidden nodes)
    weight_2 = init_weight(shape=[25, 10], st_dev=10.0)
    bias_2 = init_bias(shape=[10], st_dev=10.0)
    layer_2 = fully_connected(layer_1, weight_2, bias_2)

    # 第三层(5 hidden nodes)
    weight_3 = init_weight(shape=[10, 5], st_dev=10.0)
    bias_3 = init_bias(shape=[5], st_dev=10.0)
    layer_3 = fully_connected(layer_2, weight_3, bias_3)

    # 第四层(1 output value)
    weight_4 = init_weight(shape=[5, 1], st_dev=10.0)
    bias_4 = init_bias(shape=[1], st_dev=10.0)
    final_output = fully_connected(layer_3, weight_4, bias_4)

    # 损失函数 (L1)
    loss = tf.reduce_mean(tf.abs(y_target - final_output))

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

loss_vec = []
test_loss = []

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    for i in range(1000):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])

        _, l = session.run([optimizer, loss], feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(l)

        test_l = session.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_loss.append(test_l)
        if (i + 1) % 200 == 0:
            print('Generation: %s. Loss = %s' % (i + 1, l))

# Plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
