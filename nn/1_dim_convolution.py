import tensorflow as tf
import numpy as np


# 生成一维数据
data_size = 25
data_1d = np.random.normal(size=data_size)

graph = tf.Graph()

with graph.as_default():
    # 初始化 placeholders
    x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])


    # 卷积
    def conv_layer_1d(input_1d, my_filter):
        # TensorFlow的conv2d函数只适用于4D数组：
        # [batch, width, height, channels]
        # height是输入的长度
        input_2d = tf.expand_dims(input_1d, 0)
        input_3d = tf.expand_dims(input_2d, 0)
        input_4d = tf.expand_dims(input_3d, 3)
        # 使用stride = 1执行卷积
        convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1, 1, 1, 1], padding="VALID")
        # 降维
        return tf.squeeze(convolution_output)


    # 卷积过滤器
    my_filter = tf.Variable(tf.random_normal(shape=[1, 5, 1, 1]))
    # 卷积层，输出一维数组，长21
    my_convolution_output = conv_layer_1d(x_input_1d, my_filter)
    # 激活层，输出一维数组，长21
    my_activation_output = tf.nn.relu(my_convolution_output)


    # Max Pool
    def max_pool(input_1d, width):
        # 就像conv2d，max_pool也适用于4D数组
        # [batch_size=1, width=1, height=num_input, channels=1]
        input_2d = tf.expand_dims(input_1d, 0)
        input_3d = tf.expand_dims(input_2d, 0)
        input_4d = tf.expand_dims(input_3d, 3)
        # max-window ('width')
        pool_output = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='VALID')
        # 降维
        return tf.squeeze(pool_output)

    # 输出一维数组，17
    my_maxpool_output = max_pool(my_activation_output, width=5)


    # 全连接层
    def fully_connected(input_layer, num_outputs):
        # 首先找到需要的shape：权重矩阵的乘积
        # 维度是输入的长 × num_outputs
        weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs]]))
        # 权重初始化，17×5的二维数组
        weight = tf.random_normal(weight_shape, stddev=0.1)
        # 偏差初始化
        bias = tf.random_normal(shape=[num_outputs])
        # 为了矩阵乘，把1D的输入数组，变成2D的数组
        input_layer_2d = tf.expand_dims(input_layer, 0)

        full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)
        # 降维
        return tf.squeeze(full_output)


    my_full_output = fully_connected(my_maxpool_output, 5)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    feed_dict = {x_input_1d: data_1d}

    print(data_1d)

    print(session.run(my_convolution_output, feed_dict=feed_dict))

    print(session.run(my_activation_output, feed_dict=feed_dict))

    print(session.run(my_maxpool_output, feed_dict=feed_dict))

    print(session.run(my_full_output, feed_dict=feed_dict))

