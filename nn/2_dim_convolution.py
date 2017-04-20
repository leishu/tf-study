import tensorflow as tf
import numpy as np

# 生成二维数据
data_size = [10, 10]
data_2d = np.random.normal(size=data_size)

graph = tf.Graph()

with graph.as_default():
    # 初始化 placeholders
    x_input_2d = tf.placeholder(dtype=tf.float32, shape=data_size)


    # 卷积
    def conv_layer_2d(input_1d, my_filter):
        # TensorFlow的conv2d函数只适用于4D数组：
        # [batch, width, height, channels]
        # height是输入的长度
        input_3d = tf.expand_dims(input_1d, 0)
        # [1, 10, 10, 1]
        input_4d = tf.expand_dims(input_3d, 3)
        # 使用stride = 1执行卷积
        convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1, 2, 2, 1], padding="VALID")
        # 降维
        return tf.squeeze(convolution_output)


    # 卷积过滤器
    my_filter = tf.Variable(tf.random_normal(shape=[2, 2, 1, 1]))
    # 卷积层，输出5×5
    my_convolution_output = conv_layer_2d(x_input_2d, my_filter)
    # 激活层
    my_activation_output = tf.nn.relu(my_convolution_output)


    # Max Pool
    def max_pool(input_2d, width, height):
        # 就像conv2d，max_pool也适用于4D数组
        # [batch_size=1, width=given, height=given, channels=1]
        input_3d = tf.expand_dims(input_2d, 0)
        input_4d = tf.expand_dims(input_3d, 3)

        pool_output = tf.nn.max_pool(input_4d, ksize=[1, height, width, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='VALID')
        # 降维
        return tf.squeeze(pool_output)


    # Max-Pool层,输出4×4
    my_maxpool_output = max_pool(my_activation_output, width=2, height=2)


    # 全连接层
    def fully_connected(input_layer, num_outputs):
        # 输出是1D数组
        flat_input = tf.reshape(input_layer, [-1])
        # [16, 5]二维
        weight_shape = tf.squeeze(tf.stack([tf.shape(flat_input), [num_outputs]]))
        # 权重初始化
        weight = tf.random_normal(weight_shape, stddev=0.1)
        # 偏差初始化
        bias = tf.random_normal(shape=[num_outputs])

        input_2d = tf.expand_dims(flat_input, 0)

        full_output = tf.add(tf.matmul(input_2d, weight), bias)

        return tf.squeeze(full_output)


    my_full_output = fully_connected(my_maxpool_output, 5)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    feed_dict = {x_input_2d: data_2d}

    print(data_2d)

    print(session.run(my_convolution_output, feed_dict=feed_dict))

    print(session.run(my_activation_output, feed_dict=feed_dict))

    print(session.run(my_maxpool_output, feed_dict=feed_dict))

    print(session.run(my_full_output, feed_dict=feed_dict))
