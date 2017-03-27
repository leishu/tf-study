
import tensorflow as tf
from dataInit import DataInit
import numpy as np


data = DataInit()

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data.getDataSet()


# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
# 梯度下降训练的时候，不要太多数据
# 子集加快训练
train_subset = 10000

graph = tf.Graph()
with graph.as_default():
    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.
    # 输入数据
    # 加载训练、校验和测试数据
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
    # 变量
    # 他们是我们将要训练的参数
    # weight矩阵将被初始化成随机值，服从truncated正态分布（正态分布的一个区间，留下中间的，抛弃两端）
    # biases初始化为0
    weights = tf.Variable(
        tf.truncated_normal([data.image_size * data.image_size, data.num_labels]))
    biases = tf.Variable(tf.zeros([data.num_labels]))

    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.
    # 训练计算
    # inputs和weight矩阵相乘，再加上biases
    # 计算softmax和cross-entropy（在TensorFlow内是一个操作，因为它很常用，并能被优化）
    # 在全部训练集上取cross-entropy的均值，这就是loss
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    # 优化器
    # 使用梯度下降法，找到最小的loss
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    # 预测
    # 这不是训练的一部分，只是能反映训练精度
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 801


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


with tf.Session(graph=graph) as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the
    # biases.
    # 运行一次，确保参数初始化为我们描述的图
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        # 运算
        # 返回loss和predictions
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(
                predictions, train_labels[:train_subset, :]))
            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph
            # dependencies.
            # 返回一个numpy数组
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))