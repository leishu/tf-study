# Tic-Tac-Toe又称井字棋，即在3 x 3的棋盘上，双方轮流落子，先将3枚棋子连成一线的一方获得胜利。
#
# Tic-Tac-Toe变化简单，只有765个可能局面，26830个棋局

# 该示例旨在将许多不同组合的最佳移动示例提供给神经网络，以便训练模型来玩Tic-Tac-Toe。
# 脚本的结尾可以和训练模型下棋。

# 考虑所有的几何转换，可以减少为：
# - 旋转90度.
# - 旋转180度.
# - 旋转270度.
# - 水平反射.
# - 垂直反射.

# 所有可能的转换都可以由基本转换生成，最多需要两次转换
# base_tic_tac_toe_moves.csv文件的每一行，都表示一个最佳应对

# 我们这样表示棋盘：'X' = 1, 'O'= -1，0表示空位置
# 最后一列是最佳应对位置索引，棋盘的索引：
# 0 | 1 | 2
# ---------
# 3 | 4 | 5
# ---------
# 6 | 7 | 8
# 举个例子，棋盘可能是这样的：
# O |   |
# ---------
# X | O | O
# ---------
#   |   | X
# 等于该行：[-1, 0, 0, 1, -1, -1, 0, 0, 1].



# 我们使用一个隐藏层的神经网络做预测


import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np
import random

response = 6
batch_size = 50
symmetry = ['rotate180', 'rotate90', 'rotate270', 'flip_v', 'flip_h']


# 打印棋盘
def print_board(board):
    symbols = ['O', ' ', 'X']
    board_plus1 = [int(x) + 1 for x in board]
    print(' ' + symbols[board_plus1[0]] + ' | ' + symbols[board_plus1[1]] + ' | ' + symbols[board_plus1[2]])
    print('___________')
    print(' ' + symbols[board_plus1[3]] + ' | ' + symbols[board_plus1[4]] + ' | ' + symbols[board_plus1[5]])
    print('___________')
    print(' ' + symbols[board_plus1[6]] + ' | ' + symbols[board_plus1[7]] + ' | ' + symbols[board_plus1[8]])


def get_symmetry(board, response, transformation):
    '''
    :param board: 整数list，长度为9:
     对手 = -1
     电脑 = 1
     空 = 0
    :param transformation: 位置转换:
     'rotate180', 'rotate90', 'rotate270', 'flip_v', 'flip_h'
    :return: tuple: (心棋盘, 应对)
    '''
    if transformation == 'rotate180':
        new_response = 8 - response
        return (board[::-1], new_response)
    elif transformation == 'rotate90':
        new_response = [6, 3, 0, 7, 4, 1, 8, 5, 2].index(response)
        tuple_board = list(zip(*[board[6:9], board[3:6], board[0:3]]))
        return ([value for item in tuple_board for value in item], new_response)
    elif transformation == 'rotate270':
        new_response = [2, 5, 8, 1, 4, 7, 0, 3, 6].index(response)
        tuple_board = list(zip(*[board[0:3], board[3:6], board[6:9]]))[::-1]
        return ([value for item in tuple_board for value in item], new_response)
    elif transformation == 'flip_v':
        new_response = [6, 7, 8, 3, 4, 5, 0, 1, 2].index(response)
        return (board[6:9] + board[3:6] + board[0:3], new_response)
    elif transformation == 'flip_h':  # flip_h = rotate180, then flip_v
        new_response = [2, 1, 0, 5, 4, 3, 8, 7, 6].index(response)
        new_board = board[::-1]
        return (new_board[6:9] + new_board[3:6] + new_board[0:3], new_response)
    else:
        raise ValueError('Method not implemented.')


def get_moves_from_csv(csv_file):
    '''
    :param csv_file: 文件
    :return: moves: 最佳应对列表
    '''
    moves = []
    with open(csv_file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            moves.append(([int(x) for x in row[0:9]], int(row[9])))
    return moves


def get_rand_move(moves, rand_transforms=2):
    '''
    :param moves: 应对列表
    :param rand_transforms: 每次执行多少个随机变换
    :return: (board, response), board是9个整数的list, response是一个整数
    '''
    (board, response) = random.choice(moves)
    possible_transforms = ['rotate90', 'rotate180', 'rotate270', 'flip_v', 'flip_h']
    for i in range(rand_transforms):
        random_transform = random.choice(possible_transforms)
        (board, response) = get_symmetry(board, response, random_transform)
    return board, response


# 判断输赢
def check(board):
    wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    for i in range(len(wins)):
        if board[wins[i][0]] == board[wins[i][1]] == board[wins[i][2]] == 1.:
            return 1
        elif board[wins[i][0]] == board[wins[i][1]] == board[wins[i][2]] == -1.:
            return 1
    return 0


moves = get_moves_from_csv('base_tic_tac_toe_moves.csv')

# 训练集
train_length = 500
train_set = []
for t in range(train_length):
    train_set.append(get_rand_move(moves))

# 要看网络是否学到新的东西，我们将会删除棋盘的所有实例
# 它的醉解应对是索引6,我们将在最后做测试
test_board = [-1, 0, 0, 1, -1, -1, 0, 0, 1]
train_set = [x for x in train_set if x[0] != test_board]


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape))


def model(X, A1, A2, bias1, bias2):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, A1), bias1))
    layer2 = tf.add(tf.matmul(layer1, A2), bias2)
    return layer2


# 学习速率
learning_rate = 0.025

graph = tf.Graph()

with graph.as_default():
    X = tf.placeholder(dtype=tf.float32, shape=[None, 9])
    Y = tf.placeholder(dtype=tf.int32, shape=[None])

    A1 = init_weights([9, 81])
    bias1 = init_weights([81])
    A2 = init_weights([81, 9])
    bias2 = init_weights([9])

    model_output = model(X, A1, A2, bias1, bias2)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=Y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    prediction = tf.argmax(model_output, 1)

loss_vec = []
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    for i in range(10000):
        rand_indices = np.random.choice(range(len(train_set)), batch_size, replace=False)
        batch_data = [train_set[i] for i in rand_indices]
        x_input = [x[0] for x in batch_data]
        y_target = np.array([y[1] for y in batch_data])

        _, l = session.run([optimizer, loss], feed_dict={X: x_input, Y: y_target})
        loss_vec.append(l)
        if i % 500 == 0:
            print('iteration %s Loss: %s' % (i, l))

    # 预测
    test_boards = [test_board]
    feed_dict = {X: test_boards}
    logits, predictions = session.run([model_output, prediction], feed_dict=feed_dict)
    print(predictions)

    plt.plot(loss_vec, 'k-', label='Loss')
    plt.title('Loss (MSE) per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.show()

    # 对抗
    game_tracker = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
    win_logical = False
    num_moves = 0
    while not win_logical:
        player_index = input('Input index of your move (0-8): ')
        num_moves += 1
        # 电脑
        game_tracker[int(player_index)] = 1.

        # 首先获取每个位置的logits值
        [potential_moves] = session.run(model_output, feed_dict={X: [game_tracker]})
        print(potential_moves)
        # 找到所有允许的位置 (空位置)
        allowed_moves = [ix for ix, x in enumerate(game_tracker) if x == 0.0]
        # 找到最佳位置
        model_move = np.argmax([x if ix in allowed_moves else -999.0 for ix, x in enumerate(potential_moves)])

        # 对手
        game_tracker[int(model_move)] = -1.
        print('Model has moved')
        print_board(game_tracker)
        # 检查输赢
        if check(game_tracker) == 1 or num_moves >= 5:
            print('Game Over!')
            win_logical = True
