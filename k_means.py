import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

iris = datasets.load_iris()

# 150
num_pts = len(iris.data)
# 4
num_feats = len(iris.data[0])

# k-means 参数
# 有三类iris花
k = 3
generations = 25

graph = tf.Graph()

with graph.as_default():
    data_points = tf.Variable(iris.data)
    cluster_labels = tf.Variable(tf.zeros([num_pts], dtype=tf.int64))

    # 随机选择起始点，随机选三个数据
    rand_starts = np.array([iris.data[np.random.choice(len(iris.data))] for _ in range(k)])
    # shape(3, 4)
    centroids = tf.Variable(rand_starts)

    # 为了计算每个数据点和质心之间的距离，我们在k矩阵(num_points)内重复质心
    # shape(150, 3, 4)
    centroid_matrix = tf.reshape(tf.tile(centroids, [num_pts, 1]), [num_pts, k, num_feats])
    # reshape
    point_matrix = tf.reshape(tf.tile(data_points, [1, k]), [num_pts, k, num_feats])
    distances = tf.reduce_sum(tf.square(point_matrix - centroid_matrix), axis=2)

    centroid_group = tf.argmin(distances, 1)


    # group average
    def data_group_avg(group_ids, data):
        # 组求和
        sum_total = tf.unsorted_segment_sum(data, group_ids, 3)
        # count
        num_total = tf.unsorted_segment_sum(tf.ones_like(data), group_ids, 3)

        avg_by_group = sum_total / num_total
        return avg_by_group


    means = data_group_avg(centroid_group, data_points)

    update = tf.group(centroids.assign(means), cluster_labels.assign(centroid_group))

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    for i in range(generations):
        print('Calculating gen {}, out of {}.'.format(i, generations))
        _, centroid_group_count = session.run([update, centroid_group])
        group_count = []
        for ix in range(k):
            group_count.append(np.sum(centroid_group_count == ix))
        print('Group counts: {}'.format(group_count))

    [centers, assignments] = session.run([centroids, cluster_labels])


    # 组和标签对应
    # 首先，需要最常见元素方法
    def most_common(my_list):
        return (max(set(my_list), key=my_list.count))


    label0 = most_common(list(assignments[0:50]))
    label1 = most_common(list(assignments[50:100]))
    label2 = most_common(list(assignments[100:150]))

    group0_count = np.sum(assignments[0:50] == label0)
    group1_count = np.sum(assignments[50:100] == label1)
    group2_count = np.sum(assignments[100:150] == label2)

    accuracy = (group0_count + group1_count + group2_count) / 150.

    print('Accuracy: {:.2}'.format(accuracy))

# plot 输出
# 首先，使用PCA降维，从4维到2维
pca_model = PCA(n_components=2)
reduced_data = pca_model.fit_transform(iris.data)
# centers
reduced_centers = pca_model.transform(centers)

# 网格图每步的大小
h = .02

# 绘制边框，每种一个颜色
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 获取网格点的k-means分类
xx_pt = list(xx.ravel())
yy_pt = list(yy.ravel())
xy_pts = np.array([[x, y] for x, y in zip(xx_pt, yy_pt)])
mytree = cKDTree(reduced_centers)
dist, indexes = mytree.query(xy_pts)

# 将结果放入彩色图
indexes = indexes.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(indexes, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

# 绘制每个真实的iris数据组
symbols = ['o', '^', 'D']
label_name = ['Setosa', 'Versicolour', 'Virginica']
for i in range(3):
    temp_group = reduced_data[(i * 50):(50) * (i + 1)]
    plt.plot(temp_group[:, 0], temp_group[:, 1], symbols[i], markersize=10, label=label_name[i])
# 将质心绘制为白色X
plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on Iris Dataset\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='lower right')
plt.show()
