import tensorflow as tf
import numpy as np
import math


N = tf.placeholder(dtype=tf.float32)
M = tf.placeholder(dtype=tf.float32)
bbb = tf.log(N)
aaa = tf.multiply(M, bbb)



A = tf.placeholder(shape=[2, 2], dtype=tf.float32)
matrix = np.arange(1, 5).reshape(2, 2)

y = tf.square(A)

sess = tf.Session()#建立会话


print(sess.run(bbb, feed_dict={N: 0.5}))

print(sess.run(aaa, feed_dict={N: 0.5, M: 2}))

print(sess.run(y, feed_dict={A: matrix}))

a = -2/3 * math.log2(2/3) -1/3 * math.log2(1/3)
print(-0.5 * math.log2(0.5))


l2_norm = tf.reduce_sum(tf.square(A))
print("l2_norm=", sess.run(l2_norm, feed_dict={A: matrix}))



X = tf.placeholder(shape=[2, 2], dtype=tf.float32)
matrixA = np.arange(1, 5).reshape(2, 2)
Y = tf.placeholder(shape=[2, 2], dtype=tf.float32)
matrixB = np.arange(5, 9).reshape(2, 2)

rA = tf.reshape(tf.reduce_sum(tf.square(X), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(Y), 1), [-1, 1])
sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(X, tf.transpose(Y)))),
                 tf.transpose(rB))


sq_, r1, r2 = sess.run([sq_dist, rA, rB], feed_dict={X: matrixA, Y: matrixB})
print("sq_dist=", sq_)
print(r1)
print(r2)

print("r1*r2=", sess.run(tf.multiply(r1, r2)))

sess.close()