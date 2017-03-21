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

sess.close()