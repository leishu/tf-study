import tensorflow as tf
import numpy as np

A = np.array([[4, 0], [0, 2], [1, 1]], dtype=np.float)
b = np.array([[2], [0], [11]], dtype=np.float)

# Create graph
sess = tf.Session()

# Create tensors
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

# Matrix inverse solution
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
tA_A_inv = tf.matrix_inverse(tA_A)
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
solution = tf.matmul(product, b_tensor)

solution_eval = sess.run(solution)

# Extract coefficients
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]

print('slope: ' + str(slope))
print('y_intercept: ' + str(y_intercept))