import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

x_vals = A.reshape((6, 1))
# Get best fit line
best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)


fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1)

#Plot the results
ax.plot(A, b, 'o', label='Data')
ax.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
ax.legend(loc='upper left')




x = np.linspace(0, 40, 30)
y = np.linspace(0, 20, 30)

ax = fig.add_subplot(1, 2, 2, projection='3d')
xx, yy = np.meshgrid(x, y, sparse=True)
z = (xx + 2 * yy) / 4

ax.plot_surface(xx, yy, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

