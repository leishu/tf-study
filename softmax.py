"""Softmax."""

import numpy as np


#scores = [3.0, 1.0, 0.2]
#scores = [0.3, 0.1, 0.02]

#scores = [1.0, 2.0, 3.0]

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])


def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    ###return np.exp(x) / np.sum(np.exp(x), axis=0)

    y = [];
    for i in x:
        y.append(np.exp(i))

    pass  # TODO: Compute and return softmax(x)
    return y / sum(y)


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()