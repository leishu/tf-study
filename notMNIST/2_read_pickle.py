
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import numpy as np
import pylab as pl


n_row = 4
n_col = 5
"""Helper function to plot a gallery of portraits"""
pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

train_folders = ['notMNIST_large/A', 'notMNIST_large/B', 'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E',
                 'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 'notMNIST_large/I', 'notMNIST_large/J']

test_folders = ['notMNIST_small/A', 'notMNIST_small/B', 'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E',
                'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 'notMNIST_small/I', 'notMNIST_small/J']


def plot_gallery(files):
    for i in range(len(files)):
        pl.subplot(n_row, n_col, i + 1)
        with open(files[i] + '.pickle', 'rb') as f:
            letters = pickle.load(f)
            sample = np.random.randint(len(letters))

            pl.imshow(letters[sample], cmap=pl.cm.gray)
            pl.title(files[i], size=12)
            pl.xticks(())
            pl.yticks(())





train_folders.extend(test_folders)
plot_gallery(train_folders)

plt.show()


