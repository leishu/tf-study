import numpy as np
from six.moves import cPickle as pickle


class DataInit(object):

    def __init__(self):
        self.pickle_file = '../notMNIST.pickle'
        self.image_size = 28
        self.num_labels = 10

    def reformat(self, dataset, labels):
        dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(self.num_labels) == labels[:, None]).astype(np.float32)
        return dataset, labels

    def getDataSet(self):
        with open(self.pickle_file, 'rb') as f:
            save = pickle.load(f)
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            valid_dataset = save['valid_dataset']
            valid_labels = save['valid_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
            del save  # hint to help gc free up memory
            print('Training set', train_dataset.shape, train_labels.shape)
            print('Validation set', valid_dataset.shape, valid_labels.shape)
            print('Test set', test_dataset.shape, test_labels.shape)

            train_dataset, train_labels = self.reformat(train_dataset, train_labels)
            valid_dataset, valid_labels = self.reformat(valid_dataset, valid_labels)
            test_dataset, test_labels = self.reformat(test_dataset, test_labels)
            print('Training set', train_dataset.shape, train_labels.shape)
            print('Validation set', valid_dataset.shape, valid_labels.shape)
            print('Test set', test_dataset.shape, test_labels.shape)

        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels