import numpy as np
from six.moves import cPickle as pickle
import os



def matrix_cmp(matrix_1, matrix_2):
    if matrix_1.shape[1] != matrix_2.shape[1]:
        print('Input vectors must havs same size!')
        return -1
    count = 0
    for i in matrix_1:
        for j in matrix_2:
            dis = np.linalg.norm(i - j)
            if dis == 0:
                count = count + 1
    return count / matrix_1.shape[0]



pickle_file = os.path.join('.', 'notMNIST.pickle')

try:
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

        train_data = data['train_dataset']
        # test_data = data['test_dataset']
        valid_data = data['valid_dataset']

        print(matrix_cmp(train_data, valid_data))

except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise