import numpy as np
from six.moves import cPickle as pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


###50, 100, 1000 and 5000

train_num = 1000
test_num = 500

def linereg(train_data, train_labels, test_dataset, test_labels):
    reg = LogisticRegression()
    reg.fit(train_data, train_labels)
    pred = reg.predict(test_dataset)

    acc = accuracy_score(pred, test_labels)
    return acc


pickle_file = os.path.join('.', 'notMNIST.pickle')

try:
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

        train_data = data['train_dataset']
        test_data = data['test_dataset']
        train_labels = data['train_labels']
        test_labels = data['test_labels']

        train_num = len(train_data)
        test_num = len(test_data)

        train_data = train_data[:train_num].reshape(train_num, -1)
        test_data = test_data[:test_num].reshape(test_num, -1)
        train_labels = train_labels[:train_num]
        test_labels = test_labels[:test_num]


        print(linereg(train_data, train_labels, test_data, test_labels))

except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise