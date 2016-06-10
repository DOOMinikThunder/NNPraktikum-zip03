# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import shuffle
from data.data_set import DataSet


class MNISTSeven(object):
    """
    Small subset (5000 instances) of MNIST data to recognize the digit 7

    Parameters
    ----------
    data_path : string
        Path to a CSV file with delimiter ',' and unint8 values.
    num_train : int
        Number of training examples.
    num_valid : int
        Number of validation examples.
    num_test : int
        Number of test examples.
    one_hot: bool
        If this flag is set, then all labels which are not `targetDigit` will
        be transformed to False and `targetDigit` bill be transformed to True.
        Set it to False for full MNIST task
    target_digit : string
        Label of the dataset, e.g. '7'.

    Attributes
    ----------
    training_set : list
    validation_set : list
    test_set : list
    """

    # dataPath = "data/mnist_seven.csv"

    def __init__(self, data_path,
                 num_train=3000,
                 num_valid=1000,
                 num_test=1000,
                 one_hot=True,
                 target_digit='7'):

        self.training_set = []
        self.validation_set = []
        self.test_set = []

        self.load(data_path, num_train, num_valid, num_test,
                  one_hot, target_digit)

    def load(self, data_path, num_train, num_valid, num_test,
             one_hot, target_digit):
        """Load the data."""
        print("Loading data from " + data_path + "...")

        data = np.genfromtxt(data_path, delimiter=",", dtype="uint8")

        # The last numTest instances ALWAYS comprise the test set.
        train, test = data[:num_train+num_valid], data[num_train+num_valid:]
        shuffle(train)

        train, valid = train[:num_train], train[num_train:]

        self.training_set = DataSet(train, one_hot=one_hot,
                                    target_digit=target_digit)
        self.validation_set = DataSet(valid, one_hot=one_hot,
                                      target_digit=target_digit)
        self.test_set = DataSet(test, one_hot=one_hot,
                                target_digit=target_digit)

        print("Data loaded.")
