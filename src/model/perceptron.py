# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
from sklearn.metrics import accuracy_score

from util.activation_functions import Activation
from model.classifier import Classifier


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learning_rate : float
    epochs : positive int

    Attributes
    ----------
    learning_rate : float
    epochs : int
    training_set : list
    validation_set : list
    test_set : list
    weight : ndarray
    """
    def __init__(self, train, valid, test, learning_rate=0.01, epochs=50):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        # Initialize the weight vector with small random values
        # around 0 and 0.1

        self.weight = np.random.rand(self.training_set.input.shape[1])/10

        # add bias weights at the beginning with the same random initialize
        self.weight = np.insert(self.weight, 0, np.random.rand()/10)

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)

    def train(self, verbose=True):
        """
        Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validation_set.label,
                                          self.evaluate(self.validation_set))
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy*100))
                print("-----------------------------")

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        for img, label in zip(self.training_set.input,
                              self.training_set.label):
            output = self._fire(img)  # real output of the neuron
            error = int(label) - int(output)
            # online learning: updating weights after seeing 1 instance
            self.weight += self.learning_rate * error * img

        # if we want to do batch learning, accumulate the error
        # and update the weight outside the loop

    def classify(self, test_instance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # Here you have to implement the classification for one instance,
        # i.e., return True if the testInstance is recognized as a 7,
        # False otherwise
        return self._fire(test_instance)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.test_set.input

        # Here is the map function of python - a functional programming concept
        # It applies the "classify" method to every element of "test"
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def _fire(self, inp):
        """Fire the output of the perceptron corresponding to the input """
        # I already implemented it for you to see how you can work with numpy
        return Activation.sign(np.dot(np.array(inp), self.weight))

    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
