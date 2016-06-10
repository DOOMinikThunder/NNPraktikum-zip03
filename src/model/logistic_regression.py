# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
from sklearn.metrics import accuracy_score

# from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learning_rate : float
    epochs : positive int

    Attributes
    ----------
    training_set : list
    validation_set : list
    test_set : list
    learning_rate : float
    epochs : positive int
    performances: array of floats
    """

    def __init__(self, train, valid, test, learning_rate=0.01, epochs=50):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        # Use a logistic layer as one-neuron classification (output) layer
        self.layer = LogisticLayer(train.input.shape[1], 1,
                                   is_classifier_layer=True)

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validation_set.label,
                                          self.evaluate(self.validation_set))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        for img, label in zip(self.training_set.input,
                              self.training_set.label):

            # Use LogisticLayer to do the job
            # Feed it with inputs

            # Do a forward pass to calculate the output and the error
            self.layer.forward(img)

            # Compute the derivatives w.r.t to the error
            # Please note the treatment of nextDerivatives and nextWeights
            # in case of an output layer
            self.layer.computeDerivative(np.array(label - self.layer.outp),
                                         np.array(1.0))

            # Update weights in the online learning fashion
            self.layer.updateWeights(self.learning_rate)

    def classify(self, test_instance):
        """Classify a single instance.

        Parameters
        ----------
        test_instance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """

        # Here you have to implement classification method given an instance
        outp = self.layer.forward(test_instance)
        return outp > 0.5

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
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
