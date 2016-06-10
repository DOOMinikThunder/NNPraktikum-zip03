# -*- coding: utf-8 -*-

from random import random
from model.classifier import Classifier

__author__ = "ABC XYZ"  # Adjust this when you copy the file
__email__ = "ABC.XYZ@student.kit.edu"  # Adjust this when you copy the file


class StupidRecognizer(Classifier):
    """
    This class demonstrates how to follow an OOP approach to build a digit
    recognizer.

    It also serves as a baseline to compare with other
    recognizing method later on.

    The method is that it will randomly decide the digit is a "7" or not
    based on the probability 'by_chance'.
    """

    def __init__(self, train, valid, test, by_chance=0.5):

        self.by_chance = by_chance

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

    def train(self):
        # Do nothing
        pass

    def classify(self, test_instance):
        # byChance is the probability of being correctly recognized
        return random() < self.by_chance

    def evaluate(self):
        return list(map(self.classify, self.test_set.input))
