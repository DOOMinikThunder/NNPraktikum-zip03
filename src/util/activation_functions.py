# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

from numpy import exp
from numpy import divide
from numpy import sum


class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(outp, threshold=0):
        return outp >= threshold

    @staticmethod
    def sigmoid(outp):
        # use e^x from numpy to avoid overflow
        return 1/(1+exp(-1.0*outp))

    @staticmethod
    def sigmoid_prime(outp):
        # Here you have to code the derivative of sigmoid function
        # netOutput.*(1-netOutput)
        return outp*(1.0-outp)

    @staticmethod
    def tanh(outp):
        # return 2*Activation.sigmoid(2*netOutput)-1
        ex = exp(1.0*outp)
        exn = exp(-1.0*outp)
        return divide(ex-exn, ex+exn)  # element-wise division

    @staticmethod
    def tanh_prime(outp):
        # Here you have to code the derivative of tanh function
        pass

    @staticmethod
    def identity(outp):
        return lambda x: x

    @staticmethod
    def identity_prime(outp):
        # Here you have to code the derivative of identity function
        pass

    @staticmethod
    def softmax(outp):
        # Here you have to code the softmax function

        return exp(outp) / sum(exp(outp), axis=0)


    @staticmethod
    def softmax_prime(outp):
        # Here you have to code the the derivative of softmax function
        pass

    @staticmethod
    def get_activation(function_name):
        """
        Returns the activation function corresponding to the given string
        """

        if function_name == 'sigmoid':
            return Activation.sigmoid
        elif function_name == 'softmax':
            return Activation.softmax
        elif function_name == 'tanh':
            return Activation.tanh
        elif function_name == 'linear':
            return Activation.identity
        else:
            raise ValueError('Unknown activation function: ' + function_name)

    @staticmethod
    def get_derivative(function_name):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if function_name == 'sigmoid':
            return Activation.sigmoid_prime
        elif function_name == 'tanh':
            return Activation.tanh_prime
        elif function_name == 'softmax':
            return Activation.softmax_prime
        elif function_name == 'linear':
            return Activation.identity_prime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + function_name)
