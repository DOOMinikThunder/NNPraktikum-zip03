
import numpy as np
import sys

from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from util.activation_functions import Activation
from sklearn.metrics import accuracy_score



class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='softmax',
                 inputActivation='sigmoid', cost='crossentropy',
                 learning_rate=0.01, epochs=50, learningRateReductionFactor=1.0,
                 layerNeurons=[10]):

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

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_task = output_task  # Either classification or regression
        self.output_activation = output_activation
        self.cost = cost

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers
        self.input_weights = input_weights

        
        # activation function for the hidden layers
        self.inputActivation = inputActivation
        # reduction factor of learning rate per epoch
        self.learningRateReductionFactor = learningRateReductionFactor
        
        
        #######################
        #    CREATE LAYERS    #
        #######################
        # Build up the network from specific layers
        self.layers = []
        # check for correct argument
        if (len(layerNeurons) < 1):
            raise ValueError('Error: layerNeurons must contain at least one layer with neurons!')
        # if there is only one layer it is an output layer
        if (len(layerNeurons) == 1):
            self.layers.append(LogisticLayer(train.input.shape[1], layerNeurons[0], None, self.output_activation, True))
        # if there are more than one layer
        else:
            # first layer (hidden layer)
            self.layers.append(LogisticLayer(train.input.shape[1], layerNeurons[0], None, self.inputActivation, False))
            # rest of the hidden layers
            for i in xrange(1, len(layerNeurons)-1):
                self.layers.append(LogisticLayer(layerNeurons[i-1], layerNeurons[i], None, self.inputActivation, False))
            # output layer
            self.layers.append(LogisticLayer(layerNeurons[len(layerNeurons)-2], layerNeurons[len(layerNeurons)-1], None, self.output_activation, True))
     
        
        # total number of output neurons
        self.totalOutputs = layerNeurons[len(layerNeurons)-1]
        # total number of layers
        self.totalLayers = len(self.layers)


        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)



    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        
        # do the forward pass for the first layer separately (because the bias was already added)
        inp = self._get_layer(0).forward(inp)
        
        # for all layers except the first: do the forward pass
        for i in xrange(1, self.totalLayers):
            # add bias value ("1") at the beginning
            inp = np.insert(inp, 0, 1, axis=0)
            inp = self._get_layer(i).forward(inp)
    

    def _compute_error(self, target):
        """
        Compute the total error of the network

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        
        outputDeltas = np.ndarray(self.totalOutputs)
        for i in xrange(self.totalOutputs):
            # index equals the class label, so the "1" is at the index of the class label
            if i != target:
                outputDeltas[i] = 0 - self._get_output_layer().outp[i]
            else:
                outputDeltas[i] = 1 - self._get_output_layer().outp[i]
        
        return outputDeltas
        

    def _update_weights(self):
        """
        Update the weights of the layers by propagating back the error
        """
        
        for l in self.layers:
            l.updateWeights(self.learning_rate)
            

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        
        # Run the training "epochs" times, print out the logs
        for epoch in xrange(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))


            self._train_one_epoch()
            # reduce learning_rate each epoch
            self.learning_rate *= self.learningRateReductionFactor


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
            
            ########################
            #    FORWARD PASSES    #
            ########################
            # forward passes through all layers
            self._feed_forward(img)

            ###################################
            #    BACKPROPAGATION OF DELTAS    #
            ###################################
            # do the backpropagation for the output layer:
            # compute output deltas and pass to output layer
            outputDeltas = self._compute_error(label)
            self._get_output_layer().computeDerivative(outputDeltas,
                                                       None)
            # do the backpropagation for all hidden layers:
            # go backwards and ignore the last/output layer
            for i in xrange((self.totalLayers - 2), -1, -1):
                self._get_layer(i).computeDerivative(self._get_layer(i+1).deltas,
                                                     self._get_layer(i+1).weights)
        
            ########################
            #    WEIGHT UPDATES    #
            ########################
            # Update weights in the online learning fashion for all layers
            self._update_weights()
        
        
    def classify(self, test_instance):#, test_label):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        
        self._feed_forward(test_instance)
        # return index of highest probability (indices are equal to the class labels)
        return np.argmax(self._get_output_layer().outp)
    

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
        
#         # following code does not work somehow
#         evalList = []
#         for i in xrange(len(self.test_set.input)):
#             evalList.append(self.classify(self.test_set.input[i], self.test_set.label[i]))
#              
#         return evalList
        
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))


    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
