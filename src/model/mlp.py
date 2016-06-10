
import numpy as np

# from util.activation_functions import Activation
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
                 cost='crossentropy', learning_rate=0.01, epochs=50):

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


        # Build up the network from specific layers
        # Here is an example of a MLP acting like the Logistic Regression
        self.layers = []
        #output_activation = "sigmoid"
    
        self.layers.append(LogisticLayer(train.input.shape[1], 16, None, self.output_activation, True))
        self.layers.append(LogisticLayer(16, len(set(train.label)), None, self.output_activation, True))
        #self.layers.append(LogisticLayer(10, 1, None, output_activation, True))
        
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
        
        newInput = inp
        # for all layers
        for l in self.layers:
            newInput = l.forward(newInput)
            # add bias values ("1"s) at the beginning of all data sets
            newInput = np.insert(newInput, 0, 1, axis=0)
            
        return newInput
    

    def _compute_error(self, target):
        """
        Compute the total error of the network

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        pass

    def _update_weights(self):
        """
        Update the weights of the layers by propagating back the error
        """
        pass

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

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
            #self.layer.forward(img)
            output = self._feed_forward(img)

            # Compute the derivatives w.r.t to the error
            # Please note the treatment of nextDerivatives and nextWeights
            # in case of an output layer
            
            
           

            #self._get_output_layer.computeDerivative(np.array(label - self._get_output_layer.outp),
            #                             np.array(1.0))
            #softmaxDistribution = Activation.get_activation(self.output_activation)(self._get_output_layer.outp)
            softmaxDistribution = Activation.softmax(output)
            
            outputDeltas = np.ndarray(len(softmaxDistribution))
            for i in xrange(len(softmaxDistribution)):
                if i == label:
                    outputDeltas[i] = 1 - softmaxDistribution[i]
                else:
                    outputDeltas[i] = 0 - softmaxDistribution[i]
                    
            self._get_output_layer().computeDerivative(outputDeltas,
                                                     None)
            
            
            # (self.totalLayers - 1) because output layer already covered
            for i in range((self.totalLayers - 2), -1, -1):
                self._get_layer(i).computeDerivative(self._get_layer(i+1).deltas,
                                                    self._get_layer(i+1).weights)
                
            # Update weights in the online learning fashion
            for l in self.layers:
                l.updateWeights(self.learning_rate)
        
        
    def classify(self, test_instance, test_label):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        
        output = self._feed_forward(test_instance)
        #softmaxDistribution = Activation.get_activation(self.output_activation)(output)
        softmaxDistribution = Activation.softmax(output)
        
        # get index of highest probability
#         highestProbability = 0
#         indexOfHighestProbability = 0
#         for i in xrange(len(softmaxDistribution)):
#             if softmaxDistribution[i] > highestProbability:
#                 highestProbability = softmaxDistribution[i]
#                 indexOfHighestProbability = i
            
            
            
            
            
                
#         max_index = 0
#         for i in range(1, len(softmaxDistribution)):
#             if(softmaxDistribution[i] > softmaxDistribution[max_index]):
#                 max_index = i


        print softmaxDistribution
        

        max_index = np.argmax(softmaxDistribution)

        print "LAAAAAAAAAAAAAAAAAABEL"
        print test_label                
        print "LAAAAAAAAAAAAAAAAAABEL"
        #print test_instance
        print "BBBBBBBBBBBBBBBBBBBBB"
        print max_index
        #print "===================="
        #print self.findByInput(test_instance)
        print "BBBBBBBBBBBBBBBBBBBBB"
        
        
        return max_index
        
        
#     def findByInput(self, target_img):
#         for label, img in zip(self.test_set.label, self.test_set.input):
#             counter = 0
#             for i, t in zip(img, target_img):
# 
#                 if i == t:
#                     counter = counter + 1
#             if (counter) == (len(img)):
#                 return label
#         return -1

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
        #print self.test_set.input
        #print self.test_set.input[0]
        
            #test = self.test_set
        # Once you can classify an instance, just use map for all of the test
        # set.
        evalList = []
        for i in xrange(len(self.test_set.input)):
            evalList.append(self.classify(self.test_set.input[i], self.test_set.label[i]))
            
        return evalList
    
        #return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
