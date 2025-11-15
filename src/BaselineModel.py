import activations
import DataExtractor
import random
import numpy as np


class MLP():
    
    def __init__(self, n_features=2, n_layers=1, n_nodes=3, n_classes=2, lr=0.1, max_iter=1000, sample_size=None):
        random.seed(0)
        self.n_features = n_features
        if n_classes > 2:
            self.n_outputs = n_classes
        else:
            self.n_outputs = 1

        self.n_classes = n_classes
        self.lr = lr
        self.max_iter = max_iter
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        self.input_weights = np.random.rand(self.n_features, self.n_nodes)

        self.internal_weights = np.random.rand(self.n_layers, self.n_nodes, self.n_nodes)

        self.output_weights = np.random.rand(self.n_nodes, self.n_outputs)

        self.u_values = np.zeros((self.n_layers, self.n_nodes)) # unactivated
        self.a_values = np.zeros((self.n_layers, self.n_nodes)) # activated

        self.internal_bias_matrices = np.zeros((self.n_layers, self.n_nodes))
        self.output_bias = np.zeros((1, self.n_outputs))

        self.internalActivationFunction = activations.tanh
        self.outputActivationFunction = activations.Sigmoid

        self.sample_size = sample_size


    def fit_single_batch(self, X, y):
        for _ in range(self.max_iter):
            self.nextEpoch(X, y)


    def nextEpoch(self, X, y):
        
        # get the batch sample to use
        inputs = X

        ## forward pass

        # input weights
        self.u_values[0] = np.dot(inputs, self.input_weights) + self.internal_bias_matrices[0]
        self.a_values[0] = self.internalActivationFunction(self.u_values[0])

        # internal layers
        for layer in range(1,self.n_layers):
            self.u_values[layer] = np.dot(self.a_values[layer-1], self.internal_weights[layer-1]) + self.internal_bias_matrices[layer]
            self.a_values[layer] = self.internalActivationFunction(self.u_values[layer])

        # output layer
        output_u_values = np.dot(self.a_values[-1], self.output_weights) + self.output_bias
        output = self.outputActivationFunction(output_u_values)

        # error calcualtion
        error = output-y






    def predict():
        pass







if (__name__ == "__main__"):
    D1 = DataExtractor.DataExtractor()
    
    #read data_batch 1
    train_data = D1.readData(1)
    X_train = [img.getLinearImage for img in train_data]
    y_train = [img.getClassification for img in train_data]
    
    

    

    

    