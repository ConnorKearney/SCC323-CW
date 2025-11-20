import activations
import lossFunctions
import DataExtractor
import random
import numpy as np


class MLP():
    
    def __init__(self, n_features=2, n_layers=1, n_nodes=3, n_classes=2, lr=0.1, max_iter=10000, sample_size=None):
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

        # He Normalisation as using ReLU
        self.input_weights = np.random.normal(0, np.sqrt(2 / self.n_features), (self.n_features, self.n_nodes))

        self.internal_weights = np.random.normal(0, np.sqrt(2 / self.n_nodes), (self.n_layers, self.n_nodes, self.n_nodes))

        std_dev_out = np.sqrt(2/(self.n_nodes + self.n_outputs)) 
        self.output_weights = np.random.normal(0, std_dev_out, (self.n_nodes, self.n_outputs))

        self.u_values = np.zeros((self.n_layers, self.n_nodes)) # unactivated
        self.a_values = np.zeros((self.n_layers, self.n_nodes)) # activated

        self.internal_bias_matrices = np.zeros((self.n_layers, 1, self.n_nodes))
        self.output_bias = np.zeros((1, self.n_outputs))

        self.internalActivationFunction = activations.ReLU
        self.outputActivationFunction = activations.Softmax

        self.sample_size = sample_size
        self.class_map = np.array(self.n_classes)

        #self.print_info()

    def print_info(self):
        print("input weights:\n", self.input_weights)
        print("internal weights:\n", self.internal_weights)
        print("output weights:\n", self.output_weights)

        print((self.internal_weights < 1.0e-10).sum())
        print(np.sqrt(2 / self.n_features))
        print()

    def fit(self, X, y):
        for i in range(X.shape[0]):
            self.fit_single_batch(X[i], y[i])

    def fit_single_batch(self, X, y):
        for i in range(self.max_iter):
            
            loss = self.nextEpoch(X, y)
            print(i, " loss: ", loss)
            if i%1000 == 0:
                print(i, " loss: ", loss)
            if not loss.any():
                return
            
            #print(loss)

    def generate_classification_map(self, y):
        unique_classes = list(np.unique(y))
        
        if len(unique_classes) != self.n_classes:
            return None
        
        self.class_map = unique_classes
        return True

    def find_in_class_map(self, value:int):
        return self.class_map.index(value)

    def nextEpoch(self, X, y):
        
        # get the batch sample to use
        inputs = X
        batch_size = len(X)

        # used so that no matter what classes are used, the program can still calculate error
        if not self.generate_classification_map(y):
            print("class number mismatch")
            return None

        y_vector_form = np.zeros((batch_size, self.n_outputs))
        for i in range(batch_size):
            y_vector_form[i][self.find_in_class_map(y[i])] = 1

        #print("class map initialized")

        ## forward pass

        self.u_values = np.zeros((self.n_layers, batch_size, self.n_nodes)) # unactivated
        self.a_values = np.zeros((self.n_layers, batch_size, self.n_nodes)) # activated


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

        #print(output)
        #if (not np.all(output)):
            #print(output)

        # error calcualtion
        #error = output-y_vector_form

        # loss
        avg_loss = lossFunctions.batch_Entropy_loss(output, y_vector_form)
        #print("loss:",avg_loss)

        ## back propogation
        # output layer delta

        #d_output = avg_loss * self.outputActivationFunction.d(output)  # might need to change later
        d_output = output - y_vector_form
        d_output_weights = np.dot(self.a_values[-1].T, d_output) / batch_size
        d_output_bias = np.sum(d_output, axis=0, keepdims=True) / batch_size

        # update relevant weights
        self.output_weights -= self.lr * d_output_weights
        self.output_bias -= self.lr*d_output_bias

        # internal layers deltas and weights updates

        d_previous_layer = d_output
        

        for layer in range(self.n_layers-1, -1, -1):
            #print(layer)
            # deltas
            
            if layer == self.n_layers-1:
                d_activated_values = np.dot(d_previous_layer, self.output_weights.T)
            else:
                d_activated_values = np.dot(d_previous_layer, self.internal_weights[layer].T)

            d_unactivated_values = d_activated_values * self.internalActivationFunction.d(self.u_values[layer])

            if layer==0:
                previous_activation = inputs

            else:
                previous_activation = self.a_values[layer-1]

            d_layer_weights = np.dot(previous_activation.T, d_unactivated_values) / batch_size
            d_layer_bias = (np.sum(d_unactivated_values, axis=0, keepdims=True) / batch_size)

            # updates
            
            if layer == 0:
                self.input_weights -= self.lr * d_layer_weights
                self.internal_bias_matrices[0] -= self.lr * d_layer_bias
            else:
                self.internal_weights[layer-1] -= self.lr * d_layer_weights
                self.internal_bias_matrices[layer] -= self.lr * d_layer_bias

            d_previous_layer = d_unactivated_values 

        return avg_loss

    def __predict__(self, X):
        inputs = X
        batch_size = len(X)

        ## forward pass

        self.u_values = np.zeros((self.n_layers, batch_size, self.n_nodes)) # unactivated
        self.a_values = np.zeros((self.n_layers, batch_size, self.n_nodes)) # activated


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


    def predict(self, X): # redo
        """
        Performs a forward pass on the input data X to generate predictions.
        
        Args:
            X (np.ndarray): The input data matrix, shape (N_samples, n_features).

        Returns:
            tuple: (probabilities, predicted_classes).
        """
        inputs = X
        
        # --- Forward Pass ---

        # 1. Input Layer to First Hidden Layer (Layer 0)
        # Assumes internal_bias_matrices[0] is (1, n_nodes) for correct broadcasting.
        u_l0 = np.dot(inputs, self.input_weights) + self.internal_bias_matrices[0]
        a_l0 = self.internalActivationFunction(u_l0)

        current_activation = a_l0

        # 2. Internal Layers
        for layer in range(1, self.n_layers):
            # Assumes internal_bias_matrices[layer] is (1, n_nodes).
            u_l = np.dot(current_activation, self.internal_weights[layer-1]) + self.internal_bias_matrices[layer]
            current_activation = self.internalActivationFunction(u_l)

        # 3. Output Layer
        # Assumes output_bias is (1, n_outputs).
        output_u_values = np.dot(current_activation, self.output_weights) + self.output_bias
        
        probabilities = self.outputActivationFunction(output_u_values)
        
        # --- Classification ---
        # Get the index of the highest probability (the predicted class)
        predicted_classes = np.argmax(probabilities, axis=1)

        return probabilities, predicted_classes

        
    def calculate_accuracy(self, X, y): # redo
        """
        Calculates the classification accuracy of the model on the given dataset.
        
        Args:
            X (np.ndarray): The input data matrix, shape (N_samples, n_features).
            y (np.ndarray): The true target labels (integer class indices).
            
        Returns:
            float: The accuracy (proportion of correct predictions).
        """
        import numpy as np
        
        # 1. Get predictions from the forward pass
        # The predict function returns (probabilities, predicted_classes). We only need the latter.
        _, predicted_classes = self.predict(X)
        
        # 2. Ensure the true labels are a flat vector for comparison
        # This handles cases where y might be (N_samples, 1) or (N_samples,).
        y_flat = y.flatten()

        # 3. Compare predictions to true labels
        # np.sum(predicted_classes == y_flat) counts the number of correct matches.
        correct_predictions = np.sum(predicted_classes == y_flat)
        
        # 4. Calculate accuracy
        total_samples = y.shape[0]
        
        # Accuracy is (Correct / Total)
        accuracy = correct_predictions / total_samples
        
        return accuracy




if (__name__ == "__main__"):
    D1 = DataExtractor.DataExtractor()
    
    #read data_batch 1
    train_data = D1.readData(1)
    X_train = np.array([img.getLinearImage() for img in train_data])
    y_train = np.array([img.getClassification() for img in train_data])
    
    X_train = np.divide(X_train, 255)
    #y_train = y_train[0:1000]

    MLP1 = MLP(n_features=3072, n_layers=2, n_nodes=10, n_classes=5, lr=0.01)

    MLP1.fit(np.array([X_train]), np.array([y_train]))
    
    test_data = D1.readData()
    X_test = np.array([img.getLinearImage() for img in test_data])
    y_test = np.array([img.getClassification() for img in test_data])


    print(MLP1.calculate_accuracy(X_test, y_test))

    

    

    