import numpy as np
import activations
import DataExtractor
import lossFunctions



class DeepModel:
    def __init__(self, n_features=2, n_layers=1, n_nodes = 3, n_classes=2, lr=0.1, max_iter = 1000, lr_coef = 0.8, layer_shape_array = None, clipping_range=1.0, bn_momentum=0.9, batch_size=256):
        self.generator = np.random.default_rng(0)

        self.n_features = n_features

        self.n_outputs = n_classes
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.lr = lr
        self.lr_coef = lr_coef
        self.clipping_range = clipping_range
        self.max_iter = max_iter
        
        if (layer_shape_array):
            self.layer_shape_array = layer_shape_array
            self.n_nodes = None
            self.n_layers = len(self.layer_shape_array)
            self.weightInitialisationWithSpecificArrays()
        else:
            self.n_layers = n_layers
            self.n_nodes = n_nodes
            self.weightInitialisationWithUniformArray()


        self.internalActivationFunction = activations.ReLU
        self.outputActivationFunction = activations.Softmax   
        
        self.bn_momentum = bn_momentum
        self.epsilon = 1e-8

        self.bn_cache = [None] * len(self.layer_shape_array)

        self.print_info()
         

        
    def print_info(self):
        print("input weights:\n", self.input_weights)
        print("internal weights:\n", self.internal_weights)
        print("output weights:\n", self.output_weights)

    def weightInitialisationWithUniformArray(self):
        print("uniform internal structure")
        # He Normalisation as using ReLU
        self.input_weights = self.generator.normal(0, np.sqrt(2 / self.n_features), (self.n_features, self.n_nodes))

        self.internal_weights = self.generator.normal(0, np.sqrt(2 / self.n_nodes), (self.n_layers, self.n_nodes, self.n_nodes))

        std_dev_out = np.sqrt(2/(self.n_nodes + self.n_outputs)) 
        self.output_weights = self.generator.normal(0, std_dev_out, (self.n_nodes, self.n_outputs))

        self.u_values = np.zeros((self.n_layers, self.n_nodes)) # unactivated
        self.a_values = np.zeros((self.n_layers, self.n_nodes)) # activated

        self.internal_bias_matrices = np.zeros((self.n_layers, 1, self.n_nodes))
        self.output_bias = np.zeros((1, self.n_outputs))

    def weightInitialisationWithSpecificArrays(self):
        self.input_weights = self.generator.normal(0,np.sqrt(2/self.n_features), (self.n_features, self.layer_shape_array[0]))

        self.internal_weights = []
        self.u_values = []
        self.a_values = []

        self.internal_bias_matrices = []

        self.gamma_params = []
        self.beta_params = []
        self.running_mean = []
        self.running_var = []


        for i in range(len(self.layer_shape_array)):
        
            layer_size = self.layer_shape_array[i]
        
            self.internal_bias_matrices.append(np.zeros((1, layer_size)))
            self.u_values.append(np.zeros((layer_size, 1))) 
            self.a_values.append(np.zeros((layer_size, 1))) 

            self.gamma_params.append(np.ones((1,layer_size)))
            self.beta_params.append(np.zeros((1,layer_size)))

            self.running_mean.append(np.zeros((1,layer_size)))
            self.running_var.append(np.ones((1,layer_size)))

            if i < len(self.layer_shape_array) - 1:
                input_size = self.layer_shape_array[i]
                output_size = self.layer_shape_array[i+1]
                std_dev = np.sqrt(2 / input_size) # He initialization for ReLU
        
                self.internal_weights.append(self.generator.normal(0, std_dev, (input_size, output_size)))
            
        std_dev_out = np.sqrt(2/(self.layer_shape_array[-1] + self.n_outputs))     
        self.output_weights = self.generator.normal(0,std_dev_out, (self.layer_shape_array[-1], self.n_outputs))

        self.output_bias = np.zeros((1,self.n_outputs))


    def fit(self, X, y):
        number_of_batches = int(np.floor(len(X)/self.batch_size))
        X_batches = []
        y_batches = []

        for i in range(number_of_batches):
            X_batches.append(X[i*self.batch_size:(i+1)*self.batch_size])
            y_batches.append(y[i*self.batch_size:(i+1)*self.batch_size])


        X_batches, y_batches = np.array(X_batches), np.array(y_batches)

        self.fitBatches(X_batches, y_batches)

    
    def fitBatches(self, X, y):
        all_y_flat = [label for batch_y in y for label in batch_y]
        if not self.generate_classification_map(all_y_flat):
            print("class number mismatch")
            return None

        for epoch in range(self.max_iter):
            print(f"--- Starting Epoch: {epoch + 1}/{self.max_iter} ---")
            total_epoch_loss = 0
            
            for i in range(len(X)):
                X_batch = np.array(X[i])
                y_batch_raw = np.array(y[i])
                batch_size = len(X_batch)
                
                y_vector_form = np.zeros((batch_size, self.n_outputs))
                for j in range(batch_size):
                    y_vector_form[j][self.find_in_class_map(y_batch_raw[j])] = 1

                #print(f"fitting batch: {i+1} (size: {batch_size})")

                if self.layer_shape_array:
                    self.u_values = []
                    self.a_values = []
                    for layer_size in self.layer_shape_array:
                        self.u_values.append(np.zeros((batch_size, layer_size))) 
                        self.a_values.append(np.zeros((batch_size, layer_size)))
                else: 
                    self.u_values = np.zeros((self.n_layers, batch_size, self.n_nodes))
                    self.a_values = np.zeros((self.n_layers, batch_size, self.n_nodes))

                loss = self.nextIteration(X_batch, y_vector_form)
                
                if loss is None:
                    print(f"Error in batch {i+1}, stopping training.")
                    return
                
                total_epoch_loss += loss * batch_size # Accumulate weighted loss

            # --- Epoch End Logic ---
            
            # Decay learning rate every X epochs (500 in your original code)
            if epoch % 500 == 0 and epoch != 0:
                self.lr *= self.lr_coef
            
            # Print epoch loss every Y epochs (100 in your original code, adjusted for Epoch)
            avg_epoch_loss = total_epoch_loss / len(all_y_flat) # Divide by total samples
            if epoch % 1 == 0: # Print every epoch
                print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss}")

    def generate_classification_map(self, y):
        unique_classes = list(np.unique(y))
        
        if len(unique_classes) != self.n_classes:
            return None
        
        self.class_map = unique_classes
        return True

    def find_in_class_map(self, value:int):
        return self.class_map.index(value)

        
    def nextIteration(self, X, y):
        inputs = X
        batch_size = len(X)


        if self.layer_shape_array:
        # If they are lists of NumPy arrays, iterate and zero out contents
            for i in range(len(self.u_values)):
                self.u_values[i][:] = 0
                self.a_values[i][:] = 0
        else:
        # If they are single 3D NumPy arrays (uniform case), use direct assignment
            self.u_values[:] = 0
            self.a_values[:] = 0

        self.u_values[0] = inputs @ self.input_weights + self.internal_bias_matrices[0]
        self.u_values[0] = self.batchNormalization(self.u_values[0], 0)
        self.a_values[0] = self.internalActivationFunction(self.u_values[0])

        for layer in range(1,self.n_layers):
            self.u_values[layer] = self.a_values[layer-1] @ self.internal_weights[layer-1] + self.internal_bias_matrices[layer]

            self.u_values[layer] = self.batchNormalization(self.u_values[layer], layer)

            self.a_values[layer] = self.internalActivationFunction(self.u_values[layer])

        output_u_values = self.a_values[-1] @ self.output_weights + self.output_bias
        output = self.outputActivationFunction(output_u_values)

        # loss
        avg_loss = lossFunctions.batch_Entropy_loss(output, y) 

        if self.outputActivationFunction == activations.Softmax:
            d_output = output - y
        else:
            d_output = avg_loss * self.outputActivationFunction.d(output)


        d_output_weights = np.dot(self.a_values[-1].T, d_output) / batch_size
        d_output_bias = np.sum(d_output, axis=0, keepdims=True) / batch_size

        d_output_weights = np.clip(d_output_weights, -self.clipping_range, self.clipping_range)
        d_output_bias = np.clip(d_output_bias, -self.clipping_range, self.clipping_range)

        self.output_weights -= self.lr * d_output_weights
        self.output_bias -= self.lr * d_output_bias

        d_previous_layer = d_output

        for layer in range(self.n_layers-1, -1, -1):
            #print(layer)
            # deltas
            
            if layer == self.n_layers-1:
                d_activated_values = np.dot(d_previous_layer, self.output_weights.T)
            else:
                d_activated_values = np.dot(d_previous_layer, self.internal_weights[layer].T)

            d_bn_output = d_activated_values * self.internalActivationFunction.d(self.u_values[layer])

            d_unactivated_values, d_gamma, d_beta = self.batchNormBackpropogation(d_bn_output, layer)

            if layer==0:
                previous_activation = inputs
            else:
                previous_activation = self.a_values[layer-1]

            d_layer_weights = np.dot(previous_activation.T, d_unactivated_values) / batch_size
            d_layer_bias = (np.sum(d_unactivated_values, axis=0, keepdims=True) / batch_size)
            


            # clipping gradients
            d_layer_weights = np.clip(d_layer_weights, -self.clipping_range, self.clipping_range)
            d_layer_bias = np.clip(d_layer_bias, -self.clipping_range, self.clipping_range)
            d_gamma = np.clip(d_gamma, -self.clipping_range, self.clipping_range)
            d_beta = np.clip(d_beta, -self.clipping_range, self.clipping_range)

            # updates
            
            if layer == 0:
                self.input_weights -= self.lr * d_layer_weights
                self.internal_bias_matrices[0] -= self.lr * d_layer_bias
            else:
                self.internal_weights[layer-1] -= self.lr * d_layer_weights
                self.internal_bias_matrices[layer] -= self.lr * d_layer_bias

            self.gamma_params[layer] -= self.lr*d_gamma
            self.beta_params[layer] -= self.lr*d_beta


            d_previous_layer = d_unactivated_values 

        return avg_loss

    def batchNormalization(self, u_layer, layer_index, is_training=True):
        
        gamma = self.gamma_params[layer_index]
        beta = self.beta_params[layer_index]

        if (not is_training):
            mu_global = self.running_mean[layer_index]
            variance_global = self.running_var[layer_index]

            u_hat = (u_layer-mu_global) / (np.sqrt(variance_global + self.epsilon))

            y_layer = gamma * u_hat + beta
            return y_layer
        
        mu_B = np.mean(u_layer, axis=0, keepdims=True)
        variance_B = np.var(u_layer, axis=0, keepdims=True)

        self.running_mean[layer_index] = self.bn_momentum * self.running_mean[layer_index] + (1-self.bn_momentum) * mu_B
        self.running_var[layer_index] = self.bn_momentum * self.running_var[layer_index] + (1-self.bn_momentum) * variance_B

        u_hat = (u_layer - mu_B) / np.sqrt(variance_B + self.epsilon)

        self.bn_cache[layer_index] = {
                'u_hat': u_hat,
                'mu_B': mu_B,
                'sigma2_B': variance_B,
                'u_layer': u_layer, # needed for d_sigma2
                'sqrt_var': np.sqrt(variance_B + self.epsilon)
            }
        
        y_layer = gamma * u_hat + beta
        return y_layer

    def batchNormBackpropogation(self, d_out, layer):
        layer_cache = self.bn_cache[layer]
        u_hat = layer_cache['u_hat']
        u_layer = layer_cache['u_layer']
        mu_B = layer_cache['mu_B']
        sqrt_var = layer_cache['sqrt_var']
        var = sqrt_var**2

        gamma = self.gamma_params[layer]
        D = self.layer_shape_array[layer]

        d_beta = np.sum(d_out, axis=0, keepdims=True)

        d_gamma = np.sum(d_out * u_hat, axis=0, keepdims=True)
        d_u_hat = d_out * gamma

        d_inv_var = np.sum(d_u_hat * (u_layer-mu_B), axis=0, keepdims=True)

        d_u_minus_mu1 = d_u_hat * (1/sqrt_var)

        d_sqrt_var = d_inv_var * (-1/var)

        d_var = d_sqrt_var * 1/2 * (1/np.sqrt(var + self.epsilon))

        d_u_minus_mu2 = d_var * (2/self.batch_size) * (u_layer-mu_B)

        d_u_minus_mu = d_u_minus_mu1 + d_u_minus_mu2
        
        d_u_layer1 = d_u_minus_mu

        d_mu_B = -1 * np.sum(d_u_minus_mu, axis=0, keepdims=True)

        d_u_layer2 = d_mu_B * (1/self.batch_size)

        d_u_layer = d_u_layer1 + d_u_layer2

        return d_u_layer, d_gamma, d_beta
        




    def predict(self, X):
        inputs = X
        

        ## forward pass

        if self.layer_shape_array:
            # If they are lists of NumPy arrays, iterate and zero out contents
            for i in range(len(self.u_values)):
                self.u_values[i][:] = 0
                self.a_values[i][:] = 0
        else:
            # If they are single 3D NumPy arrays (uniform case), use direct assignment
            self.u_values[:] = 0
            self.a_values[:] = 0


        # input weights
        u_0 = np.dot(inputs, self.input_weights) + self.internal_bias_matrices[0]
        bn_0 = self.batchNormalization(u_0, 0, is_training=False)
        a_0 = self.internalActivationFunction(bn_0)
        current_activation = a_0

        # internal layers
        for layer in range(1,self.n_layers):
            u_l = np.dot(current_activation, self.internal_weights[layer-1]) + self.internal_bias_matrices[layer]
            bn_l = self.batchNormalization(u_l, layer, is_training=False)
            current_activation = self.internalActivationFunction(bn_l)

        # output layer
        output_u_values = np.dot(current_activation, self.output_weights) + self.output_bias
        probabilities = self.outputActivationFunction(output_u_values)

        output = np.argmax(probabilities, axis=1)

        return output
        
    def calculate_accuracy(self, X, y): # redo 

        predicted_classes = self.predict(X)
        for i in range(len(predicted_classes)):
            predicted_classes[i] = self.class_map[predicted_classes[i]]
        
        y_flat = y.flatten()

        correct_predictions = np.sum(predicted_classes == y_flat)
        

        total_samples = y.shape[0]
        
        accuracy = correct_predictions / total_samples
        
        return accuracy    



if __name__ == "__main__":
    D1 = DataExtractor.DataExtractor()
    
    train_data = []
    X_train = []
    y_train = []
    
   

    for i in range(1,6):
        current_batch = D1.readData(i)

        if current_batch:
            X_batch = [img.getLinearImage() for img in current_batch]
            y_batch = [img.getClassification() for img in current_batch]
        
            # Append the new batch array to the lists. This preserves the batch structure.
            X_train.extend(X_batch)
            y_train.extend(y_batch)

    X_train, y_train = np.array(X_train), np.array(y_train) 

    #X_train, y_train = X_train[:size], y_train[:size]

    #y_train = y_train[0:1000]

    DM1 = DeepModel(n_features=3072, n_layers=3, n_nodes=10, n_classes=5, lr=0.01, max_iter=400, layer_shape_array=[256,128,128,64,64,32,32,16,16,8], clipping_range=1)
    print("training")
    DM1.fit(X_train, y_train)
    
    test_data = D1.readData()
    X_test = np.array([img.getLinearImage() for img in test_data])
    y_test = np.array([img.getClassification() for img in test_data])


    print(DM1.calculate_accuracy(X_test, y_test))