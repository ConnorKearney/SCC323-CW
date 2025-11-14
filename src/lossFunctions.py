import numpy as np
import activations

def MSE_loss(predicted, actual):
    total = 0
    for i in range(len(predicted)):
        total += (predicted[i] - actual[i])**2
    
    return total/len(predicted)