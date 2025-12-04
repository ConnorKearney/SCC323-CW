import cupy as np

def MSE_loss(predicted, actual):
    total = 0
    for i in range(len(predicted)):
        total += (predicted[i] - actual[i])**2
    
    return total/len(predicted)



def Entropy_loss(predicted, actual, eps=1e-7):
    pred_clipped = np.clip(predicted, eps, 1-eps)
    
    total = 0
    for i in range(len(pred_clipped)):
        total += -actual[i] * np.log(pred_clipped[i])

    return total


def batch_Entropy_loss(predicted, actual):
    individual_loss = []
    for i in range(len(predicted)):
        individual_loss.append(Entropy_loss(predicted[i], actual[i]))

    return sum(individual_loss)/len(individual_loss)