import numpy as np
import math 

np.seterr(under="warn")

class ReLU_func:
    def _call_scalar(self,x):
            return np.maximum(0,x)
        
    def _d_scalar(self,x):
            if x>0:
                return 1
            return 0  
        
    def __call__(self,x):
        return np.vectorize(self._call_scalar)(x)
        
    def d(self,x):
        return np.vectorize(self._d_scalar)(x)

class Sigmoid_func:
    def _call_scalar(self, x):
        try:
            val = np.exp(x+1.0e-07)
        except RuntimeWarning as w:
            print(x, w)

        #if x != 0: print(x)    
        
        return np.where(x >= 0, 
                        1.0 / (1.0 + np.exp(-x)), 
                        np.exp(x) / (np.exp(x) + 1.0))
    
    def _d_scalar(self, x):
        value = self(x)
        return value * (1-value)
    
    
    def __call__(self, x):
        return np.vectorize(self._call_scalar)(x)
    
    def d(self, x):
        #print(x)
        return np.vectorize(self._d_scalar)(x)
        
    
class tanh_func:
    def __call__(self, x):
        numerator = (np.exp(x) - np.exp(-x))
        denominator = (np.exp(x) + np.exp(-x))
    
        return numerator/denominator
    
    def d(self, x):
        return 1-self(x)**2
    
class softmax_func:
    def _call_single(self,x:np.array):
        shift = np.max(x)
        shifted_x = x - shift
        
        return (np.exp(shifted_x)/np.sum(np.exp(shifted_x)))
    
    def _d_single(self,x):
        return self(x)*(np.identity(len(x))-self(x))
    
    def __call__(self, x):
        out = []
        for u_value in x:
            out.append(self._call_single(u_value))
        return np.array(out)
    
    def d(self,x):
        out = []
        for val in x:
            out.append(self._d_single(val))
        return np.array(out)


def Heaviside_Step(x, activation_value=0):
    if x > activation_value: return 1
    return 0

def PReLU(x, alpha=0.1):
    if x > 0: return x
    return (alpha*x)

def LeReLU(x):
    if x > 0: return x
    return (PReLU(x,0.01))

def Para_Sigmoid(x, alpha=1):
    return (1/(1+np.e**(-x/alpha)))

def softsign(x):
    return x/(1+abs(x))


# Derivatives

def d_SiLU(x):
    return Sigmoid(x) + x*d_Sigmoid(x)


ReLU = ReLU_func()
Softmax = softmax_func()
Sigmoid = Sigmoid_func()
tanh = tanh_func()


if __name__ == "__main__":
    value = 1
    param = 1
    print(ReLU(value))
    print(Heaviside_Step(value))
    print(Sigmoid(value))
    print(tanh(value))
    print(PReLU(value, param))
    print(LeReLU(value))
    print(Para_Sigmoid(value, param))
    print(softsign(value))

    print(Softmax(np.array([1.3,5.1,2.2,0.7,1.1])))
    
