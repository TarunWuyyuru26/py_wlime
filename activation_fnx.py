
# Loading Required Packages
import numpy as np

# Activations
def linear(x):
    return x
    
def binary_step(x):
    try:
        if(x >= 0):
            return(1)
        elif(x < 0):
            return(0)
        else:
            print("Incorrect input provided to parameter 'x'")
    except:
        print("Error Occured")

def sigmoid(x):
    return (1/(1 + np.exp(-x)))
    
def tanh(x):
    return ((np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)))

def relu(x):
    try:
        if(x > 0):
            return(x)
        elif(x <= 0):
            return(0)
        else:
            print("Incorrect input provided to parameter 'x'")
    except:
        print("Error Occured")
    
def leaky_relu(x):
    try:
        if(x > 0):
            return(x)
        elif(x <= 0):
            return(0.01*x)
        else:
            print("Incorrect input provided to parameter 'x'")
    except:
        print("Error Occured")
    
def param_relu(x, alpha=0.5):
    try:
        if(x > 0):
            return(x)
        elif(x <= 0):
            return(alpha*x)
        else:
            print("Incorrect input provided to parameter 'x'")
    except:
        print("Error Occured")

def bipolar_relu(x):
    try:
        if(x%2 == 0):
            return(relu(x))
        elif(x%2 != 0):
            return(-relu(-x))
        else:
            print("Incorrect input provided to parameter 'x'")
    except:
        print("Error occured")

def swish(x):
    return (x/(1 + np.exp(-x)))

def mish(x):
    return (x*tanh(np.log(1+np.exp(x))))
    
def softmax(x):
    exp_x = np.exp(x)
    total = sum(exp_x)
    probs = np.array(exp_x)/total
    return probs


















