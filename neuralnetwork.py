#!/usr/bin/python
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
from math import log

def transfer_function(a):
    return a / float((1 + abs(a)))


def transfer_function_derivative(a):
    return 1 / float(1 + abs(a))**2


"""
Returns a function that returns a tuple (y,A) where y is the
output of the neural network and A is the a's before applying
an activation function to create the z's
"""
def neural_network(w1,w2,h):
    h = np.vectorize(h)
    def y(x):
        a = w1.dot(np.array(x))        
        return (w2.dot(np.append(h(a),1)), a)
    return y


# One dimentional sum-of-squares
def error_function(y,t):
    return 0.5*(y-t)**2


def error_backpropagation(w1,w2,h,dh):
    NN = neural_network(w1,w2,h)
    h = np.vectorize(h)
    dh = np.vectorize(dh)
    def gradient(x,t):
        # Forward-propagation
        y,a = NN(x)
        z = h(np.append(a,1))

        # Back-propagation
        d_k = y - t    
        d_j = dh(a).T*w2[:,:-1]*d_k

        # First and second layer derivatives
        dE1 = d_j*x
        dE2 = d_k*z

        return (dE1.T, dE2)
    return gradient

"""
This function creates random weights for the two layers of our
neural network. Note that we append a 1 to the output weights,
this is to represent the bias parameter.
"""
def create_weights(M):
    return (rnd.randn(M,2), np.insert(rnd.randn(1,M),M,1,axis=1))


def steepest_decent_training(max_iter, training_data, M, step_size):
    w1, w2 = create_weights(M)
    h = transfer_function
    dh = transfer_function_derivative
    NN_before = neural_network(w1,w2,h)
    error = []
    
    for i in range(0,max_iter):
        dE1, dE2 = np.zeros(w1.shape), np.zeros(w2.shape)
        grad = error_backpropagation(w1,w2,h,dh)
        NN = neural_network(w1,w2,h)

        # Compute the compound gradient for all training patterns
        # for the batch training.
        err = 0
        for x,t in data:
            err += (NN(np.array([[x],[1]]))[0] - t)**2
            new_dE1, new_dE2 = grad(np.array([[x],[1]]),t)            
            dE1 += new_dE1.T
            dE2 += new_dE2.T
        error.append(err / float(training_data.shape[0]))

        w1 = w1 - step_size * dE1
        w2 = w2 - step_size * dE2

    return (NN_before, neural_network(w1,w2,h), error)


def verify_gradient(M,e):
    f = open('data/sincTrain25.dt')
    data = np.loadtxt(f)

    # We use the first training pattern to verify our gradient
    x,t = data[0]
    x = np.array([[x],[1]])
    
    h = transfer_function
    dh = transfer_function_derivative
    E = error_function

    w1, w2 = create_weights(5)

    # Compute the difference between backpropagation and
    # finite differences for changing the input weights
    w1_diff = np.zeros(w1.shape)
    for i in range(0,M):
        for j in [0,1]:
            w1_e = np.copy(w1)
            w1_e[i,j] += e
            
            NN = neural_network(w1,w2,h)
            NN_e = neural_network(w1_e,w2,h)
            fin_diff = (E(NN_e(x)[0],t) - E(NN(x)[0],t)) / e

            grad = error_backpropagation(w1,w2,h,dh)
            dE, _ = grad(x,t)
            
            w1_diff[i,j] = abs(fin_diff - dE[i,j])

    #w2_diff = np.zeros(w2.shape)
    return w1_diff


if __name__ == "__main__":
    print verify_gradient(5,0.000001)


    """
    f = open('data/sincTrain25.dt')
    data = np.loadtxt(f)
    M = 20
    NN_b, NN_a, E = steepest_decent_training(10000,data,M,0.001)

    #plt.plot(E)
    #plt.yscale('log')
    #plt.show()


    X = data[:,0]
    X_cont = np.arange(min(X),max(X),0.1)
    t = data[:,1]
    first_guess = [NN_b(np.array([[x],[1]]))[0] for x in X_cont]
    Y = [NN_a(np.array([[x],[1]]))[0] for x in X_cont]

    plt.scatter(X,t,color='red',label='training')
    plt.plot(X_cont,first_guess,color='blue',label='initial guess')
    plt.plot(X_cont,Y,color='green',label='result')
    plt.legend()
    plt.show()
    """
