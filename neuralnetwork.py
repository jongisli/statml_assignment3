#!/usr/bin/python
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
from math import log, sin

img_format = 'png'

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
    return (rnd.sample([M,2]), rnd.sample([1,M+1]))


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

        # Compute the average gradient for all training patterns
        # for the batch training.
        err = 0
        for x,t in data:
            err += (NN(np.array([[x],[1]]))[0] - t)**2
            new_dE1, new_dE2 = grad(np.array([[x],[1]]),t)      
            dE1 += new_dE1 / float(training_data.shape[0])
            dE2 += new_dE2.T / float(training_data.shape[0])
        error.append(err / float(training_data.shape[0]))

        w1 = w1 - step_size * dE1
        w2 = w2 - step_size * dE2

    return (NN_before, neural_network(w1,w2,h), error)


def verify_gradient(M,e,number_test_patterns):
    f = open('data/sincTrain25.dt')
    data = np.loadtxt(f)

    # We use the first training pattern to verify our gradient
    for x,t in data[0:number_test_patterns]:
        x = np.array([[x],[1]])

        h = transfer_function
        dh = transfer_function_derivative
        E = error_function

        w1, w2 = create_weights(M)

        # Compute the difference between backpropagation and
        # finite differences for changing the layer 1 weights
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

                w1_diff[i,j] += abs(fin_diff - dE[i,j]) / 5.0

        # Compute the difference between backpropagation and
        # finite differences for changing the layer 2 weights
        w2_diff = np.zeros(w2.shape)
        for i in range(0,M+1):
            w2_e = np.copy(w2)
            w2_e[0,i] += e

            NN = neural_network(w1,w2,h)
            NN_e = neural_network(w1,w2_e,h)
            fin_diff = (E(NN_e(x)[0],t) - E(NN(x)[0],t)) / e

            grad = error_backpropagation(w1,w2,h,dh)
            _, dE = grad(x,t)

            w2_diff[0,i] += abs(fin_diff - dE[i]) / 5.0

    # We don't return the partial derivative for the bias to the output
    return (w1_diff, w2_diff[[0],0:20])

def create_plots(iterations, data, M, step_size):
    NN_b, NN_a, E = steepest_decent_training(iterations, data, M, step_size)

    X = data[:,0]
    X_cont = np.arange(-10,10,0.1)
    
    t = data[:,1]
    Y = [NN_a(np.array([[x],[1]]))[0] for x in X_cont]
    f = np.vectorize(lambda(x): sin(x) / x)

    plt.figure(1)
    plt.plot(X_cont,Y,label='Neural network \n (M = %d)' % M)
    plt.plot(X_cont,f(X_cont),label='sinc(x)')
    plt.legend()
    plt.savefig('images/nn_vs_real_%d_%d.%s' % (iterations,M,img_format), format=img_format)

    plt.figure(2)
    plt.plot(X_cont,Y,label='Neural network \n (M = %d)' % M)
    plt.scatter(X,t,color='red',label='Training data')
    plt.legend()
    plt.savefig('images/nn_vs_training_%d_%d.%s' % (iterations,M,img_format), format=img_format)

    plt.figure(3)
    plt.plot(E, label='Error (M = %d)' % M)
    plt.yscale('log')
    plt.legend()
    plt.savefig('images/error_%d_%d.%s' % (iterations,M,img_format), format=img_format)
    
    plt.show()
    

if __name__ == "__main__":
    """
    d1,d2 = verify_gradient(20,10**(-6),5)
    print np.sum(d1)/20 + np.sum(d2)/20
    """
    low = 0.001
    med = 0.1
    high = 1
    high_ = 0.2
    
    f = open('data/sincTrain25.dt')
    data = np.loadtxt(f)
    create_plots(10000, data, 20, high_)
    
    
