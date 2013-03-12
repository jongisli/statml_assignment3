#!/usr/bin/python
import numpy as np

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


def error_backpropagation(w1,w2,h,dh,x,t):
    NN = neural_network(w1,w2,h)
    h = np.vectorize(h)
    dh = np.vectorize(dh)

    # Forward-propagation
    y,a = NN(x)
    z = h(np.append(a,1))
    # TODO: Check if appending 1 is correct interpretation of second layer bias

    # Back-propagation
    d_k = y - t
    d_j = dh(a)*np.sum(w2*d_k)

    # First and second layer derivatives
    dE1 = d_j*x.T
    dE2 = d_k*z

    return (dE1, dE2)

if __name__ == "__main__":
    w1 = np.array([[2,1],
                  [1,1],
                  [0,1]])
    w2 = np.array([[2,1,3,1]])

    x = np.array([[5.94911],[1]])
    t = -0.108044

    h = transfer_function
    dh = transfer_function_derivative

    NN = neural_network(w1,w2,h)

    e1, e2 = error_backpropagation(w1,w2,h,dh,x,t)
    print e1
    print e1.shape
    print e2
    print e2.shape
    
    

