#!/usr/bin/python
import numpy as np


def get_data(datafile):
    f = open(datafile)
    data = np.loadtxt(f)
    f.close()
    return data


def mean_and_sigma(data):
    mu = np.mean(data,axis=0)
    sigma  = np.var(data,axis=0)**0.5
    return mu,sigma


def normalize_data_function(mu,sigma):
    def func(data):
        return (data - mu) / sigma
    return func


def scale_data(datafiles, f_norm):
    for datafile in datafiles:
        data = get_data(datafile)
        labels = data[:,[-1]]
        dataN = f_norm(data[:,0:-1])        
        new_data = np.hstack((dataN, labels))
        np.savetxt(datafile + '.scaled', new_data)


def pretty_print(arr):
    print r'\begin{array}{l}'
    for a in arr:
        print r'%.5f \\' % a
    print r'\end{array}'

if __name__ == "__main__":
    """
    data = get_data('data/parkinsonsTrainStatML.dt')[:,0:-1]    
    mu, sigma = mean_and_sigma(data)
    f_norm = normalize_data_function(mu,sigma)

    scale_data(['data/parkinsonsTrainStatML.dt','data/parkinsonsTestStatML.dt'], f_norm)
    """
    training_data = get_data('data/parkinsonsTrainStatML.dt')
    scaled_test_data = get_data('data/parkinsonsTestStatML.dt.scaled')

    training_mu, training_sigma = mean_and_sigma(training_data)
    test_scaled_mu, test_scaled_sigma = mean_and_sigma(scaled_test_data)

    #pretty_print(training_mu)
    #pretty_print(training_sigma)
    #pretty_print(test_scaled_mu)
    #pretty_print(test_scaled_sigma)

    scaled_training_data = get_data('data/parkinsonsTrainStatML.dt.scaled')
    scaled_training_mu, scaled_training_sigma = mean_and_sigma(scaled_training_data)
    pretty_print(scaled_training_mu)
    pretty_print(scaled_training_sigma)
