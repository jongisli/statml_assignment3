#!/usr/bin/python
import numpy as np

#Pre: datafile points to an
#     existant file with no
#     restrictions.
#Ret: The data inside the
#     datafile
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


if __name__ == "__main__":
    data = get_data('data/parkinsonsTrainStatML.dt')[:,0:-1]    
    mu, sigma = mean_and_sigma(data)
    f_norm = normalize_data_function(mu,sigma)

    scale_data(['data/parkinsonsTrainStatML.dt','data/parkinsonsTestStatML.dt'], f_norm)
