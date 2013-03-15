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

def meanVar(data):
    mu = np.mean(data,axis=0)
    v  = np.var(data,axis=0)
    return mu,v

def fNorm(data,mu,v):
    return (data-mu)/v


data = get_data('data/parkinsonsTrainStatML.dt')
(mu,v) = meanVar(data)

print "mean training: "
print mu
print "variance training: "
print v

col1 = data[:,0]

col1m = np.mean(col1)
print "col1 mean:"
print col1m

col1_ = col1-col1m

print "col1_ mean:"
print np.mean(col1_)

dataM = fNorm(data,mu,v)
(muM,vM) = meanVar(dataM)

print "mean normalized: "
print muM
print "variance normalized: "
print vM

dataTest = get_data('data/parkinsonsTestStatML.dt')
(muTest,vTest) = meanVar(fNorm(dataTest,mu,v))

print "mean Test: "
print muTest
print "var Test: "
print vTest
