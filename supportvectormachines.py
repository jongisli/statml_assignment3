#!/usr/bin/python
import sys
import numpy as np
import scale as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('/Users/jongisli/Downloads/libsvm-3.16/python')
from svmutil import *

def svm_model(trainfile, testfile):
    data = sc.get_data(trainfile)
    x = data[:,0:22].tolist()
    y = data[:,-1].tolist()

    dataTest = sc.get_data(testfile)
    xTest = dataTest[:,0:22].tolist()
    yTest = dataTest[:,-1].tolist()
    
    C = [10**n for n in range(-4,3)]
    gamma = [10**n for n in range(-6,1)]
    
    prob = svm_problem(y,x)
    accs = []
    Acc = np.zeros((len(C),len(gamma)))
    for c in C:
        for g in gamma:
            print "(c,g)"
            print (c,g)
            
            param = svm_parameter('-s 0 -t 2 -c %f -g %f' % (c,g))
            m = svm_train(prob, param)
            p_labels, p_acc, p_vals = svm_predict(yTest, xTest, m)

            print "Acc"
            print p_acc
            accs.append(p_acc)

    CC, Gamma = np.meshgrid(C,gamma)

    return accs

if __name__ == "__main__":
    accs = svm_model('data/parkinsonsTrainStatML.dt.scaled', 'data/parkinsonsTestStatML.dt.scaled')
    print max(accs)

    """
    def svm_model(trainfile, testfile):
    dataTrain = sc.get_data(trainfile)
    xTrain = dataTrain[:,0:22].tolist()
    yTrain = dataTrain[:,-1].tolist()

    dataTest = sc.get_data(testfile)
    xTest = dataTest[:,0:22].tolist()
    yTest = dataTest[:,-1].tolist()
    
    c = [10**n for n in range(-4,3)]
    gamma = [10**n for n in range(-6,1)]
    prob = svm_problem(yTrain,xTrain)
    acc = get_accuracy(prob, xTest, yTest)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    C, Gamma = np.meshgrid(c,gamma)
    Acc
    Acc = [accV(C,Gamma)

    ax.plot_surface(C,Gamma, Acc)
    plt.show()
    """
