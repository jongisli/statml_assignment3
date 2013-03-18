#!/usr/bin/python
import sys
import numpy as np
import scale as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append('/Users/jongisli/Downloads/libsvm-3.16/python')
from svmutil import *

def grid_search(trainfile, testfile):
    data = sc.get_data(trainfile)
    x = data[:,0:22].tolist()
    y = data[:,-1].tolist()

    dataTest = sc.get_data(testfile)
    xTest = dataTest[:,0:22].tolist()
    yTest = dataTest[:,-1].tolist()

    C_min, C_max = -2,8
    gamma_min, gamma_max = -6,4
    C = [10**n for n in range(C_min,C_max)]
    gamma = [10**n for n in range(gamma_min,gamma_max)]
    
    prob = svm_problem(y,x)
    accs = []
    Acc = np.zeros((len(C),len(gamma)))
    max_acc = 0
    optim_C, optim_gamma = 0,0
    for i,c in enumerate(C):
        for j,g in enumerate(gamma):
            param = svm_parameter('-s 0 -t 2 -c %f -g %f -q' % (c,g))
            m = svm_train(prob, param)
            p_labels, p_acc, p_vals = svm_predict(yTest, xTest, m)

            if p_acc[0] > max_acc:
                max_acc = p_acc[0]
                optim_C = c
                optim_gamma = g
                
            Acc[i,j] = p_acc[0]
            accs.append(p_acc)

    
    img = plt.imshow(Acc,extent=[gamma_min,gamma_max,C_max,C_min]
                     ,interpolation='nearest', cmap=cm.Reds)
    plt.colorbar(img)
    plt.show()

    return optim_C, optim_gamma, max_acc

if __name__ == "__main__":
    accs = grid_search('data/parkinsonsTrainStatML.dt.scaled', 'data/parkinsonsTestStatML.dt.scaled')
    print "optimal hyperparameters"
    print accs

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
