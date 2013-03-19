#!/usr/bin/python
import sys
import numpy as np
import scale as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append('/Users/jongisli/Downloads/libsvm-3.16/python')
from svmutil import *

img_format = 'png'

def grid_search(dataTrain, dataTest):
    #dataTrain = sc.get_data(trainfile)
    x = dataTrain[:,0:22].tolist()
    y = dataTrain[:,-1].tolist()

    #dataTest = sc.get_data(testfile)
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

    return Acc, (optim_C, optim_gamma, max_acc)

def plot_accuracies(acc_matrix,i,j):
    C_min, C_max = -2,8
    gamma_min, gamma_max = -6,4

    # Trick to get rid of white border around the plot
    _, axes = plt.subplots()    
    img = axes.imshow(acc_matrix,extent=[gamma_min,gamma_max,C_max,C_min]
                     ,interpolation='nearest', cmap=cm.Reds)
    axes.autoscale(False)
    
    axes.scatter(j - 5.5, i - 1.5 , s=50,
                 c='green', marker='x', linewidth=2)
    plt.colorbar(img)
    plt.savefig('images/accuracies.%s' % img_format,
                format=img_format,
                bbox_inches='tight')
   
    
def cross_validate(data, number_of_splits):
    splits = np.array_split(data,number_of_splits)
    mean_accs = np.zeros((10,10))
    for i in range(number_of_splits):
        test_split = splits[i]
        train_splits = splits[i-1:i] + splits[i+1:]
        acc_matrix, _ = grid_search(np.vstack(train_splits), test_split)
        mean_accs += acc_matrix / number_of_splits

    i,j = np.unravel_index(np.argmax(mean_accs),(10,10))
    return mean_accs, (i,j)

def best_C_and_gamma(i,j):
    C = 10**(i - 2)
    gamma = 10**(j - 6)
    return C,gamma


def accuracy_of_model(C, gamma, dataTrain, dataTest):
    x = dataTrain[:,0:22].tolist()
    y = dataTrain[:,-1].tolist()
    xTest = dataTest[:,0:22].tolist()
    yTest = dataTest[:,-1].tolist()
    
    prob = svm_problem(y,x)
    param = svm_parameter('-s 0 -t 2 -c %f -g %f' % (C,gamma))
    m = svm_train(prob, param)
    p_labels, p_acc, p_vals = svm_predict(yTest, xTest, m)
    return p_acc


if __name__ == "__main__":
    """
    acc_matrix, opt_params = grid_search('data/parkinsonsTrainStatML.dt.scaled',
                                         'data/parkinsonsTestStatML.dt.scaled')
    print "optimal hyperparameters"
    print opt_params
    plot_accuracies(acc_matrix)
    """
    dataTrain = sc.get_data('data/parkinsonsTrainStatML.dt')
    dataTrainScaled = sc.get_data('data/parkinsonsTrainStatML.dt.scaled')
    dataTest = sc.get_data('data/parkinsonsTestStatML.dt')
    
    accmatrix, (i,j) = cross_validate(dataTrainScaled,5)    

    print best_C_and_gamma(i,j)
    #print accuracy_of_model(C, gamma, dataTrain, dataTest)
    
