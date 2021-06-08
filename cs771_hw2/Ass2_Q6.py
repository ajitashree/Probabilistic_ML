# -*- coding: utf-8 -*-
"""
Spyder Editor
@author: Ajita Shree, Roll no: 20111262, 
1st year PhD, CS771, Intro to Machine Learning
"""

'''
1) Below code will do generative classification and can run for multiple features at the time i.e. D >0
2) Covaraince matrixis capable of handling any shape of gaussian 
i.e. all entires non-zero: Line 50-57 can be uncommented to visualize this
3) The Code can be easily scaled up for multiple classes at a time.
4) TODO: Give input files in the main function, D initialization and flow starts from the main function below.
'''


import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 


# Calculate probability value for a given feature vector for a given distribution
def density(feature, mu, cov):
    
    covInv = np.linalg.inv(cov)
    dTerm = 1/(2 * np.pi * np.sqrt(np.linalg.det(cov)))
    eTerm = np.matmul(np.matmul(feature - mu, covInv), (feature - mu).T).item(0)
    
    prob = dTerm * np.exp(-0.5 * eTerm)
    return prob
    
# Calculate parameters of Gaussian i.e. Mean, Sigma
def distribution(data, D):
    
    mu = []
    cov = np.zeros((D, D))
    
    # Mean
    for d in range(D):
        mu += [np.mean(data['x' + str(d)])]
        
    # Covariance Matrix 
    for d in range(D):
        
        x = 'x' + str(d)
        cov[d][d] = np.mean(data[x] * data[x]) - pow(mu[d], 2)
        
        # Comment for : Sigma with only diagonal entries 
        '''
        for d_ in range(d + 1, D):
            
            x_ = 'x' + str(d_)
            var = np.mean(data[x] * data[x_]) - mu[d]*mu[d_]
            cov[d][d_] = var
            cov[d_][d] = var
        '''
    return mu, cov

# Prediction of class for all training points: To update for handling multiple classes
def decisionBoundary(df_, D, part):
    
    condP1 = []
    condP2 = []
    label = []
    
    d1 = df_[df_['label'] == 1].copy(deep = True)
    d2 = df_[df_['label'] == -1].copy(deep = True)
    
    l = len(df_)

    p1 = len(d1)/l
    p2 = len(d2)/l
    
    mu1, cov1 = distribution(d1, D)
    mu2, cov2 = distribution(d2, D)
    mu, cov = distribution(df_, D)

    df = df_[df_.columns[:-1]]
    
    for index, row in df.iterrows():
        
        # Part b) cov1, cov2 to be updated with cov
        if (part == 'b'):
            cp1 = p1 * density(np.matrix(row), mu1, cov)
            cp2 = p2 * density(np.matrix(row), mu2, cov)
        else:
            cp1 = p1 * density(np.matrix(row), mu1, cov1)
            cp2 = p2 * density(np.matrix(row), mu2, cov2)            
        
        condP1 += [cp1]
        condP2 += [cp2]
        
        label += [1 if cp1 >= cp2 else -1]
        
    df['pred'] = label
    df['act'] = df_['label']
    print (part + 'Accuracy', sum(label == df['act'])/len(label))
    df['condP1'] = condP1
    df['condP2'] = condP2
        
    return df, mu1, cov1, mu2, cov2, cov
    
# Contout plotting: plot_contours function Credits: Bob Trenwith, Youtube    
def plot_contours(mu, cov):
    
    res = 200 # resolution
    x1g = np.linspace(-10, 40, res)
    x2g = np.linspace(-10, 40, res)
        
    col = ['black', 'green', 'orange']
    rv = multivariate_normal(mean = mu, cov = cov)
    z = np.zeros((len(x1g),len(x2g)))
    for i in range(0,len(x1g)):
        for j in range(0,len(x2g)):
            z[j,i] = rv.logpdf([x1g[i], x2g[j]])
    
    sign, logdet = np.linalg.slogdet(cov)
    normalizer = -0.5 * (2 * np.log(6.28) + sign * logdet)
    
    for offset in range(1,4):
        plt.contour(x1g,x2g,z, levels=[normalizer - offset], colors=col, linewidths=2.0, linestyles='solid')


# Plotting the data: Only 2 dimensional
def plot(tag, df, mu1, cov1, mu2, cov2, cov, part):

    d1 = df[df[tag] == 1].copy(deep = True)
    d2 = df[df[tag] == -1].copy(deep = True)
    
    plt.grid()
    #plt.xlim(0, 40) # limit along x1-axis
    #plt.ylim(0, 40) # limit along x2-axis
    
    plt.plot(d1['x0'], d1['x1'], marker = '.', ls = 'None', c = 'r')
    plt.plot(d2['x0'], d2['x1'], marker = '.', ls = 'None', c = 'b')  
    
    if (part == 'b'):
        plot_contours(mu1, cov)
        plot_contours(mu2, cov)
    else:
        plot_contours(mu1, cov1)
        plot_contours(mu2, cov2)        
        
    plt.xlabel('x0', fontsize = 14, color = 'black')
    plt.ylabel('x1', fontsize = 14, color = 'black')
    plt.title(part + ') Generative Modelling', fontsize = 14, color = 'black')
    plt.show()
  
  
# Part c of question
def SVMclassification(df):
    

    X = df[['x0', 'x1']]
    Y = df['label']
    clf = svm.LinearSVC(C = 1, max_iter = 5000).fit(X, Y)
    Y_pred = clf.predict(X)
    df['svm_pred'] = Y_pred
    
    print ('SVM Accuracy', sum(Y_pred == Y)/len(Y))
    d1 = df[df['svm_pred'] == 1].copy(deep = True)
    d2 = df[df['svm_pred'] == -1].copy(deep = True)
    
    plt.grid()    
    plt.plot(d1['x0'], d1['x1'], marker = '+', ls = 'None', c = 'r')
    plt.plot(d2['x0'], d2['x1'], marker = '.', ls = 'None', c = 'b')  
    
    xx, yy = np.meshgrid(np.arange(-30, 40, .1),   np.arange(-30, 40, .1))
    Z = clf.predict(np.c_[xx.ravel(),  yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)
    plt.title('Support Vector Machine Decision Boundary')
    plt.show()
    
    
if __name__ == "__main__":
    
    D = 2
    # Readin Dataset; Renaming column labels
    data = pd.read_csv('binclass.txt', header = None) #'binclassv2.txt'
    cols = ['x' + str(i) for i in range(D)]
    data.columns = cols + ['label']
    
    # Part A: Sigma+ and Sigma-
    df, mu1, cov1, mu2, cov2, cov = decisionBoundary(data, D, 'a')
    #plot('act', df, mu1, cov1, mu2, cov2, cov, 'a') # Plot actual labels 
    plot('pred', df, mu1, cov1, mu2, cov2, cov, 'a') # Plot predicted labels based on generative classification
    
    # Part B: Same Sigma for both classes {+1, -1} 
    df, mu1, cov1, mu2, cov2, cov = decisionBoundary(data, D, 'b')
    #plot('act', df, mu1, cov1, mu2, cov2, cov, 'b') # Plot actual labels 
    plot('pred', df, mu1, cov1, mu2, cov2, cov, 'b') # Plot predicted labels based on generative classification
    
    # Part: SVM clasifier
    '''
    data = pd.read_csv('binclass_libSVM.txt', header = None) #'binclass_libSVMv2.txt'
    data['index'] = data.index
    data['x0'] = data['index'].apply(lambda x: float(data[0][x].split(' ')[1].split(':')[1]))
    data['x1'] = data['index'].apply(lambda x: float(data[0][x].split(' ')[2].split(':')[1]))
    data['label'] = data['index'].apply(lambda x: int(data[0][x].split(' ')[0]))
    '''
    SVMclassification(data)



    
    
    