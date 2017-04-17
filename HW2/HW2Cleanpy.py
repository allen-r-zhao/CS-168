# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 21:40:48 2017

@author: djhof
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.spatial import distance
from scipy.sparse import csc_matrix
import warnings
import csv


def makeHeatMap(data, names, color, outputFileName):
	#to catch "falling back to Agg" warning
	with warnings.catch_warnings():
   		warnings.simplefilter("ignore")
		#code source: http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
		fig, ax = plt.subplots()
		#create the map w/ color bar legend
		heatmap = ax.pcolor(data, cmap=color)
		cbar = plt.colorbar(heatmap)

		# put the major ticks at the middle of each cell
		ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
		ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

		# want a more natural, table-like display
		ax.invert_yaxis()
		ax.xaxis.tick_top()

		ax.set_xticklabels(names)
		ax.set_yticklabels(names)
		plt.xticks(rotation=90)
		plt.figure(figsize=(23,20))
      
		#plt.tight_layout()
		plt.savefig(outputFileName, format = 'png')
		plt.close()
        
# Constructs a scatterplot from two input arrays (ordered x- and y-coordinates of points)
def scatterPlot(XArray, YArray, XLabel, YLabel, outputFileName):
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.plot(XArray, YArray, 'bo')
    
    plt.savefig(outputFileName, format = 'png')
    plt.close()

#Handles importing data. 
def importFiles2():
    #Constructs the 1000x61067 table where i,j is the frequency of the j'th 
    #word in the i'th article
    with open('data50.csv', 'rb') as a:
        reader1 = csv.reader(a)
        rowNum = 0
        for row1 in reader1:
            article = int(row1[0])
            word = int(row1[1])
            freq = int(row1[2])
            main2[article - 1, word - 1] = freq
    global sparse
    sparse = csc_matrix(main2)
     
    #Constructs the groups-table
    with open('groups.csv', 'rb') as b:
        reader2 = csv.reader(b)
        rowNum = 0
        for row2 in reader2:
            groups[rowNum] = row2[:]
            rowNum += 1
    #Constructs the labels-table
    with open('label.csv', 'rb') as c:
        reader3 = csv.reader(c)
        rowNum = 0
        for row3 in reader3:
            labels[rowNum] = row3[0]
            rowNum += 1
            
# Methods for calculating the cosine-, L2- and Jaccard differences between articles
# output tables such that i,j lists the metric-difference
# between the i'th and j'th article. Makes use of scipy-methods per
# Piazza post 84
def scipyCosine():
    global cosineResult
    cosineResult = scipy.spatial.distance.pdist(main2, 'cosine')
    cosineResult = scipy.spatial.distance.squareform(cosineResult)
    for i in range(1000):
        for j in range(1000):
            dist = 1 - cosineResult[i,j]
            dataCos[int(labels[i])-1, int(labels[j])-1] += dist / 2500
    makeHeatMap(dataCos, groups, 'Blues', 'scipyCos3.png')
    
    
def scipyJacc():
    result = scipy.spatial.distance.pdist(main2, 'jaccard')
    result = scipy.spatial.distance.squareform(result)
    for i in range(1000):
        for j in range(1000):
            dist = (1 - result[i,j]) /4
            dataJacc[int(labels[i])-1, int(labels[j])-1] += dist / 2500
    makeHeatMap(dataJacc, groups, 'Blues', 'scipy2Jaccard.png')

def scipyL2():
    importFiles2()
    result = scipy.spatial.distance.pdist(main2, 'sqeuclidean')
    result = scipy.spatial.distance.squareform(result)
    for i in range(1000):
        for j in range(1000):
            dist =  - 1 * (result[i,j] ** 0.5)
            dataL2[int(labels[i])-1, int(labels[j])-1] += dist / 2500
    makeHeatMap(dataL2, groups, 'Blues', 'scipy2L2.png')


#Method for calculating the nearest neighbours of all articles based on their
#cosine differences. Stores these in a 20x20 table based on the garticle groups
def scipyNN():
    scipyCosine()
    global baselineCosineNN
    global errorCounter
    baselineCosineNN = np.zeros(shape=(20,20)) #table for cosine NN heatmap
    errorCounter = 0.0
    for i in range (1000):
        vector = [0] * 1000
        for j in range (1000):
                if i != j:
                    vector[j] = 1 - cosineResult[i, j]
        groupNN = int (labels[np.argmax(vector)])
        owngroup = int(labels[i])
        if groupNN != owngroup:
            errorCounter += 1
        baselineCosineNN[owngroup - 1, groupNN - 1] += 1
    
# Method for calculating and printing the baseline cosine NN-map
def scipybaselineNNMaster(filename):
    scipyNN()
    makeHeatMap(baselineCosineNN, groups, 'Blues', filename)
    print("Average classification error: " + str(errorCounter/1000)) 


# Method for reducing the main table to dimension d. Subsequently computes and outputs
# the NN-heatmap
def scipyDimensionReduction(d):
    dimArray = randomTable(61067, d)
    global main2
    main2 = sparse.dot(dimArray)
    scipyNN()
    filename = 'scipyDimRed' + str(d) + '.png'
    print filename
    makeHeatMap(baselineCosineNN, groups, 'Blues', filename)
    
# Constructs a row * col matrix whose values are independently drawn from a 
# Gaussian distribution
def randomTable(row, col):
    dimArray = np.zeros(shape=(row, col),dtype=float)  # Main data table
    for i in range (row):
        for j in range (col):
            dimArray[i, j] = np.random.normal(0, 1)  
    return dimArray

# Outer loop for calculating the NN heatmaps for d=10, 25, 50 and 100. 
# Also outputs the average error per dimension
def dimRedMain():
    dim = [10, 25, 50, 100]
    global dimTable
    dimTable = np.zeros(shape=(4,2),dtype=float)
    counter = 0
    for i in dim:
        scipyDimensionReduction(i)
        dimTable[counter][0] = i
        dimTable[counter][1] = errorCounter / 1000
        counter += 1
    print dimTable
    
    
    
# Method for computing the average number of errors and size of S for
# an LSH-scheme of dimension d.      
def LSH(d):
    LSHSetup(d)
    scipyCosine()
    avError = 0.0
    avS = 0.0
    for i in range (1000):
        vecI = sparse.getrow(i).transpose().toarray().flatten()
        S = []
        NNVals = []
        for j in range(L):
            binVal = findHashBucket(vecI, j, d)
            for k in hashBucketList[j][binVal]:
                if ((k != i + 1) & (k not in S)):
                    S.append(k)
                    NNVals.append(cosineResult[i, k-1])
        if not S:
            avError += 1
        else:
            NN = S[np.argmax(NNVals)]
            NNlabel = labels[NN - 1]
            if (NNlabel != labels[i-1]):
                avError += 1
                avS += len(S)
        
    avError /= (1000.0) 
    avS /= (1000.0)
    LSHScatterX.append(avError)
    LSHScatterY.append(avS)
    LSHResultTable[d-5, 0] = d
    LSHResultTable[d-5, 1] = avError      
    LSHResultTable[d-5, 2] = avS

# Outer loop for the LSH-scheme. Constructs the results-table and scatter plot 
# for all 16 dimensions 
def LSHMain():
    importFiles2()
    global LSHResultTable
    LSHResultTable = np.zeros(shape=(16,3),dtype=float)
    scipyCosine()
    for d in range (5, 21):
        LSH(d)
    scatterPlot(LSHScatterX, LSHScatterY, 'Average Error', 'Average size of $S_q$', 'LSHTotal2.png')
    print LSHResultTable
    

            
# constructs a list of L dx61067 hashtables with values drawn randomly from a normal distr.
# with mean 0 and var 1. Populates the hashbuckets accordingly.
def LSHSetup(d):
    global hashTableList
    global hashBucketList
    hashTableList = [0]*L
    hashBucketList = []
    
    for i in range (L):
        hashTableList[i] = scipy.sparse.coo_matrix(randomTable(d, 61067))
        bucketList = []
        for i in range (2 ** d):
            bucketList.append([])                
        hashBucketList.append(bucketList)
    
    for i in range(1000):
        vecI = sparse.getrow(i).transpose().toarray().flatten() #kx1 sparse vector with word frequencies of article                             
        for j in range(L):  
            binVal = findHashBucket(vecI, j, d)
            hashBucketList[j][binVal].append(i + 1)
    
# For a given vector, hashfunction and dimension returns the index of the
# appropriate bucket
def findHashBucket(vector, function, d):
    prodVec = scipy.sparse.coo_matrix.dot(hashTableList[function], vector)
    binVec = [0] * d
    if (len(prodVec) > d):
        print len(prodVec)
        sys.exit()
    for i in range (d):
        if prodVec[i] <= 0:
            binVec[i] = 0
        else:
            binVec[i] = 1
    binVal = ""
    for k in binVec:
        binVal += str(k)
    return int(binVal, 2)
    
# Variables
global main
global main2
global sparse
global groups
global labels
global dataJacc
global dataL2
global Cos
global divArr
global L
global hashBucketList
global LSHScatterX
global LSHScatterY
global dCounter

main = np.zeros(shape=(129532,3),dtype=float)  # Main data table
main2 = np.zeros(shape=(1000,61067),dtype=float)  # Main data table
groups = ['0'] * 20    # List of group names
labels = [0] * 1000    # List of labels for the articles
dataJacc = np.zeros(shape=(20,20))   # Where the data for Jaccard heatmap will go
dataL2 = np.zeros(shape=(20,20))   # Where the data for L2 heatmap will go
dataCos = np.zeros(shape=(20,20))   # Where the data for Cosine heatmap will go
divArr = np.full((20,20),2500.) 
L = 128

LSHScatterX = []
LSHScatterY= []
dCounter = 0
importFiles2()

# Executables

#scipyCosine()
#scipyJacc()
#scipyL2()
#scipybaselineNNMaster('scipyNN.png')
#dimRedMain()
#LSHMain()
