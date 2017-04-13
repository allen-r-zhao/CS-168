# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:01:48 2017

@author: AZ
"""

import matplotlib.pyplot as plt
import numpy as np
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

		ax.set_xticklabels(range(1, 21))
		ax.set_yticklabels(names)

		plt.tight_layout()

		plt.savefig(outputFileName, format = 'png')
		plt.close()
        
# Handles importing the provided datasets
def importFiles():
    with open('data50.csv', 'rb') as a:
        reader1 = csv.reader(a)
        rowNum = 0
        for row1 in reader1:
            main[rowNum,0] = row1[0]
            main[rowNum,1] = row1[1]
            main[rowNum,2] = row1[2]
            rowNum += 1
    with open('groups.csv', 'rb') as b:
        reader2 = csv.reader(b)
        rowNum = 0
        for row2 in reader2:
            groups[rowNum] = row2[:]
            rowNum += 1
    with open('label.csv', 'rb') as c:
        reader3 = csv.reader(c)
        rowNum = 0
        for row3 in reader3:
            labels[rowNum] = row3[0]
            rowNum += 1

# Subroutine for Jaccard similarity; compares 2 articles
# Inputs: article1, article2 are integer inputs (such as 1 or 3)
def jaccSubroutine(article1, article2):
    set1 = main[main[:,0] == article1, 1:3]  # Set of article1 rows; cuts redundant first column
    set2 = main[main[:,0] == article2, 1:3]  # Set of article2 rows; cuts redundant first column
    num = 0.0   # Numerator
    den = 0.0   # Denominator
    for i in range(len(set1[:,0])):
        unionFreq = set2[set2[:,0] == set1[i,0],1]  # Frequency of the matching word in article2
        if len(unionFreq) > 0:
            num += min(set1[i,1], unionFreq[0])
            den += max(set1[i,1], unionFreq[0])
        else:
            den += set1[i,1]
    for j in range(len(set2[:,0])):
        if len(set1[set1[:,0] == set2[j,0],1]) == 0:
            den += set2[j,1]
    if num > 0 and den > 0:
        return (float(num)/den)   # Casts num into a float to make the result a float
    else:
        return 0.0


def L2Subroutine(article1, article2):
    set1 = main[main[:,0] == article1, 1:3]  # Set of article1 rows; cuts redundant first column
    set2 = main[main[:,0] == article2, 1:3]  # Set of article2 rows; cuts redundant first column
    diffTot = 0.0
    for i in range(len(set1[:,0])): #Compares all words in article1 with appropriate word in article2, if present
        freq1 = set1[i, 1]
        freq2 = set2[set2[:,0] == set1[i,0], 1]
        if not freq2:
            freq2 = 0
        else:
            freq2 = freq2[0]
        diffTot += (freq1 - freq2)**2
        set1[i, 1] = -1 #indicates in set2 that this word has already been compared
    
    for i in range(len(set2[:, 0])): #iterates over all words in article2 that are not in article1
        freq1 = set1[set1[:,0] == set2[i,0],1]
        freq2 = set2[i, 1]
        if (freq1 != -1):
            diffTot += set2[i, 1] ** 2
    return (diffTot ** (0.5)) * -1.0 #returns negation of sqrt of diffTotal
    
def cosineSubroutine(article1, article2):
    num = 0.0
    den1 = 0.0 
    den2 = 0.0 
    set1 = main[main[:,0] == article1, 1:3]  # Set of article1 rows; cuts redundant first column
    set2 = main[main[:,0] == article2, 1:3]  # Set of article2 rows; cuts redundant first column
    #Compares all words in article1 with appropriate word in article2
    #Note that unline for L2, for cosine whenever a word is present in only one of two articles 
    #this does not affect the similarity measure and can therefore be ignored
    for i in range(len(set1[:,0])):
        freq1 =  set1[i, 1]
        freq2 = set2[set2[:,0] == set1[i,0], 1]
        if not freq2:
            freq2 = 0
        else:
            freq2 = freq2[0]
        num += freq1 * freq2
        den1 += freq1**2
        den2 += freq2**2
    
    if ((den1 == 0.0) | (den2 == 0.0) | (num == 0.0)):
        return 0
    else:
        return (num / (den1 * den2))
 
# Function that calls jaccSubroutine many times for each possible pair of articles in groups
def jaccMaster():
    # Iterate over all pairs of articles between groups
    # Plug data as we go into dataJacc array
    cnt = 0
    for i in range(1000):
        for j in range(1000):
            temp = jaccSubroutine(i,j)
            dataJacc[int(labels[i])-1, int(labels[j])-1] += temp
            cnt += 1
            if cnt > 5000:
                print i,j
                cnt = 0
    np.divide(dataJacc,divArr)
    makeHeatMap(dataJacc, groups, 'Blues', 'jaccMap.png')

def L2Master():
    for i in range(1000):
        for j in range(1000):
            temp = L2Subroutine(i,j)
            temp /= 50
            dataL2[int(labels[i])-1, int(labels[j])-1] += temp
    #print(dataCos)
    # np.divide(dataCos, (50.0 ** 3))
    makeHeatMap(dataL2, groups, 'Blues', 'cosMap.png')

def cosineMaster():
    for i in range(1000):
        for j in range(1000):
            temp = cosineSubroutine(i,j)
            temp /= 50
            dataCos[int(labels[i])-1, int(labels[j])-1] += temp
    #print(dataCos)
    # np.divide(dataCos, (50.0 ** 3))
    makeHeatMap(dataCos, groups, 'Blues', 'cosMap.png')

# Returns the category number of the article with the cosine similarity from the input article
# Input: article number
def cosineNN(article):
    curMax = cosineSubroutine(article, 1)
    curNNGroup = int ( labels[0])
    for i in range (2, 1000):
        cosineSim = cosineSubroutine(article, i)
        if (cosineSim > curMax):
            curMax = cosineSim
            curNNGroup = int( labels[i-1])
    return curNNGroup

# Iterates over all articles and increments the value of the cosineNN in a table
# Prints the table as a heatmap
def baselineCosineNN():
    baselineCosineNN = np.zeros(shape=(20,20)) #table for cosine NN heatmap
    errorCounter = 0
    for i in range (1, 1000):
        NN = cosineNN(i)
        ownLabel = int (labels[i])
        if (ownLabel != NN):
            errorCounter += 1
        baselineCosineNN[ownLabel - 1, NN - 1]  += 1                
    makeHeatMap(baselineCosineNN, groups, 'Blues', 'baselineCosineNN.png')
    errorCounter /= 1000
    print("Average classification error: ") 
    print(errorCounter)

def dimensionReduction():
    dList = [10, 25, 50, 100]
    for i in dList:
        dimArray = np.zeros(shape=(129532,i),dtype=float)  # Main data table
        for j in dimArray:
            for k in j:
                j[k] = np.random.normal(0, 1)
                
        print(dimArray)
                           

    
global main
global groups
global labels
global dataJacc
global dataL2
global Cos
global divArr

main = np.zeros(shape=(129532,3),dtype=int)  # Main data table
groups = ['0'] * 20    # List of group names
labels = [0] * 1000    # List of labels for the articles
importFiles()
dataJacc = np.zeros(shape=(20,20))   # Where the data for Jaccard heatmap will go
dataL2 = np.zeros(shape=(20,20))   # Where the data for L2 heatmap will go
dataCos = np.zeros(shape=(20,20))   # Where the data for Cosine heatmap will go
divArr = np.full((20,20),2500.)

dimensionReduction()
#jaccMaster()
#L2Master()
#cosineMaster()
#baselineCosineNN()