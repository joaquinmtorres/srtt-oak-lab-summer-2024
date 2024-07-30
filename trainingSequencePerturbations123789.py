# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:42:28 2024

@author: joaquinmtorres

A script taking the training sequence from the Sequence Experiment and randomly
assigning one perturbation, and creating 12 unique sequences randomly organized
in a data frame.
"""
# import libraries
import random
import pandas as pd
import numpy as np
        
# define variables
indices = [0,1,2,3,4,5,6,7,8,9,10,11] # to choose one of twelve items in a trainingSeq
trainingDF = [] # empty list to start

# for loop that changes an item in the training sequence to a different number and adds the new sequence to a data frame
for x in indices:
    trainingSeq = [2, 9, 3, 1, 8, 7, 9, 3, 2, 8, 7, 1] # training sequence used
    availNums = [1, 2, 3, 7, 8, 9] # numbers that are used in the experiment
    newList = trainingSeq # reset newList, which should be back to trainingSeq
    newAvailNums = availNums # reset newAvailNums, which should be back to availNums
    newNum = random.choice(list(availNums)) # randomly chooses a number from availNums    
    if newNum != newList[x]:
        newList[x] = newNum # replaces number in position x with newNum
        trainingDF.append(newList) # adds newList array to trainingDF data frame
    else: # if number is being replaced with the same number
        newAvailNums = availNums # resets newAvailNums
        newAvailNums.remove(int(newNum)) # excludes newNum from availNums to prevent repetition
        newNum = random.choice(list(newAvailNums)) # reassigns newNum to a different randomly chosen number
        newList[x] = newNum # replaces number in position x with newNum
        trainingDF.append(newList) # adds newList array to trainingDF data frame
        
# Shuffle rows of trainingDF        
trainingDF = pd.DataFrame(trainingDF)
trainingDF = trainingDF.sample(frac=1)
trainingDF = trainingDF.transpose() # transposes rows into columns


# Save data frame as .csv
data = np.array(trainingDF)
df = pd.DataFrame(data)
filePath = input('Enter path where you would like to save this file: ')
df.to_csv(filePath, index=False)
