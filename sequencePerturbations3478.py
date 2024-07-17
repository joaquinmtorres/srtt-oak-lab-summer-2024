# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:45:26 2024

@author: joaqu

Based on trainingSequencePerturbations, this script takes a new sequence of 
size 8 (using 3, 4, 7, and 8 keys), and randomly assigning one perturbation 
for each of 15 randomly assorted sequences.
"""

# import libraries.
import random
import pandas as pd
import numpy as np
        
# define variables
indices = np.arange(0, 8) # to choose one of 15 items in a trainingSeq
trainingDF = [] # empty list to start

# for loop that changes an item in the training sequence to a different number and adds the new sequence to a data frame
for i in indices:
    trainingSeq = [3, 4, 8, 7, 8, 3, 7, 4] # training sequence used
    availNums = [3, 4, 7, 8] # numbers that are used in the experiment
    newList = trainingSeq # reset newList, which should be back to trainingSeq
    newAvailNums = availNums # reset newAvailNums, which should be back to availNums
    newNum = random.choice(list(availNums)) # randomly chooses a number from availNums    
    if newNum != newList[i]:
        newList[i] = newNum # replaces number in position x with newNum
        trainingDF.append(newList) # adds newList array to trainingDF data frame
    else: # if number is being replaced with the same number
        newAvailNums = availNums # resets newAvailNums
        newAvailNums.remove(int(newNum)) # excludes newNum from availNums to prevent repetition
        newNum = random.choice(list(newAvailNums)) # reassigns newNum to a different randomly chosen number
        newList[i] = newNum # replaces number in position x with newNum
        trainingDF.append(newList) # adds newList array to trainingDF data frame
        
# Shuffle rows of trainingDF        
trainingDF = pd.DataFrame(trainingDF)
trainingDF = trainingDF.sample(frac=1)
trainingDF = trainingDF.transpose() # transposes rows into columns

# Save data frame as .csv
data = np.array(trainingDF)
df = pd.DataFrame(data)
filePath = input('Enter path where you would like to save this file: ')
fileName = input('What would you like to name your file? (include .csv at the end): ')
df.to_csv(filePath + '/' + fileName, index=False, header=False)
