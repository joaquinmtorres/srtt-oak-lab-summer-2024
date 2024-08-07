# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:25:37 2024

@author: joaquinmtorres

Script to check older adults srtt key press data
"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from os.path import dirname

# Helper Functions
## Average calculator
### from: https://www.geeksforgeeks.org/find-average-list-python/
def average(lst): 
    return sum(lst) / len(lst) 
## Breaks list into chunks of size n
### from: https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
def divide_chunks(l, n): 
    # looping till length l 
    for x in range(0, len(l), n):  
        yield l[x:x + n] 
        
# Define file path
fileLoc = '/Users/joaqu/OneDrive/Documents/Bates/Kim Lab/dataFiles/20240805 OA/data/'
saveLoc = dirname(dirname(fileLoc)) # Files are in a folder named 'data', and we want everything to be saved at the directory before this folder
dirList = sorted(glob.glob(fileLoc + '/*'))

# Get sample file to take key press stimuli sequence
sampleFile = pd.read_csv(os.path.join(fileLoc + 'srttOA01.csv'))
stimuli = sampleFile['empty_column'].iloc[:-1] # omits final unnecessary row
stimuli = stimuli[-420:].reset_index(drop=True).replace('None', np.nan) # omits practice trial rows, then resets index and replaces 'None' objects with nan
stimuli = stimuli[:-60].tolist() # Omit last 5 test trials and convert to list
stimuli = [str(i) for i in stimuli] # Turn each item into a string

# Set up dataframe
trials = np.arange(1, 36) # Trial numbers
allKeyPresses = pd.DataFrame() # df will have sequences as first column which each OA dataset will be compared to

# Loop through each data file to make Key Press Responses dataframe
for file in dirList:
    df = pd.read_csv(file)
    
    # Set file names and save directories
    fileName = file.split('/')[-1:][0].split('\\')[1].split('.')[0]
    
    # Get participant responses
    responses = df['response'].iloc[:-1] # omits final unnecessary row
    responses = responses[-420:].reset_index(drop=True).replace('None', np.nan) # omits practice trial rows, then resets index and replaces 'None' objects with nan
    responses = responses[:-60] # omit last 5 test trials
    
    # Append to dataframe
    allKeyPresses[fileName] = responses
    
# Create bins of three trials (36 key presses each bin)
arrCounts = [] # Set empty array where the number of correct responses for each key press will be appended to
for index, row in allKeyPresses.iterrows():
    x = row.tolist() # make each row (key press) a list
    countVal = x.count(stimuli[index]) # count how many correct key presses there are (i.e. stimuli[index] corresponds to the stimulus to compare to)
    arrCounts.append(countVal)
bins = list(divide_chunks(arrCounts, 36)) # Divide arrCounts by 3 trial sequences (36 key presses)
sumBins = [sum(k) for k in bins] # Take the sums of each bin (number of correct key responses per bin)
probBins = [l/(36*len(allKeyPresses.columns)) for l in sumBins] # Get probability of correct answers per bin - z=number of correct responses per bin; 36 key presses in a bin*number of participants=total in one bin

# Plot data
plt.figure()
plt.plot(np.arange(1,len(bins)+1), probBins)
plt.xlabel('Bin #')
plt.xticks(np.arange(1,len(bins)+1))
plt.ylabel('Probability correct')
plt.yticks([0,0.5,1.0])
figBins = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figBins.savefig(saveLoc + '/binsCorrectRate.png', bbox_inches='tight')