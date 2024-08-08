# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:25:37 2024

@author: joaquinmtorres

Script to analyze older adults srtt key press and RT data - particularly, 
analysis of: 
    1. probability of invalid responses per bins of three trial sequences
    2. probability of rushed RTs (i.e. RTs < 500 ms)
"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
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
        
# Define file path (change when necessary)
fileLoc = '/Users/joaqu/OneDrive/Documents/Bates/Kim Lab/dataFiles/20240807 Invalid/data/'
saveLoc = dirname(dirname(fileLoc)) # Files are in a folder named 'data', and we want everything to be saved at the directory before this folder
dirList = sorted(glob.glob(fileLoc + '/*'))

# Define valid key presses
valResp = ['', '1', '2', '3', '7', '8', '9']

# Set up dataframes
trials = np.arange(1, 36) # Trial numbers
allKeyPresses = pd.DataFrame() # df will have sequences as first column which each OA dataset will be compared to
rtProbDF = pd.DataFrame(index=trials)

# Loop through each data file to make Key Press Responses dataframe
for file in dirList:
    df = pd.read_csv(file)
    
    # Set file names and save directories
    fileName = file.split('/')[-1:][0].split('\\')[1].split('.')[0]
    
    # Get participant responses
    responses = df['response'].iloc[:-1] # omits final unnecessary row
    responses = responses[-420:].reset_index(drop=True).replace('None', '') # omits practice trial rows, then resets index and replaces 'None' objects with empty string
    responses = responses[:-60] # omit last 5 test trials
    
    # Append to dataframe
    allKeyPresses[fileName] = responses
    
    # Divide responses into bins of three trial sequences (36 key presses)
    indivBins = list(divide_chunks(responses, 36))
    indivCounts = []
    for i in indivBins:
        indivCountInv = 36-sum(j in valResp for j in i) # Count number of invalid responses (36 key presses-number of correct responses)
        indivCounts.append(indivCountInv)
    probIndivBins = [k/36 for k in indivCounts] # Calculate probability of invalid response 
    
    # Plot data
    plt.plot(np.arange(1,len(indivBins)+1), probIndivBins, label=fileName, linewidth=0.25)
    
# Create bins of three trials (36 key presses each bin)
arrCounts = [] # Set empty array where the number of correct responses for each key press will be appended to
for index, row in allKeyPresses.iterrows():
    x = row.tolist() # make each row (key press) a list
    countInv = len(x) - sum(l in valResp for l in x) # Count how many invalid responses there are (i.e. not in valResp) per row
    arrCounts.append(countInv)
bins = list(divide_chunks(arrCounts, 36)) # Divide arrCounts by 3 trial sequences (36 key presses)
sumBins = [sum(m) for m in bins] # Take the sums of each bin (number of invalid key responses per bin)
probBins = [n/(36*len(allKeyPresses.columns)) for n in sumBins] # Get probability of invalid answers per bin - z=number of invalid responses per bin; 36 key presses in a bin*number of participants=total in one bin

# Plot data
plt.plot(np.arange(1,len(bins)+1), probBins, color = 'red', linewidth = 2, label='mean')
plt.xlabel('Bin #')
plt.xticks(np.arange(1,len(bins)+1))
plt.ylabel('Probability of invalid response')
plt.legend(loc=2, fontsize='xx-small')
figBins = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figBins.savefig(saveLoc + '/invRespPerBin OA.png', bbox_inches='tight')

############

# Checking rate of RT < 500 ms

# Define file path (change when necessary)
fileLoc = '/Users/joaqu/OneDrive/Documents/Bates/Kim Lab/dataFiles/20240807 Invalid/data/'
saveLoc = dirname(dirname(fileLoc)) # Files are in a folder named 'data', and we want everything to be saved at the directory before this folder
dirList = sorted(glob.glob(fileLoc + '/*'))

# Loop through each data file to make Key Press Responses dataframe
for file in dirList:
    df = pd.read_csv(file)
    
    # Set file names and save directories
    fileName = file.split('/')[-1:][0].split('\\')[1].split('.')[0]
    
    # Get RT data
    rtData = df['response_time'].iloc[:-1] # create array using response_time data
    rtData = rtData[-420:].reset_index(drop=True).replace('None', np.nan) # omits practice trial rows, then resets index and replaces 'None' objects with nan
    
    rtProbIndiv = [] # Set empty array to append to rtProbDF for each file
    rtData = list(divide_chunks(rtData, 12)) # Divide rtData by each trial (12 key presses)
    rushCounts = []
    for tri in rtData:
        rushPerTrial = sum(rt < 500 for rt in tri) # In one trial, count the number of RTs <500ms
        rushCounts.append(rushPerTrial) # append count to rushCounts
    probRush = [rc/12 for rc in rushCounts] # Divide each count in rushCounts by 12 to get probability that the key press was rushed
    rtProbDF[fileName] = probRush # append probabilities to dataframe

    # Plot data
    plt.plot(trials, probRush, label=fileName, linewidth=0.25)
    
# Plot mean data
rtProbDF['Mean'] = rtProbDF.mean(axis=1) # Create column taking the mean of each row (trial)
rtProbDF['SEM'] = rtProbDF.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
plt.plot(trials, rtProbDF['Mean'].values, color = 'red', linewidth = 2, label='mean')
plt.errorbar(trials, rtProbDF['Mean'].values, yerr = rtProbDF['SEM'].values, fmt='.r', elinewidth=0.5)
plt.xlabel('Trial #')
plt.xticks(trials)
plt.ylabel('Probability of RT < 500 ms')
plt.yticks([0, 0.5, 1.0])
plt.legend(fontsize='xx-small')
figRush = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figRush.savefig(saveLoc + '/rushProb OA.png', bbox_inches='tight')