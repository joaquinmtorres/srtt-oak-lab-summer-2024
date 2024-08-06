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
dirList = sorted(glob.glob(fileLoc + '/*'))

# Get sample file to take key press stimuli sequence
sampleFile = pd.read_csv(os.path.join(fileLoc + 'srttOA01.csv'))
sequences = sampleFile['empty_column'].iloc[:-1] # omits final unnecessary row
sequences = sequences[-420:].reset_index(drop=True).replace('None', np.nan) # omits practice trial rows, then resets index and replaces 'None' objects with nan

# Set up dataframe
trials = np.arange(1, 36) # Trial numbers
allKeyPresses = pd.DataFrame({'Stimulus':sequences}) # df will have sequences as first column which each OA dataset will be compared to

# Loop through each data file
for file in dirList:
    df = pd.read_csv(file)
    
    # Set file names and save directories
    fileName = file.split('/')[-1:][0].split('\\')[1].split('.')[0]
    
    # Get participant responses
    responses = df['response'].iloc[:-1] # omits final unnecessary row
    responses = responses[-420:].reset_index(drop=True).replace('None', np.nan) # omits practice trial rows, then resets index and replaces 'None' objects with nan
    
    # Append to dataframe
    allKeyPresses[fileName] = responses
    
allKeyPresses.to_csv(fileLoc + '/keyPressData.csv', index=False)