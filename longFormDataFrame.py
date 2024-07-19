# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:29:47 2024

@author: joaquinmtorres

Making a long form dataframe/csv file of sequence data. 
"""
# Import libraries
import numpy as np
import pandas as pd


# Input data file and convert to pandas data frame
dataFile = input('Input data file path (MUST BE CSV FILE): ')
dataFile = pd.read_csv(dataFile)
dataFile = dataFile.iloc[4:] # omit first 4 practice/pre-trial rows
dataFile = dataFile.reset_index(drop=True) # resets index
# Clean up dataframe by only keeping necessary columns
## First: survey data
surveyData = dataFile[['survey_awareness', 'survey_order', 'gender', 
                       'age_dropdown', 'race', 'ethnicity', 'ADHD_diagnosis', 
                       'Tourettes_diagnosis', 'Medication',  'sleep_dropdown', 
                       'sleepiness', 'caffeine_consumption', 
                       'drug_consumption', 'modEx_dropdown', 'vigEx_dropdown', 
                       'vision_type']].iloc[-1] # Only call last row (which has the responses)
## Then experimental data
exData = dataFile[['accuracy', 'average_response_time', 'correct', 'response', 
                    'response_time', 'total_response_time', 'empty_column']]


# Make arrays for each column
## 1. Key Press Number (keyPress)
keyPress = np.arange(1, len(exData['empty_column'])+1)
### NOTE: empty_column refers to the specific key that is displayed (stimulus)

## 2. Trial Number (trialNum)
### for loop creating a list dividing the whole set of key presses into sequences
keyAccDF = pd.DataFrame({'key': exData['empty_column'], 'accuracy':exData['correct']}) # create df which will be basis of next arrays
keyAccArr = [] # set an empty array
for index, row in keyAccDF.iterrows():
    keyAccArr.append([row['key'], row['accuracy']]) # manipulates keyAccArr so that it'll store arrays of [key, accuracy] from keyAccDF
    # sourced from: https://stackoverflow.com/questions/16476924/how-can-i-iterate-over-rows-in-a-pandas-dataframe
bigList = [] # initialize list where all sequences will go
smallList = [] # initialize smallList (denoting one sequence)
for i in keyAccArr:
    if [row[1] for row in smallList].count(1) == 8:
        bigList.append(smallList) # if smallList contains the right sequence (i.e. correct keyPresses=1, including error key presses w/in that same sequence), that sequence is appended to bigList
        smallList = [] # reset smallList
        smallList.append(i) # append current i to new smallList
    else:
        smallList.append(i) # append the current [key, accuracy] to smallList
bigList.append(smallList) # append remaining data (not part of a whole sequence) to bigList
### Assigning trial number to each keyPress
trials = np.arange(1, len(bigList)+1) # Set an array of trial numbers
trialNum = [] # Set an empty array where (trial number * number of items in each sequence in bigList) will go
for j in trials:
    for k in bigList[j-1]:
        trialNum.extend([j] * int(len(k)/2)) # len(k) divided by 2 because len(k) takes both key and accuracy values into account
trialNum = np.array(trialNum) # convert list to array

## 3. Key Transitions (transList)
count = 0 # initialize count
transList = [] # set empty list where all transitions will go
### for loop appending each transition ([key1, key2]) to transList
for l in exData['empty_column']:
    try:
        while count < len(exData['empty_column']):
            transition = [exData['empty_column'][count-1], exData['empty_column'][count]] # set transition as [previous key, current key]
            transList.append(transition)
            count += 1
    except KeyError: # when faced with a KeyError, which is during the first iteration (because there's no key before the first key)
        transList.append([np.nan]) # instead, append a NaN value
        count += 1

## 4. Reaction Time (rt)
rt = np.array(exData['response_time']) # create array using response_time data

## 5. Phase (phases)
### Split trialNum into two arrays
trainTrials = [x for x in trialNum if 1 <= x <= 16] # Create array containing every 1-16 (training trials) from trialNum
testTrials = [y for y in trialNum if 17 <= y <= 32] # Create array containing every 17-32 (testing trials) from trialNum
trainPhase = ['train'] * len(trainTrials) # Create array with 'train' repeated (according to length of trainTrials)
testPhase = ['test'] * len(testTrials) # Create array with 'test' repeated (according to length of testTrials)
phases = trainPhase + testPhase # Merge arrays into one to be appended into the big dataFrame


# Concatenate all arrays into one big dataframe
allData = pd.DataFrame({'Key Press #':keyPress, 'Trial #':trialNum, 'Key Transition':transList, 'Reaction Time':rt, 'Phase':phases})
# Save to csv?
saveChoice = input('Save as .csv? [y/n]: ')
if saveChoice == 'y':
    savePath = input('Enter path where the file will be saved: ')
    saveName = input('What would you like to name your file? (include .csv at the end): ')
    allData.to_csv(savePath + '/' + saveName, index=False)