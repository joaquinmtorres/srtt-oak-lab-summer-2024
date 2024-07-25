# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:22:32 2024

@author: joaquinmtorres

Sequence data analysis for one set of data
"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

# This initial step of creating the data files for analysis is taken from longFormDataFrame.py

'''# Determine training sequence
trainSeq = input('Enter training sequence, split using commas without space (e.g. 1,2,3,4): ')
trainSeq = trainSeq.split(',')
trainSeq = [int(num) for num in trainSeq]
'''

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
                    'response_time', 'total_response_time', 'empty_column']].iloc[:-1]


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
    if [row[1] for row in smallList].count(1) == 8: # 8 - number of items in a sequence (change if necessary)
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
# Save to csv - delete ''' ''' if you'd like to save allData as a .csv
''' saveChoice = input('Save as .csv? [y/n]: ')
if saveChoice == 'y':
    savePath = input('Enter path where the file will be saved: ')
    saveName = input('What would you like to name your file? (include .csv at the end): ')
    allData.to_csv(savePath + '/' + saveName, index=False) '''

##############

# Data Analysis
## 1. Change in reaction time across trials
dfTrans = pd.DataFrame({'Trial #': trialNum, 'Key Transition':transList, 'RT':rt}) # filter important data for this section
### a. for each trial, across the whole session
step = 0 # initialize step (to a total of len(dfTrans))
trialStep = 0 # initialize trialStep (should end up with len(trials))
rtPerTrial = [] # initialize empty list where arrays will go. These arrays correspond to the RTs of each trial
trialRTs = [] # initialize empty list that will be recursively reset after every trialStep. This will contain RTs of a single trial
#### for loop that recursively appends rtPerTrial with arrays of RTs for each trial
for index, row in dfTrans.iterrows():
    if dfTrans['Trial #'][step] == trials[trialStep]: # if row's trial value corresponds to current trialStep
        trialRTs.append(dfTrans['RT'][step]) # append that row's RT to trialRTs
        step += 1 # move onto next row
        continue
    else: # when the for loop moves to a new trial #
        rtPerTrial.append(trialRTs) # add trialRTs to larger array
        trialStep += 1 # move onto next trial #
        trialRTs = [] # reset for next iteration
        trialRTs.append(dfTrans['RT'][step]) # ensures current row's RT also gets added to new trialRTs list
        step += 1 # move onto next row
        continue
##### Note: the last trial's RTs is not appended yet, so the following is needed
rtPerTrial.append(trialRTs)
### Plot the average RT of each trial
aveRTList = [average(m) for m in rtPerTrial] # make array of average RTs of each trial
aveRTDF = pd.DataFrame({'Trial Number':np.arange(1,len(aveRTList)+1), 'Average RT':aveRTList}) # creates a dataframe
plt.plot(np.arange(1, len(aveRTList)+1), aveRTList)
plt.xlabel('Trial Number')
plt.xticks(np.arange(1, len(aveRTList)+1))
plt.ylabel('Reaction time (ms)')
plt.title('Change in reaction time across all trial sequences')
plt.show()
# OR IF YOU WANT TO SAVE
## plt.savefig(savePath + 'allTrials.png', bbox_inches='tight')

## 2. Number of key presses per trial
keyPressPerTrial = [len(n) for n in rtPerTrial] # get length of each item in rtPerTrial and concatenate into an array
plt.plot(np.arange(1, len(keyPressPerTrial)+1), keyPressPerTrial)
plt.xlabel('Trial Number')
plt.xticks(np.arange(1, len(keyPressPerTrial)+1))
plt.ylabel('Count')
plt.yticks(np.arange(8,10))
plt.title('Number of key presses per trial')
plt.show()


########## These are not necessary as of now
'''## 1. Reaction times across session
arrayRT = allData['Reaction Time'].to_numpy() # isolates response_time column into array
aveRT = average(arrayRT) # calculates average of all response_time
### show result (average RT)
print(f'The average response time is {aveRT} ms')'''
''' #Misc - from previous data analysis with 12 in a sequence, including all transition data
dfTrans = exData.filter(['empty_column', 'response_time'], axis=1) # filter only necessary info from exData
arrayKey = dfTrans['empty_column'].to_numpy() # isolates empty_column into array
### NOTE: empty_column refers to the specific key that is displayed (stimulus)
bigList = [] # empty dataframe that will contain each transition
step = 0 # set up step/index
### for loop grouping each transition in the whole sequence together
for index, row in dfTrans.iterrows():
     smallList = []
     while step < len(dfTrans)-1:
         smallList.append(dfTrans.loc[[step, step+1]])
         step += 1
         bigList.append(smallList)
#### code adapted from: https://stackoverflow.com/questions/16476924/how-can-i-iterate-over-rows-in-a-pandas-dataframe

### Make data frame containing key transition and corresponding data (key + RT for both values)
count = 0 # set up count/index
arrTrans = [] # empty list - each specTrans will be appended into this list
allData = [] #empty list - each transition data will be appended into this list
for i in bigList[0]:
    while count < len(bigList[0])-1:
        transition = np.array(bigList[0][count].values) # creates an array from each transition - ([key1, rt1], [key2, rt2])  
        specTrans = [transition[0,0], transition[1,0]] # creates an array of the specific key transition - [key1, key2]
        arrTrans.append(specTrans)
        allData.append(transition)
        count += 1

### Concatenate a dataframe with arrTrans and allData as columns
orgTrans = pd.DataFrame({'keyTransitions':arrTrans, 'transitionsRT':allData})
uniqueTrans = np.unique(np.array(arrTrans), axis=0) # creates an array of unique key transitions across the whole sequence
arrOrgTrans = np.array(orgTrans) # convert orgTrans into array for compatibility

### Create dataframe setting an index to each uniqueTrans item
indexDF = pd.DataFrame({'index':np.arange(0,len(uniqueTrans.tolist())), 'uniqueTransitions':uniqueTrans.tolist()})

### Create dataframe setting columns as indices, then append each RT2 to corresponding index of its uniqueTrans
#### code adapted from: https://stackoverflow.com/questions/53317739/adding-value-to-only-a-single-column-in-pandas-dataframe
#### and https://stackoverflow.com/questions/25941979/remove-nan-cells-without-dropping-the-entire-row-pandas-python3
transList = arrOrgTrans.tolist() # convert arrOrgTrans to list (compatible with appending to dataframe)
utList = uniqueTrans.tolist() # convert uniqueTrans to list (compatible with appending to dataframe)
rtDF = pd.DataFrame(columns = np.arange(0, len(uniqueTrans.tolist()))) # create empty dataframe with column names as indices (of each unique transition according to indexDF)
#### This for loopiterates through each transition in transList, looks at whether its transition == each uniqueTransition, and if yes, appends the RT2 to the column of corresponding uniqueTransition in rtDF
for j in range(0, len(transList)):
     for k in range(0, len(utList)):
         if utList[k] == transList[j][0]:
             rtDF = rtDF.append({k:transList[j][1][1][1]}, ignore_index=True)
         else:
             continue
#### This for loop appends a value to each row, so each row would have one RT2 in one column, while the rest is NaN
#### To clean up NaN values:
rtDF = rtDF.apply(lambda x: pd.Series(x.dropna().values)) # removes most NaN values
rtDF = rtDF.fillna('') # replaces remaining NaN values with empty strings

### Create a dataframe containing uniqueTransitions (from indexDF) and RT2 values (from rtDF)
rtDF = rtDF.transpose() # transpose datatframe, so that it can easily merge with indexDF
finalDF = indexDF.join(rtDF) # merge indexDF and rtDF into one dataframe
finalDF = finalDF.drop('index', axis=1) # delete index column, so the first column is uniqueTransitions
finalDF = finalDF.transpose() # transpose dataframe so that uniqueTransitions can be pseudo-column names

### Save dataframe as CSV file
filePath = input('Enter path where you would like to save this file: ')
finalDF.to_csv(filePath + '/transitionDataForEachRT.csv', index=False, header=False) #parameters ensure that row and column names (indices) are excluded

##############

## 3. Above two items except considering number of tries to get to the correct response
savePath = input('Enter path where plots will be saved: ') # input where images of plots will go to
### 3.1 Plot how average RT of each sequence changes after every trial
n = 12 # set n to 12 (number of items for every sequence) to divide arrayRT
chunksRT = list(divide_chunks(arrayRT, n)) # divide_chunks function divides arrayRT into chunks of size 12 (sequence size)
aveRTList = [average(i) for i in chunksRT] # calculate average of each chunk
aveRTPerTrial = pd.DataFrame({'Trial Number':np.arange(1,len(aveRTList)+1), 'Average RT':aveRTList})
plt.plot(np.arange(1, len(aveRTList)+1), aveRTList)
plt.xlabel('Trial Number')
plt.xticks(np.arange(1, len(aveRTList)+1))
plt.ylabel('Reaction time (ms)')
plt.title('Change in reaction time across all trial sequences')
plt.show()
# OR IF YOU WANT TO SAVE
## plt.savefig(savePath + 'allTrials.png', bbox_inches='tight')

### 3.2 Plot change in RT for each key transition
for column in finalDF:
    # reset all values
    plt.figure()
    col1 = []
    col2 = []
    col2 = finalDF[column][1:] # picks a column and creates an array of only the RTs
    col2 = col2[col2!=''].astype(int) # deletes all the empty strings
    if len(col2) > 1: # this code only plots for key transitions where there's 2 or more saved RTs
        col1 = range(1, len(col2)+1)
        keyTransition = uniqueTrans[column].tolist()
        plt.plot(col1, col2)
        plt.xlabel('Trial Number')
        plt.xticks(col1) # sets x-axis to go in intervals of 1
        plt.ylabel('Reaction time (ms)')
        plt.title(f'Change in reaction time across trials of {keyTransition}')
        plt.show()
        # OR IF YOU WANT TO SAVE
        ## plt.savefig(savePath + f'{keyTransition}.png', bbox_inches='tight')'''
'''## 4. Success rate within each sequence across the session
arrayCorrect = exData['correct'].to_numpy() # isolates correct column into array
### NOTE: correct column encodes either 1 (correct response) or 0 (incorrect response)
### an average of 1 = perfect responses.
n = 12 # set n to 12 (number of items for every sequence) to divide arrayCorrect
chunksCorr = list(divide_chunks(arrayCorrect, n)) # use divide_chunks function to divide arrayCorrect into chunks of size 12
avesList = [average(i) for i in chunksCorr] # calculate average of each chunk
aveCorrTotal = average(avesList) # calculate total average correct
### Create dataframe showing average success rate for each trial sequence
successRatePerTrial = pd.DataFrame({'Trial Number':np.arange(1,len(avesList)+1), 'Success Rate':avesList})

##############

## 5. Whether perturbing the sequence caused an increase in error rates, reaction times, or attempts to correct response'''