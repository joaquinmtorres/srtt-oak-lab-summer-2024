# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:19:20 2024

@author: joaquinmtorres

Sequence data analysis for participant's data from sequence task that uses 
12-items in a sequence. This iterates through multiple data files from one
location and saves them into another. This script is based on dataAnalysis12.py.
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import os.path


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


# Locate data files
fileLoc = input('Enter path where files are located: ')
dirList = sorted(glob.glob(fileLoc + '*'))

# Create aggregate dataframes
## 1. Average RT per trial
trials = np.arange(1, 36) # Set an array of trial numbers
aggDF = pd.DataFrame(index=trials) # Set df with index as trial numbers

# Loop through each data file
for file in dirList:
    df = pd.read_csv(file)
    
    # Clean up dataframe by only keeping necessary columns
    ## First: survey data
    surveyData = df[['survey_awareness', 'survey_order', 'gender', 
                           'age_dropdown', 'race', 'ethnicity', 'ADHD_diagnosis', 
                           'Tourettes_diagnosis', 'Medication',  'sleep_dropdown', 
                           'sleepiness', 'caffeine_consumption', 
                           'drug_consumption', 'modEx_dropdown', 'vigEx_dropdown', 
                           'vision_type']].iloc[-1] # Only call last row (which has the responses)
    ## Then experimental data
    exData = df[['empty_column', 'response', 'correct', 'response_time', 
                       'accuracy', 'average_response_time', 'total_response_time']].iloc[:-1] # omits final unnecessary row
    exData = exData[-420:].reset_index(drop=True).replace('None', np.nan) # omits practice trial rows, then resets index and replaces 'None' objects with nan
    ### NOTE: empty_column refers to the specific key that is displayed (stimulus)
    
    
    # Make arrays for each column
    ## 1. Key Press Number (keyPress)
    keyPress = np.arange(1, len(exData['empty_column'])+1)
    
    ## 2. Trial Number (trialNum)
    ### Create array where each trial number (1-35) are multiplied by the number of items in one sequence (12)
    trialNum = np.arange(1, 36).repeat(12)
    
    ## 3. Key Transitions (transList)
    count = 0 # initialize count
    transList = [] # set empty list where all transitions will go
    ### for loop appending each transition ([key1, key2]) to transList
    for i in exData['empty_column']:
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
    rt = np.where(rt >= 750, np.nan, rt) # convert all rts >= 750 to nan (since those are the misses - late responses)
    
    ## 5. Phase (phases)
    ### Split trialNum into two arrays
    train1Trials = [x for x in trialNum if 1 <= x <= 15] # array for the first block of training trials (trials 1-15)
    train2Trials = [y for y in trialNum if 16 <= y <= 30] # array for second block of training trials (trials 16-30)
    testTrials = [z for z in trialNum if 31 <= z <= 35] # array for test trials block (trials 31-35)
    trainPhase = ['train'] * (len(train1Trials) + len(train2Trials)) # Create array with 'train' repeated (according to length of both trainTrials)
    testPhase = ['test'] * len(testTrials) # Create array with 'test' repeated (according to length of testTrials)
    phases = trainPhase + testPhase # Merge arrays into one to be appended into the big dataFrame
    
    # Concatenate all arrays into one big dataframe
    allData = pd.DataFrame({'Key Press #':keyPress, 'Trial #':trialNum, 'Key Transition':transList, 'Response Time':rt, 'Phase':phases})
    # Saving data
    fileName = file.split('/')[-1:][0].split('\\')[1].split('.')[0]
    saveDir = os.path.join(fileLoc, fileName)
    os.makedirs(saveDir, exist_ok=True)
    try:
        allData.to_csv(saveDir + '/allData.csv', index=False)
    except FileExistsError:
        continue
    
###################
    
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
    aveRTList = [average(j) for j in rtPerTrial] # make array of average RTs of each trial
    aggDF[fileName] = aveRTList # Append aveRTList to aggDF as a column    
    aveRTDF = pd.DataFrame({'Trial Number':np.arange(1,len(aveRTList)+1), 'Average RT':aveRTList}) # creates a dataframe
    plt.plot(np.arange(1, len(aveRTList)+1), aveRTList)
    plt.xlabel('Trial Number')
    plt.xticks(np.arange(1, len(aveRTList)+1))
    plt.ylabel('Reaction time (ms)')
    plt.title('Change in reaction time across all trial sequences')
    fig1 = plt.gcf()
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    try:
        fig1.savefig(saveDir + '/allTrials.png', bbox_inches='tight')
    except FileExistsError:
        continue
       
    
    ## 2. Change in reaction time across trials for each unique transition
    ### First: set up data
    uniqueTrans = np.unique(transList)[1:].tolist() # Creates an array of all unique items from transList, omitting the first np.nan value
    ### Create dataframe setting an index to each uniqueTrans item
    indexDF = pd.DataFrame({'index':np.arange(0,len(uniqueTrans)), 'uniqueTransitions':uniqueTrans})
    ### This for loop iterates through each transition in transList, compares it with each uniqueTransition, and if yes appends the RT2 to the column of corresponding uniqueTransition in rtDF
    #### code adapted from: https://stackoverflow.com/questions/53317739/adding-value-to-only-a-single-column-in-pandas-dataframe
    #### and https://stackoverflow.com/questions/25941979/remove-nan-cells-without-dropping-the-entire-row-pandas-python3
    rtDF = pd.DataFrame(columns = np.arange(0, len(uniqueTrans))) # create empty dataframe with column names as indices (of each unique transition according to indexDF)
    for k in range(0, len(allData)):
         for l in range(0, len(uniqueTrans)):
             if uniqueTrans[l] == allData['Key Transition'][k]:
                 rtDF = rtDF.append({l:allData['Response Time'][k]}, ignore_index=True)
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
    
    ### Then plot data
    for column in finalDF:
        # reset all values
        col1 = []
        col2 = []
        col2 = finalDF[column][1:] # picks a column and creates an array of only the RTs
        col2 = col2[col2!=''].astype(int) # deletes all the empty strings
        if len(col2) > 1: # this code only plots for key transitions where there's 2 or more saved RTs
            col1 = range(1, len(col2)+1)
            keyTransition = uniqueTrans[column]
            plt.plot(col1, col2)
            plt.xlabel('Trial Number')
            plt.xticks(col1) # sets x-axis to go in intervals of 1
            plt.ylabel('Reaction time (ms)')
            plt.title(f'Change in reaction time across trials of {keyTransition}')
            figs = plt.gcf()
            plt.show(block=False)
            plt.pause(2)
            plt.close()
            try:
                figs.savefig(saveDir + f'/{keyTransition}.png', bbox_inches='tight')
            except FileExistsError:
                continue
            
            
    ## 4. Success rate within each sequence across the session
    arrayCorrect = exData['correct'].to_numpy() # isolates correct column into array
    ### NOTE: correct column encodes either 1 (correct response) or 0 (incorrect response)
    ### an average of 1 = perfect responses (no errors)
    perTrial = list(divide_chunks(arrayCorrect, 12)) # use divide_chunks function to divide arrayCorrect into chunks of size 12
    aveCorr = [average(i) for i in perTrial] # calculate average of each chunk
    aveCorrTotal = average(aveCorr) # calculate total average correct
    ### Create dataframe showing average success rate for each trial sequence
    successRates = pd.DataFrame({'Trial #':trials, 'Success Rate':aveCorr})
    ### Plot success rate
    plt.figure() # reset figure
    plt.plot(successRates['Trial #'], successRates['Success Rate'])
    plt.xlabel('Trial Number')
    plt.xticks(trials)
    plt.ylabel('Success Rate')
    plt.yticks([0, 0.5, 1.0])
    plt.title('Average success rates for each trial')
    figSR = plt.gcf()
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    try:
        figSR.savefig(saveDir + '/successRate.png', bbox_inches='tight')
    except FileExistsError:
        continue

###################

# Plot and save aggregate average RTs
aggDF['Mean'] = aggDF.mean(axis=1)
plt.plot(np.arange(1, len(aveRTList)+1), aggDF['Mean'])
plt.xlabel('Trial Number')
plt.xticks(np.arange(1, len(aveRTList)+1))
plt.ylabel('Reaction time (ms)')
plt.title('Change in average response times per trial across all participants')
figAve = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figAve.savefig(fileLoc + '/aggRTs.png', bbox_inches='tight')
