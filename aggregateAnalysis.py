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

# Locate data files
fileLoc = input('Enter path where files are located: ')
dirList = sorted(glob.glob(fileLoc + '/*'))
saveLoc = dirname(dirname(fileLoc)) # Files are in a folder named 'data', and we want everything to be saved at the directory before this folder

# Set up aggregate dataframes
trials = np.arange(1, 36) # Trial numbers
aggDF = pd.DataFrame(index=trials) # aggregated average RTs per trial
srDF = pd.DataFrame(index=trials) # aggregated average success rates per trial
aggMiss = pd.DataFrame(index=trials) # aggregated miss rates (RT >= 1000) per trial
aggInc = pd.DataFrame(index=trials) # aggregated incorrect rates (wrong key press within time limit) per trial
patternAware = pd.DataFrame(index=trials)  # aggregated rts of participants who responded aware of a sequence
patternUnaware = pd.DataFrame(index=trials) # aggregated rts of participants who responded unaware or unsure of a sequence
surveyDF = pd.DataFrame() # Set empty df for aggregated survey Data

###################

# Loop through each data file
for file in dirList:
    df = pd.read_csv(file)
    
    # Set file names and save directories
    fileName = file.split('/')[-1:][0].split('\\')[1].split('.')[0]
    saveDir = os.path.join(saveLoc, fileName)
    
    # Skip file if analysis data for that file already exist
    if os.path.isdir(saveDir) == True:
        continue
    else:
        os.makedirs(saveDir, exist_ok=True)
        
        # Clean up dataframe by only keeping necessary columns
        ## First: survey data
        surveyData = df[['survey_awareness', 'survey_order', 'gender', 
                               'age_dropdown', 'race', 'ethnicity', 'ADHD_diagnosis', 
                               'Tourettes_diagnosis', 'Medication',  'sleep_dropdown', 
                               'sleepiness', 'caffeine_consumption', 
                               'drug_consumption', 'modEx_dropdown', 'vigEx_dropdown', 
                               'vision_type', 'comments']].iloc[-1] # Only call last row (which has the responses)
        surveyDF[fileName] = surveyData # Append survey data into surveyDF dataframe
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
        trialNum = trials.repeat(12)
        
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
                
        ## 4. Accuracy (corr)
        corr = np.array(exData['correct']) # create array using correct data
        
        ## 5. Reaction Time (rt)
        rt = np.array(exData['response_time']) # create array using response_time data
        
        ## 6. Phase (phases)
        ### Split trialNum into two arrays
        train1Trials = [x for x in trialNum if 1 <= x <= 15] # array for the first block of training trials (trials 1-15)
        train2Trials = [y for y in trialNum if 16 <= y <= 30] # array for second block of training trials (trials 16-30)
        testTrials = [z for z in trialNum if 31 <= z <= 35] # array for test trials block (trials 31-35)
        trainPhase = ['train'] * (len(train1Trials) + len(train2Trials)) # Create array with 'train' repeated (according to length of both trainTrials)
        testPhase = ['test'] * len(testTrials) # Create array with 'test' repeated (according to length of testTrials)
        phases = trainPhase + testPhase # Merge arrays into one to be appended into the big dataFrame
        
        # Concatenate all arrays into one big dataframe
        allData = pd.DataFrame({'Key Press #':keyPress, 'Trial #':trialNum, 'Key Transition':transList, 'Accuracy':corr,'Response Time':rt, 'Phase':phases})
        # Saving data
        try:
            allData.to_csv(saveDir + '/allData.csv', index=False)
        except FileExistsError:
            print('File already exists')
        
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
                trialStep += 1 # move onto next trial 
                trialRTs = [] # reset for next iteration
                trialRTs.append(dfTrans['RT'][step]) # ensures current row's RT also gets added to new trialRTs list
                step += 1 # move onto next row
                continue
        ##### Note: the last trial's RTs is not appended yet, so the following is needed
        rtPerTrial.append(trialRTs)
        ### Sorting
        aveRTList = [average(j) for j in rtPerTrial] # make array of average RTs of each trial
        aggDF[fileName] = aveRTList # Append aveRTList to aggDF as a column    
        #### sort into patternAware or patternUnaware
        if surveyData['survey_awareness'] == 'awarenessYes':
            patternAware[fileName] = aveRTList
        else:
            patternUnaware[fileName] = aveRTList
        ### Plot the average RT of each trial
        plt.plot(trials, aveRTList)
        plt.xlabel('Trial Number')
        plt.xticks(np.arange(1, len(aveRTList)+1))
        plt.ylabel('Reaction time (ms)')
        plt.title(f'Change in reaction time across all trial sequences of {fileName}')
        fig1 = plt.gcf()
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        try:
            fig1.savefig(saveDir + '/allTrials.png', bbox_inches='tight')
        except FileExistsError:
            print('File already exists')
                           
        ## 2. Hit/miss rates within each sequence across the session
        corrDF = pd.DataFrame({'Acc':corr, 'RT':rt}) # create df with necessary info
        corrArr = [] # set empty array where values will be either corr (correct and w/in time limit), inc (incorrect and w/in time limit), or miss (exceeds time limit)
        ### Assign corr, inc, or miss to each index
        for index, row in corrDF.iterrows():
            if corrDF['Acc'][index] == 1 and corrDF['RT'][index] < 1000:
                corrArr.append('corr')
            if corrDF['Acc'][index] == 0 and corrDF['RT'][index] < 1000:
                corrArr.append('inc')
            if corrDF['Acc'][index] == 0 and corrDF['RT'][index] >= 1000:
                corrArr.append('miss')
        perTrial = list(divide_chunks(corrArr, 12)) # use divide_chunks function to divide corrArr into chunks of size 12
        corrRatePerTrial = [n.count('corr')/12 for n in perTrial] # Create an array that gets correct rate for each trial in perTrial
        srDF[fileName] = corrRatePerTrial # Append aveCorr to srDF as a column
        incRatePerTrial = [o.count('inc')/12 for o in perTrial] # Create an array that gets incorrect rate for each trial in perTrial
        aggInc[fileName] = incRatePerTrial # Append incRatePerTrial to aggInc as a column
        missRatePerTrial = [p.count('miss')/12 for p in perTrial] # Create an array that gets miss rate for each trial in perTrial
        aggMiss[fileName] = missRatePerTrial # Append missRatePerTrial to aggMiss as a column
            
        ## 3. Plot success/incorrect/miss rates in one figure
        plt.figure() # reset
        plt.plot(trials, corrRatePerTrial, color='g', label='success') # Success rate
        plt.plot(trials, incRatePerTrial, color='r', label='incorrect') # Incorrect rate
        plt.plot(trials, missRatePerTrial, color='b', label='miss') # Miss rate
        plt.xlabel('Trial Number')
        plt.xticks(trials)
        plt.ylabel('Rate')
        plt.yticks([0, 0.5, 1.0])
        plt.title(f'Average hit/miss rates for each trial of {fileName}')
        plt.legend()
        figRates = plt.gcf()
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        try:
            figRates.savefig(saveDir + '/rates.png', bbox_inches='tight')
        except FileExistsError:
            print('File already exists') 

###################

# Save surveyDF to csv
surveyDF = surveyDF.transpose() # transpose for easier viewing
try: 
    surveyDF.to_csv(saveLoc + '/allSurveyData.csv')
except FileExistsError:
    print('File already exists')

# Plot aggregate data
## Plot and save aggregate average RTs
aggDF[aggDF >= 1000] = np.nan # convert all rts >= 750 to nan (since those are the misses - late responses)
aggDF['Mean'] = aggDF.mean(axis=1) # Create column taking the mean of each row (trial)
aggDF['SEM'] = aggDF.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
plt.plot(trials, aggDF['Mean'])
plt.errorbar(trials, aggDF['Mean'], yerr=aggDF['SEM'], fmt='.r', ecolor='red', elinewidth=0.5)
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Average reaction time (ms)')
figAve = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
try:
    figAve.savefig(saveLoc + '/allRTs.png', bbox_inches='tight')
except FileExistsError:
    print('File already exists')
## Get averages of last 5 training blocks and 5 test blocks for comparison
lastDF = pd.DataFrame({'Last 5 training blocks':aggDF['Mean'][-10:-5], 'Test blocks':aggDF['Mean'][-5:]})
lastDF = lastDF.apply(lambda x: pd.Series(x.dropna().values)) # removes NaN values and resets index
plt.figure() # reset
plt.boxplot(lastDF)
plt.xticks([1,2], ['Average of last 5 training blocks', 'Average of test blocks'])
plt.ylabel('Response time (ms)')
figLast = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
try:
    figLast.savefig(saveLoc + '/learning.png', bbox_inches='tight')
except FileExistsError:
    print('File already exists')
    
## Plot: comparison of RTs between participants aware of sequence vs. unaware/unsure
### Get means and SEM
#### patternAware
patternAware['Mean'] = patternAware.mean(axis=1) # Create column taking the mean of each row (trial)
patternAware['SEM'] = patternAware.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
#### patternUnaware
patternUnaware['Mean'] = patternUnaware.mean(axis=1) # Create column taking the mean of each row (trial)
patternUnaware['SEM'] = patternUnaware.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
### Plot data
plt.figure() # reset
plt.plot(trials, patternAware['Mean'], color='g', label='Aware')
plt.errorbar(trials, patternAware['Mean'], yerr=patternAware['SEM'], fmt='.g', elinewidth=0.5) # aware of pattern
plt.plot(trials, patternUnaware['Mean'], color='r', label='Unaware')
plt.errorbar(trials, patternUnaware['Mean'], yerr=patternUnaware['SEM'], fmt='.r', elinewidth=0.5) # unaware/unsure of pattern
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Average reaction time (ms)')
plt.legend()
figAware = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
try:
    figAware.savefig(saveLoc + '/awareness.png', bbox_inches='tight')
except FileExistsError:
    print('File already exists')
    
## Plot and save aggregate hit/miss rates
### Get means and SEM
#### success rate
srDF['Mean'] = srDF.mean(axis=1) # Create column taking the mean of each row (trial)
srDF['SEM'] = srDF.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
#### incorrect rate
aggInc['Mean'] = aggInc.mean(axis=1) # Create column taking the mean of each row (trial)
aggInc['SEM'] = aggInc.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
#### miss rate
aggMiss['Mean'] = aggMiss.mean(axis=1) # Create column taking the mean of each row (trial)
aggMiss['SEM'] = aggMiss.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
### Plot data
plt.figure() # reset
plt.plot(trials, srDF['Mean'], color='g', label='success')
plt.errorbar(trials, srDF['Mean'], yerr=srDF['SEM'], fmt='.g', ecolor='g', elinewidth=0.5) # success rates
plt.plot(trials, aggInc['Mean'], color='r', label='incorrect')
plt.errorbar(trials, aggInc['Mean'], yerr=aggInc['SEM'], fmt='.r', ecolor='r', elinewidth=0.5) # incorrect rates
plt.plot(trials, aggMiss['Mean'], color='b', label='miss')
plt.errorbar(trials, aggMiss['Mean'], yerr=aggMiss['SEM'], fmt='.b', ecolor='b', elinewidth=0.5) # miss rates
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Average hit/miss rates')
plt.yticks([0, 0.5, 1.0])
plt.legend()
figAggRates = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
try:
    figAggRates.savefig(saveLoc + '/allHitRates.png', bbox_inches='tight')
except FileExistsError:
    print('File already exists')