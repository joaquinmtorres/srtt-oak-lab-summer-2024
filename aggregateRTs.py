# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:51:54 2024

@author: joaquinmtorres

Taking all OA and YA SRTT data (RTs) and combining them into one figure.
Adapted from aggregateAnalysis.py
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from statsmodels.stats.anova import AnovaRM 
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
        
# Define file paths
fileLoc = '/Users/joaqu/OneDrive/Documents/Bates/Kim Lab/dataFiles/20240806 All/data/'
dirList = sorted(glob.glob(fileLoc + '/*'))
saveLoc = dirname(dirname(fileLoc)) # Files are in a folder named 'data', and we want everything to be saved at the directory before this folder

# Set up aggregate dataframes
trials = np.arange(1, 36) # Trial numbers
oaRTs = pd.DataFrame(index=trials)
yaRTs = pd.DataFrame(index=trials)
surveyDF = pd.DataFrame()

# Loop through each data file
for file in dirList:
    df = pd.read_csv(file)
    
    # Set file names and save directories
    fileName = file.split('/')[-1:][0].split('\\')[1].split('.')[0]
    
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
    
    rt = np.array(exData['response_time']) # create array using response_time data
    ### Create array where each trial number (1-35) are multiplied by the number of items in one sequence (12)
    trialNum = trials.repeat(12)
    
    dfTrans = pd.DataFrame({'Trial #': trialNum, 'RT':rt}) # filter important data for this section
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
    if surveyData['age_dropdown'] >= 65:
        oaRTs[fileName] = aveRTList # append aveRTList to OA dataframe if age is greater than or equal to 65
    else:
        yaRTs[fileName] = aveRTList # append aveRTList to YA dataframe if not

# Save surveyDF to csv
surveyDF = surveyDF.transpose() # transpose for easier viewing
surveyDF.to_csv(saveLoc + '/allSurveyData.csv', index=False)
    
# Take means and SEM
## OA
oaRTs[oaRTs >= 1000] = np.nan # convert all rts >= 750 to nan (since those are the misses - late responses)
oaRTs['Mean'] = oaRTs.mean(axis=1) # Create column taking the mean of each row (trial)
oaRTs['SEM'] = oaRTs.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
## YA
yaRTs[yaRTs >= 1000] = np.nan # convert all rts >= 750 to nan (since those are the misses - late responses)
yaRTs['Mean'] = yaRTs.mean(axis=1) # Create column taking the mean of each row (trial)
yaRTs['SEM'] = yaRTs.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column

# Plot data
plt.figure()
plt.plot(trials, oaRTs['Mean'], color='b', label='older adults')
plt.errorbar(trials, oaRTs['Mean'], yerr=oaRTs['SEM'], fmt='.b', ecolor='b', elinewidth=0.5)
plt.plot(trials, yaRTs['Mean'], color='r', label='younger adults')
plt.errorbar(trials, yaRTs['Mean'], yerr=yaRTs['SEM'], fmt='.r', ecolor='r', elinewidth=0.5)
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Average reaction time (ms)')
plt.title('Comparing average RTs per trial of older adults vs. younger adults')
plt.legend()
figAve = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figAve.savefig(saveLoc + '/allRTs.png', bbox_inches='tight')

###########

# Statistical test
# Define file path
fileLoc = '/Users/joaqu/OneDrive/Documents/Bates/Kim Lab/dataFiles/20240806 All/data/'
dirList = sorted(glob.glob(fileLoc + '/*'))
saveLoc = dirname(dirname(fileLoc)) # Files are in a folder named 'data', and we want everything to be saved at the directory before this folder

# Create empty arrays to append to
participants = []
rtData = []
phase = []
ageGroup = []

# Loop through each file
for file in dirList:
    df = pd.read_csv(file)
    
    # Set file names and save directories
    fileName = file.split('/')[-1:][0].split('\\')[1].split('.')[0]
    
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
    
    # Append participants array with fileName twice
    participants.append(fileName)
    participants.append(fileName)
    
    # Calculate average RT data for last 5 training and 5 test and append each to rtData
    trainAve = exData['response_time'][-10:-5].mean()
    rtData.append(trainAve)
    testAve = exData['response_time'][-5:].mean()
    rtData.append(testAve)
    
    # Append phases to phase array
    phase.append('train')
    phase.append('test')
    
    # Determine age group and append to ageGroup twice
    if surveyData['age_dropdown'] >= 65:
        age = 'OA'
    else:
        age = 'YA'
    ageGroup.append(age)
    ageGroup.append(age)

# Create dataframe with using the arrays created
statsDF = pd.DataFrame({'Participant':participants, 'RT':rtData, 'Phase':phase, 'Age Group':ageGroup})
statsDF.to_csv(saveLoc+'/allStatsData.csv', index=False)

# Run stats test (repeated measures ANOVA)
print(AnovaRM(data=statsDF, depvar='RT', subject='Participant', within=['Phase']).fit())