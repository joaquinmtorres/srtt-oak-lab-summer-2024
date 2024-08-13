# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:51:37 2024

@author: joaquinmtorres

Consolidating all analyses into one code, including:
    - change in average response time
    - average correct/incorrect/miss rates per trial
    - difference in RT change per trial between pattern aware vs. unaware
    - difference in average RTs between last 5 training blocks vs. test blocks
    - change in rushed response rate per trial
    - change in reward rate per trial
    - change in punishment rate per trial
    - stats test (a/o 20240809, only the dataframe for stats is included)

Current version of this code does not omit misses (i.e. RT > 1000ms). This 
could easily be implemented by uncommenting line 129 (rt array)
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from os.path import dirname

# Helper Functions
## Breaks list into chunks of size n
### from: https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
def divide_chunks(l, n): 
    # looping till length l 
    for x in range(0, len(l), n):  
        yield l[x:x + n] 

## To determine accuracy of reported sequence (from ChatGPT)
def isinSequence(response, sequence):
    # response = reported sequence, sequence = trainSeq
    # Iterate over sequence with a sliding window
    for t in range(len(sequence)-len(response)+1):
        if sequence[t:t+len(response)] == response:
            return len(response)/len(sequence)*2
    return False

# Generating boxplot figure with two datasets
## from: https://www.geeksforgeeks.org/how-to-create-boxplots-by-group-in-matplotlib/
# each plot returns a dictionary, use plt.setp()
# function to assign the color code
# for all properties of the box plot of particular group
# use the below function to set color for particular group,
# by iterating over all properties of the box plot
def define_box_properties(plot_name, color_code, label):
    for kk, vv in plot_name.items():
        plt.setp(plot_name.get(kk), color=color_code)
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()

#############
        
# Define necessary info
fileLoc = input('Enter location/path of data files: ')
dirList = sorted(glob.glob(fileLoc + '/*'))
saveLoc = dirname(dirname(fileLoc)) # Files are in a folder named 'data', and we want everything to be saved at the directory before this folder
howMany = int(input('How many key presses per trial? (implicit:12, explicit:6): '))

# Set up aggregate dataframes
trials = np.arange(1, 36) # Trial numbers
parID = [] # set empty array for surveyDF column
surveyDF = pd.DataFrame() # Set empty df for aggregated survey Data
## Across all OA Data
oaRTs = pd.DataFrame(index=trials) # for response times (same figure with yaRTs)
corrRateOA = pd.DataFrame(index=trials) # average success rates per trial
missRateOA = pd.DataFrame(index=trials) # average miss rates (RT >= 1000) per trial
incRateOA = pd.DataFrame(index=trials) # average incorrect rates (wrong key press within time limit) per trial
pattAwOA = pd.DataFrame(index=trials)  # rts of participants who responded aware of a sequence
pattUnOA = pd.DataFrame(index=trials) # rts of participants who responded unaware or unsure of a sequence
rushProbOA = pd.DataFrame(index=trials) # probability of rushed response (rt < 500 ms) per trial
rewardRateOA = pd.DataFrame(index=trials) # probability of reward (ding) per trial
punishRateOA = pd.DataFrame(index=trials) # probability punished per trial
## YA Data
yaRTs = pd.DataFrame(index=trials) # for response times (same figure with oaRTs)
corrRateYA = pd.DataFrame(index=trials) # average success rates per trial
missRateYA = pd.DataFrame(index=trials) # average miss rates (RT >= 1000) per trial
incRateYA = pd.DataFrame(index=trials) # average incorrect rates (wrong key press within time limit) per trial
pattAwYA = pd.DataFrame(index=trials)  # rts of participants who responded aware of a sequence
pattUnYA = pd.DataFrame(index=trials) # aggregated rts of participants who responded unaware or unsure of a sequence
rushProbYA = pd.DataFrame(index=trials) # probability of rushed response (rt < 500 ms) per trial
rewardRateYA = pd.DataFrame(index=trials) # probability of reward (ding) per trial
punishRateYA = pd.DataFrame(index=trials) # probability punished per trial

# Set empty arrays/dataframes for additional patternAware data analyses
## OA
improveOA = [] # improvement rates (average of T1-3 - average of T28-30)
regressOA = [] # regression (difference of T31-T30/difference of T1-T30)
trainSlopesOA = [] # learning rate (average difference of slopes of T1-5)
testSlopesOA = [] # learning rate (average difference of slopes of T31-35)
stabilizeOA = pd.DataFrame(index=np.arange(1,35)) # when learning stabilizes (i.e. when there is little to no difference in improvement in terms of correct rate); index=each transition
variabilityOA = pd.DataFrame(index=trials) # change in RT variability within each trial
## YA
improveYA = [] # improvement rates (average of T1-3 - average of T28-30)
regressYA = [] # regression (difference of T31-T30/difference of T1-T30)
trainSlopesYA = [] # learning rate (average difference of slopes of T1-5)
testSlopesYA = [] # learning rate (average difference of slopes of T31-35)
stabilizeYA = pd.DataFrame(index=np.arange(1,35)) # when learning stabilizes (i.e. when there is little to no difference in improvement in terms of correct rate); index=each transition
variabilityYA = pd.DataFrame(index=trials) # change in RT variability within each trial

# Set empty arrays for stats tests
participants = []
rtData = []
missRates = []
phase = []
ageGroup = []

#############

# Loop through each data file
for file in dirList:
    df = pd.read_csv(file)
    
    # Set file names and save directories
    fileName = file.split('/')[-1:][0].split('\\')[1].split('.')[0]
    saveDir = os.path.join(saveLoc, fileName)
    parID.append(fileName) # save participant id 
    
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
    exData = exData[-howMany*35:].reset_index(drop=True).replace('None', np.nan) # omits practice trial rows, then resets index and replaces 'None' objects with nan
    ### NOTE: empty_column refers to the specific key that is displayed (stimulus)
    
    #############
    # Create dataframe with all necessary information
    
    # Make arrays for each column
    ## 1. Key Press Number (keyPress)
    keyPress = np.arange(1, len(exData['empty_column'])+1)
    
    ## 2. Trial Number (trialNum)
    ### Create array where each trial number (1-35) are multiplied by the number of items in one sequence
    trialNum = trials.repeat(howMany)
    
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
    # rt[rt >= 1000] = np.nan # convert all rts >= 1000 to NaN (since those are the misses - late responses)
    
    ## 6. Phase (phases)
    ### Split trialNum into two arrays
    train1Trials = [j for j in trialNum if 1 <= j <= 15] # array for the first block of training trials (trials 1-15)
    train2Trials = [k for k in trialNum if 16 <= k <= 30] # array for second block of training trials (trials 16-30)
    testTrials = [l for l in trialNum if 31 <= l <= 35] # array for test trials block (trials 31-35)
    trainPhase = ['train'] * (len(train1Trials) + len(train2Trials)) # Create array with 'train' repeated (according to length of both trainTrials)
    testPhase = ['test'] * len(testTrials) # Create array with 'test' repeated (according to length of testTrials)
    phases = trainPhase + testPhase # Merge arrays into one to be appended into the big dataFrame
    
    # Concatenate all arrays into one big dataframe
    allData = pd.DataFrame({'Key Press #':keyPress, 'Trial #':trialNum, 'Key Transition':transList, 'Accuracy':corr,'Response Time':rt, 'Phase':phases})
    allData.to_csv(saveDir + '/allData.csv', index=False) # Save data
    
    #############
    # Fill up empty dataframes
    
    # Average response times per trial to oaRTs/yaRTs dataframes, as well as corresponding patternAware/patternUnaware dataframes
    rtPerTrial = list(divide_chunks(rt, howMany)) # Divide rt data into trials
    ### Sorting
    aveRTList = [np.nanmean(m) for m in rtPerTrial] # make array of average RTs of each trial
    if surveyData['age_dropdown'] >= 65:
        oaRTs[fileName] = aveRTList # append aveRTList to OA dataframe if age is greater than or equal to 65
        if surveyData['survey_awareness'] == 'awarenessYes':
            pattAwOA[fileName] = aveRTList
        else:
            pattUnOA[fileName] = aveRTList
    else:
        yaRTs[fileName] = aveRTList # append aveRTList to YA dataframe if not
        if surveyData['survey_awareness'] == 'awarenessYes':
            pattAwYA[fileName] = aveRTList
        else:
            pattUnYA[fileName] = aveRTList
        
    # Correct, incorrect, and miss rates per trial
    keyResp = [] # set empty array
    # Build keyResp array with integers. If response is 'None', append an empty string
    for kp in exData['response']:
        try:
            key = int(kp)
            keyResp.append(key)
        except ValueError:
            keyResp.append('')
    accuracyDF = pd.DataFrame({'Stimulus':exData['empty_column'], 'Response':keyResp, 'RT':rt}).replace(np.nan, '') # create df with necessary info, replacing missing responses (None or np.nan in) into empty string
    accuracies = [] # set empty array where values will be either corr (correct and w/in time limit), inc (incorrect and w/in time limit), or miss (exceeds time limit)
    ## Assign corr, inc, or miss to each index
    for index, row in accuracyDF.iterrows():
        if accuracyDF['Stimulus'][index] == accuracyDF['Response'][index]:
            accuracies.append('corr')
        if accuracyDF['Response'][index] == '':
            accuracies.append('miss')
        if accuracyDF['Response'][index] != '' and accuracyDF['Response'][index] != accuracyDF['Stimulus'][index]:
            accuracies.append('inc')
    accuracyDF['Accuracy'] = accuracies # Add accuracy data as a column in dataframe
    accuracyDF.to_csv(saveDir + '/accuracyData.csv', index=False) # Save data
    perTrial = list(divide_chunks(accuracies, howMany)) # divide corrArr into trials
    corrRatePerTrial = [n.count('corr')/howMany for n in perTrial] # Create an array that gets correct rate for each trial in perTrial
    incRatePerTrial = [o.count('inc')/howMany for o in perTrial] # Create an array that gets incorrect rate for each trial in perTrial
    missRatePerTrial = [p.count('miss')/howMany for p in perTrial] # Create an array that gets miss rate for each trial in perTrial
    ## Plot individual data
    plt.figure() # reset
    plt.plot(trials, corrRatePerTrial, color='g', label='correct')
    plt.plot(trials, incRatePerTrial, color='r', label='incorrect')
    plt.plot(trials, missRatePerTrial, color='b', label='miss')
    plt.xlabel('Trial Number')
    plt.xticks(trials)
    plt.ylabel('Probability')
    plt.yticks([0, 0.5, 1.0])
    plt.title(f'Change in average correct/incorrect/miss rates of {fileName}')
    plt.legend()
    figHitIndiv = plt.gcf()
    plt.close()
    figHitIndiv.savefig(saveDir + f'/hitRates {fileName}.png', bbox_inches='tight')
    ## Sorting by age group to the appropriate dataframe
    if surveyData['age_dropdown'] >= 65:
        corrRateOA[fileName] = corrRatePerTrial
        incRateOA[fileName] = incRatePerTrial
        missRateOA[fileName] = missRatePerTrial
    else:
        corrRateYA[fileName] = corrRatePerTrial
        incRateYA[fileName] = incRatePerTrial
        missRateYA[fileName] = missRatePerTrial
        
    # Last 5 training blocks vs. test blocks + stats test
    ## Append participants array with fileName twice
    participants.append(fileName)
    participants.append(fileName)
    ## Calculate average RT data for last 5 training and 5 test and append each to rtData
    trainAve = np.nanmean(aveRTList[-10:-5])
    rtData.append(trainAve)
    testAve = np.nanmean(aveRTList[-5:])
    rtData.append(testAve)
    ## Calculate miss rates for last 5 training and 5 test and append each to missRates
    trainMiss = np.nanmean(missRatePerTrial[-10:-5])
    missRates.append(trainMiss)
    testMiss = np.nanmean(missRatePerTrial[-5:])
    missRates.append(testMiss)
    ## Append phases to phase array
    phase.append('train')
    phase.append('test')
    ## Determine age group and append to ageGroup twice
    if surveyData['age_dropdown'] >= 65:
        age = 'OA'
    else:
        age = 'YA'
    ageGroup.append(age)
    ageGroup.append(age)
    
    # Probabilities of rushed response per trial
    rushPerTrial = [] # Array where counts of rushed responses per trial will go
    for q in rtPerTrial:
        rushCount = sum(r < 500 for r in q) # for each trial, count number of RTs<500ms
        rushPerTrial.append(rushCount)
    rushProbs = [s/howMany for s in rushPerTrial] # Divide each count in rushPerTrial by 12 to get probability of rush per trial
    ## Sorting data
    if surveyData['age_dropdown'] >= 65:
        rushProbOA[fileName] = rushProbs
    else:
        rushProbYA[fileName] = rushProbs
        
    # Probability of reward per trial (reward/ding condition: correct response & rt<500ms)
    ## using accuracyDF
    dingCount = [] # set empty array
    for index, row in accuracyDF.iterrows():
        if accuracyDF['Stimulus'][index] == accuracyDF['Response'][index] and accuracyDF['RT'][index] < 500:
            dingCount.append(1) # Append 1 ding if it satisfies conditions for a ding
        else:
            dingCount.append(0)
    dingPerTrial = divide_chunks(dingCount, howMany) # divide by trial
    dingRates = [np.nanmean(y) for y in dingPerTrial] # Get ding rate per trial
    if surveyData['age_dropdown'] >= 65:
        rewardRateOA[fileName] = dingRates
    else:
        rewardRateYA[fileName] = dingRates
        
    # Probability punished per trial (condition: RT>=1000ms OR incorrect response)
    ## using accuracyDF
    punishCount = [] # set empty array
    for index, row in accuracyDF.iterrows():
        if accuracyDF['Stimulus'][index] == accuracyDF['Response'][index]:
            punishCount.append(0) # if correct response within time limit, append 0
        else:
            punishCount.append(1)
    punishPerTrial = divide_chunks(punishCount, howMany) # divide by trial
    punishRates = [np.nanmean(z) for z in punishPerTrial] # Get punishment rate per trial
    if surveyData['age_dropdown'] >= 65:
        punishRateOA[fileName] = punishRates
    else:
        punishRateYA[fileName] = punishRates
        
    #############
    # patternAware data analyses
    if surveyData['survey_awareness'] == 'awarenessYes':
        
        ## Improvement Rate (average of T1-3 - average of T28-30)
        firstThree = np.nanmean(aveRTList[0:3]) # Take average of Trials 1-3 - initial training blocks
        lastThree = np.nanmean(aveRTList[27:30]) # Take average of Trials 28-30 - final training blocks
        impRate = firstThree - lastThree # Take difference to get improvement rate per participant
        ### Store into arrays
        if surveyData['age_dropdown'] >= 65:
            improveOA.append(impRate)
        else:
            improveYA.append(impRate)
            
        ## Regression @ start of test (amount of sequence-specific learning)
        numRegress = aveRTList[30] - aveRTList[29] # Get numerator, or Trial 31-Trial 30
        denRegress = aveRTList[0] - aveRTList[29] # Get denominator, or Trial 1-Trial 30
        ratioReg = numRegress/denRegress # Get ratio
        ### Store into arrays
        if surveyData['age_dropdown'] >= 65:
            regressOA.append(ratioReg)
        else:
            regressYA.append(ratioReg)
            
        ## Learning rate (ave. slopes of first 5 train blocks vs. ave. slopes of test blocks)
        aveTrainSlopes = np.nanmean(np.diff(aveRTList[:5])) # Get average slope of first 5 training blocks
        aveTestSlopes = np.nanmean(np.diff(aveRTList[-5:])) # Get average slope of test blocks
        ### Store into arrays
        if surveyData['age_dropdown'] >= 65:
            trainSlopesOA.append(aveTrainSlopes)
            testSlopesOA.append(aveTestSlopes)
        else:
            trainSlopesYA.append(aveTrainSlopes)
            testSlopesYA.append(aveTestSlopes)
            
        # When learning stabilizes (i.e. when correct rate doesn't improve more)
        improvements = np.diff(corrRatePerTrial) # get improvement per trial
        if surveyData['age_dropdown'] >= 65:
            stabilizeOA[fileName] = improvements
        else:
            stabilizeYA[fileName] = improvements
            
        # Change in RT variability within each trial
        sdPerTrial = [np.std(seq) for seq in rtPerTrial] # create array of standard deviation of each trial
        if surveyData['age_dropdown'] >= 65:
            variabilityOA[fileName] = sdPerTrial
        else:
            variabilityYA[fileName] = sdPerTrial
       
#############
# Plot data

# Change in average rt (per trial, across all participants, OA vs. YA)
## Calculate means and SEM per trial
### OA
oaRTs['Mean'] = oaRTs.mean(axis=1) # Create column taking the mean of each row (trial)
oaRTs['SEM'] = oaRTs.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
## YA
yaRTs['Mean'] = yaRTs.mean(axis=1) # Create column taking the mean of each row (trial)
yaRTs['SEM'] = yaRTs.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
## Plot data
plt.figure() # reset
plt.plot(trials, oaRTs['Mean'].values, color='b', label='older adults')
plt.errorbar(trials, oaRTs['Mean'].values, yerr=oaRTs['SEM'].values, fmt='.b', ecolor='b', elinewidth=0.5)
plt.plot(trials, yaRTs['Mean'].values, color='r', label='younger adults')
plt.errorbar(trials, yaRTs['Mean'].values, yerr=yaRTs['SEM'].values, fmt='.r', ecolor='r', elinewidth=0.5)
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Response time (ms)')
plt.title('Comparing change in average RTs of older adults vs. younger adults')
plt.legend()
figAll = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figAll.savefig(saveLoc + '/rtChangeAll.png', bbox_inches='tight')


# Change in correct vs. incorrect vs. miss rates (per trial, across all participants, separate age groups)
## OA
### Calculate means and SEM per trial
#### success rate
corrRateOA['Mean'] = corrRateOA.mean(axis=1) # Create column taking the mean of each row (trial)
corrRateOA['SEM'] = corrRateOA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
#### incorrect rate
incRateOA['Mean'] = incRateOA.mean(axis=1) # Create column taking the mean of each row (trial)
incRateOA['SEM'] = incRateOA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
#### miss rate
missRateOA['Mean'] = missRateOA.mean(axis=1) # Create column taking the mean of each row (trial)
missRateOA['SEM'] = missRateOA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
### Plot data
plt.figure() # reset
plt.plot(trials, corrRateOA['Mean'].values, color='g', label='correct')
plt.errorbar(trials, corrRateOA['Mean'].values, yerr=corrRateOA['SEM'].values, fmt='.g', ecolor='g', elinewidth=0.5) # success rates
plt.plot(trials, incRateOA['Mean'].values, color='r', label='incorrect')
plt.errorbar(trials, incRateOA['Mean'].values, yerr=incRateOA['SEM'].values, fmt='.r', ecolor='r', elinewidth=0.5) # incorrect rates
plt.plot(trials, missRateOA['Mean'].values, color='b', label='miss')
plt.errorbar(trials, missRateOA['Mean'].values, yerr=missRateOA['SEM'].values, fmt='.b', ecolor='b', elinewidth=0.5) # miss rates
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Probability')
plt.yticks([0, 0.5, 1.0])
plt.title('Change in average correct/incorrect/miss rates of older adults')
plt.legend()
figHitOA = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figHitOA.savefig(saveLoc + '/hitRatesOA.png', bbox_inches='tight')

## YA
### Calculate means and SEM per trial
#### success rate
corrRateYA['Mean'] = corrRateYA.mean(axis=1) # Create column taking the mean of each row (trial)
corrRateYA['SEM'] = corrRateYA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
#### incorrect rate
incRateYA['Mean'] = incRateYA.mean(axis=1) # Create column taking the mean of each row (trial)
incRateYA['SEM'] = incRateYA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
#### miss rate
missRateYA['Mean'] = missRateYA.mean(axis=1) # Create column taking the mean of each row (trial)
missRateYA['SEM'] = missRateYA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
### Plot data
plt.figure() # reset
plt.plot(trials, corrRateYA['Mean'].values, color='g', label='correct')
plt.errorbar(trials, corrRateYA['Mean'].values, yerr=corrRateYA['SEM'].values, fmt='.g', ecolor='g', elinewidth=0.5) # success rates
plt.plot(trials, incRateYA['Mean'].values, color='r', label='incorrect')
plt.errorbar(trials, incRateYA['Mean'].values, yerr=incRateYA['SEM'].values, fmt='.r', ecolor='r', elinewidth=0.5) # incorrect rates
plt.plot(trials, missRateYA['Mean'].values, color='b', label='miss')
plt.errorbar(trials, missRateYA['Mean'].values, yerr=missRateYA['SEM'].values, fmt='.b', ecolor='b', elinewidth=0.5) # miss rates
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Probability')
plt.yticks([0, 0.5, 1.0])
plt.title('Change in average correct/incorrect/miss rates of younger adults')
plt.legend()
figHitYA = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figHitYA.savefig(saveLoc + '/hitRatesYA.png', bbox_inches='tight')


## Comparing patternAware vs patternUnaware RTs (per trial, pattAw vs pattUn, separate age groups)
### OA
#### Get means and SEM
##### patternAware
pattAwOA['Mean'] = pattAwOA.mean(axis=1) # Create column taking the mean of each row (trial)
pattAwOA['SEM'] = pattAwOA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
#### patternUnaware
pattUnOA['Mean'] = pattUnOA.mean(axis=1) # Create column taking the mean of each row (trial)
pattUnOA['SEM'] = pattUnOA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
### Plot data
plt.figure() # reset
plt.plot(trials, pattAwOA['Mean'].values, color='b', label='Aware')
plt.errorbar(trials, pattAwOA['Mean'].values, yerr=pattAwOA['SEM'].values, fmt='.b', elinewidth=0.5) # aware of pattern
plt.plot(trials, pattUnOA['Mean'].values, color='r', label='Unaware')
plt.errorbar(trials, pattUnOA['Mean'].values, yerr=pattUnOA['SEM'].values, fmt='.r', elinewidth=0.5) # unaware/unsure of pattern
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Response time (ms)')
plt.title('Comparing change in average response time of older adults aware of pattern and older adults unaware/unsure')
plt.legend()
figAwareOA = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figAwareOA.savefig(saveLoc + '/awarenessOA.png', bbox_inches='tight')

### YA
#### Get means and SEM
##### patternAware
pattAwYA['Mean'] = pattAwYA.mean(axis=1) # Create column taking the mean of each row (trial)
pattAwYA['SEM'] = pattAwYA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
#### patternUnaware
pattUnYA['Mean'] = pattUnYA.mean(axis=1) # Create column taking the mean of each row (trial)
pattUnYA['SEM'] = pattUnYA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
### Plot data
plt.figure() # reset
plt.plot(trials, pattAwYA['Mean'].values, color='b', label='Aware')
plt.errorbar(trials, pattAwYA['Mean'].values, yerr=pattAwYA['SEM'].values, fmt='.b', elinewidth=0.5) # aware of pattern
plt.plot(trials, pattUnYA['Mean'].values, color='r', label='Unaware')
plt.errorbar(trials, pattUnYA['Mean'].values, yerr=pattUnYA['SEM'].values, fmt='.r', elinewidth=0.5) # unaware/unsure of pattern
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Response time (ms)')
plt.title('Comparing change in average response time of younger adults aware of pattern and younger adults unaware/unsure')
plt.legend()
figAwareYA = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figAwareYA.savefig(saveLoc + '/awarenessYA.png', bbox_inches='tight')


# Difference in average response time of last 5 training blocks vs. 5 test blocks (trials vs test, all participants, separate age groups)
## From RTs dataframes, concatenate data
oaLearning = [oaRTs['Mean'][-10:-5].values, oaRTs['Mean'][-5:].values]
yaLearning = [yaRTs['Mean'][-10:-5].values, yaRTs['Mean'][-5:].values]
## Plot data
### from: https://www.geeksforgeeks.org/how-to-create-boxplots-by-group-in-matplotlib/
learnTicks = ['last 5 training blocks', 'test blocks'] # define xticks
### Create separate boxplots for arrays
oaLearnPlot = plt.boxplot(oaLearning, positions=np.array(np.arange(len(oaLearning)))*2.0-0.35, widths=0.6)
yaLearnPlot = plt.boxplot(yaLearning, positions=np.array(np.arange(len(yaLearning)))*2.0+0.35, widths=0.6)
# setting colors for each groups
define_box_properties(oaLearnPlot, '#2C7BB6', 'older adults')
define_box_properties(yaLearnPlot, '#D7191C', 'younger adults')
# set the x label values
plt.xticks(np.arange(0, len(learnTicks) * 2, 2), learnTicks)
# set the limit for x axis
plt.xlim(-2, len(learnTicks)*2)
# Set axes labels and title
plt.xlabel('Phase')
plt.ylabel('Response time (ms)')
plt.title('Average RT of last 5 train blocks vs. test blocks')
figLast = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figLast.savefig(saveLoc + '/learningAll.png', bbox_inches='tight')


# Difference in average miss rates of last 5 training blocks vs. 5 test blocks (trials vs test, all participants, separate age groups)
## From missRate dataframes, concatenate data
oaMiss = [missRateOA['Mean'][-10:-5].values, missRateOA['Mean'][-5:].values]
yaMiss = [missRateYA['Mean'][-10:-5].values, missRateYA['Mean'][-5:].values]
## Plot data
### from: https://www.geeksforgeeks.org/how-to-create-boxplots-by-group-in-matplotlib/
missTicks = ['last 5 training blocks', 'test blocks'] # define xticks
### Create separate boxplots for arrays
oaMissPlot = plt.boxplot(oaMiss, positions=np.array(np.arange(len(oaMiss)))*2.0-0.35, widths=0.6)
yaMissPlot = plt.boxplot(yaMiss, positions=np.array(np.arange(len(yaMiss)))*2.0+0.35, widths=0.6)
# setting colors for each groups
define_box_properties(oaMissPlot, '#2C7BB6', 'older adults')
define_box_properties(yaMissPlot, '#D7191C', 'younger adults')
# set the x label values
plt.xticks(np.arange(0, len(missTicks) * 2, 2), missTicks)
# set the limit for x axis
plt.xlim(-2, len(missTicks)*2)
# Set axes labels and title
plt.xlabel('Phase')
plt.ylabel('Probability')
plt.title('Average RT of last 5 train blocks vs. test blocks')
figMiss = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figMiss.savefig(saveLoc + '/missRatesAll.png', bbox_inches='tight')


# Change in rate of rushed response (rt < 500ms) (per trial, all participants, separate age groups)
## OA
### Calculate means and SEM
rushProbOA['Mean'] = rushProbOA.mean(axis=1) # Create column taking the mean of each row (trial)
rushProbOA['SEM'] = rushProbOA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
### Plot
plt.figure() # reset
plt.plot(trials, rushProbOA['Mean'].values, color = 'red')
plt.errorbar(trials, rushProbOA['Mean'].values, yerr = rushProbOA['SEM'].values, fmt='.r', elinewidth=0.5)
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Probability')
plt.yticks([0, 0.5, 1.0])
plt.title('Change in rate of rushed response (RT<500ms) in older adults')
figRushOA = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figRushOA.savefig(saveLoc + '/rushProbOA.png', bbox_inches='tight')

## YA
### Calculate means and SEM
rushProbYA['Mean'] = rushProbYA.mean(axis=1) # Create column taking the mean of each row (trial)
rushProbYA['SEM'] = rushProbYA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
### Plot
plt.figure() # reset
plt.plot(trials, rushProbYA['Mean'].values, color = 'red')
plt.errorbar(trials, rushProbYA['Mean'].values, yerr = rushProbYA['SEM'].values, fmt='.r', elinewidth=0.5)
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Probability')
plt.yticks([0, 0.5, 1.0])
plt.title('Change in rate of rushed response (RT<500ms) in younger adults')
figRushYA = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figRushYA.savefig(saveLoc + '/rushProbYA.png', bbox_inches='tight')


# Change in reward rate (correct response & rt<500ms) (per trial, all participants, OA vs. YA)
## Calculate means and SEMs
rewardRateOA['Mean'] = rewardRateOA.mean(axis=1) # Create column taking the mean of each row (trial)
rewardRateOA['SEM'] = rewardRateOA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
rewardRateYA['Mean'] = rewardRateYA.mean(axis=1) # Create column taking the mean of each row (trial)
rewardRateYA['SEM'] = rewardRateYA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
### Plot
plt.figure() # reset
plt.plot(trials, rewardRateOA['Mean'].values, color = 'blue', label='older adults')
plt.errorbar(trials, rewardRateOA['Mean'].values, yerr = rewardRateOA['SEM'].values, fmt='.b', elinewidth=0.5)
plt.plot(trials, rewardRateYA['Mean'].values, color = 'red', label='younger adults')
plt.errorbar(trials, rewardRateYA['Mean'].values, yerr = rewardRateYA['SEM'].values, fmt='.r', elinewidth=0.5)
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Probability')
plt.yticks([0, 0.5, 1.0])
plt.title('Comparing change in reward rate (correct response & RT<500ms) between older and younger adults')
plt.legend()
figReward = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figReward.savefig(saveLoc + '/rewardAll.png', bbox_inches='tight')


# Change in punishment rate (incorrect response or rt>=1000ms) (per trial, all participants, separate age groups)
## Calculate means and SEM
punishRateOA['Mean'] = punishRateOA.mean(axis=1) # Create column taking the mean of each row (trial)
punishRateOA['SEM'] = punishRateOA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
punishRateYA['Mean'] = punishRateYA.mean(axis=1) # Create column taking the mean of each row (trial)
punishRateYA['SEM'] = punishRateYA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
## Plot
plt.figure() # reset
plt.plot(trials, punishRateOA['Mean'].values, color = 'blue', label = 'older adults')
plt.errorbar(trials, punishRateOA['Mean'].values, yerr = punishRateOA['SEM'].values, fmt='.b', elinewidth=0.5)
plt.plot(trials, punishRateYA['Mean'].values, color = 'red', label='younger adults')
plt.errorbar(trials, punishRateYA['Mean'].values, yerr = punishRateYA['SEM'].values, fmt='.r', elinewidth=0.5)
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Probability')
plt.yticks([0, 0.5, 1.0])
plt.title('Comparing change in punishment rate (incorrect response OR RT>=1000ms) between older and younger adults')
plt.legend()
figPunish = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figPunish.savefig(saveLoc + '/punishAll.png', bbox_inches='tight')

#############
# patternAware Data Analyses

# Improvement Rate (difference of average of T1-3 and average of T28-30) (select trials, all participants, OA vs. YA boxplot)
## Concatenate arrays
improveAll = [improveOA, improveYA]
## Plot
plt.figure() # reset
plt.boxplot(improveAll)
plt.xticks([1,2], ['OA', 'YA'])
plt.ylabel('Response time (ms)')
plt.title('Improvement over training of older vs. younger adults')
figImpRate = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figImpRate.savefig(saveLoc + '/improvementAll.png', bbox_inches='tight')


# Amount of Regression (amount of sequence-specific learning) (select trials, all participants, OA vs. YA boxplot)
## Concatenate arrays
regressAll = [regressOA, regressYA]
## Plot
plt.figure() # reset
plt.boxplot(regressAll)
plt.xticks([1,2], ['OA', 'YA'])
plt.ylabel('Response time (ms)')
plt.title('Amount of regression of older vs. younger adults')
figRegress = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figRegress.savefig(saveLoc + '/regressionAll.png', bbox_inches='tight')


# Learning rates (slopes of first five train blocks vs. slopes of test blocks) (select trials, all participants, OA vs. YA boxplots)
## Concatenate data
oaSlopes = [trainSlopesOA, testSlopesOA]
yaSlopes = [trainSlopesYA, testSlopesYA]
## Plot data
### from: https://www.geeksforgeeks.org/how-to-create-boxplots-by-group-in-matplotlib/
lrTicks = ['initial training', 'test'] # define xticks
### Create separate boxplots for arrays
oaPlot = plt.boxplot(oaSlopes, positions=np.array(np.arange(len(oaSlopes)))*2.0-0.35, widths=0.6)
yaPlot = plt.boxplot(yaSlopes, positions=np.array(np.arange(len(yaSlopes)))*2.0+0.35, widths=0.6)
# setting colors for each groups
define_box_properties(oaPlot, '#2C7BB6', 'older adults')
define_box_properties(yaPlot, '#D7191C', 'younger adults')
# set the x label values
plt.xticks(np.arange(0, len(lrTicks) * 2, 2), lrTicks)
# set the limit for x axis
plt.xlim(-2, len(lrTicks)*2)
# Set axes labels and title
plt.xlabel('Phase')
plt.ylabel('Learning rate (difference in response times)')
plt.title('Learning rates of older and younger adults')
figLearnRate = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figLearnRate.savefig(saveLoc + '/learnRateAll.png', bbox_inches='tight')


# When learning stabilizes (i.e. when correct rate doesn't improve more) (all trials, all participants, OA vs. YA)
## Get means and SEM
stabilizeOA['Mean'] = stabilizeOA.mean(axis=1) # Create column taking the mean of each row (trial)
stabilizeOA['SEM'] = stabilizeOA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
stabilizeYA['Mean'] = stabilizeYA.mean(axis=1) # Create column taking the mean of each row (trial)
stabilizeYA['SEM'] = stabilizeYA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
## Plot data
plt.figure() # reset
plt.plot(np.arange(1,35), stabilizeOA['Mean'].values, color = 'blue', label = 'older adults')
plt.errorbar(np.arange(1,35), stabilizeOA['Mean'].values, yerr = stabilizeOA['SEM'].values, fmt='.b', elinewidth=0.5)
plt.plot(np.arange(1,35), stabilizeYA['Mean'].values, color = 'red', label='younger adults')
plt.errorbar(np.arange(1,35), stabilizeYA['Mean'].values, yerr = stabilizeYA['SEM'].values, fmt='.r', elinewidth=0.5)
plt.xlabel('Transition')
plt.xticks(trials)
plt.ylabel('Improvement')
plt.yticks([0, 0.5, 1.0])
plt.title('When learning stabilizes (i.e. improvement asymptotes) in older vs. younger adults')
plt.legend()
figStab = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figStab.savefig(saveLoc + '/stabilizeAll.png', bbox_inches='tight')


# Change in RT variability (standard deviation) within each trial (all trials, all participants, OA vs. YA)
## Get means and SEM
variabilityOA['Mean'] = variabilityOA.mean(axis=1) # Create column taking the mean of each row (trial)
variabilityOA['SEM'] = variabilityOA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
variabilityYA['Mean'] = variabilityYA.mean(axis=1) # Create column taking the mean of each row (trial)
variabilityYA['SEM'] = variabilityYA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
## Plot
plt.figure() # reset
plt.plot(trials, variabilityOA['Mean'].values, color = 'blue', label = 'older adults')
plt.errorbar(trials, variabilityOA['Mean'].values, yerr = variabilityOA['SEM'].values, fmt='.b', elinewidth=0.5)
plt.plot(trials, variabilityYA['Mean'].values, color = 'red', label='younger adults')
plt.errorbar(trials, variabilityYA['Mean'].values, yerr = variabilityYA['SEM'].values, fmt='.r', elinewidth=0.5)
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Average standard deviation')
plt.title('Variability of response times per trial of older vs. younger adults')
plt.legend()
figVar = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figVar.savefig(saveLoc + '/variabilityAll.png', bbox_inches='tight')

#############
# Save survey data
surveyDF = surveyDF.transpose() # transpose for easier viewing
surveyDF['Participant ID'] = parID # add participant IDs
surveyDF.to_csv(saveLoc + '/allSurveyData.csv', index=False)

# Statistical test - repeated measures ANOVA
## dv = rt, subjects = participants, within = phase, between = ageGroup
## Create dataframe
statsDF = pd.DataFrame({'Participant':participants, 'RT':rtData, 'Phase':phase, 'Age Group':ageGroup})
statsDF.to_csv(saveLoc+'/allStatsData.csv', index=False)

## dv = miss rates, subjects = participants, within = phase, between = ageGroup
## Create dataframe
missStats = pd.DataFrame({'Participant':participants, 'Miss Rate':missRates, 'Phase':phase, 'Age Group':ageGroup})
missStats.to_csv(saveLoc+'/missStatsData.csv', index=False)