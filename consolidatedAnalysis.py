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
    - learning rate
    - improvement rate
    - amount of regression
    - how/when learning stabilizes
    - response time variability
    - response time counts
    - dataframes for stats test

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
missRateOAAw = pd.DataFrame(index=trials) # average miss rates per trial for seqAware
missRateOAUn = pd.DataFrame(index=trials) # average miss rates per trial for seqUnaware
incRateOA = pd.DataFrame(index=trials) # average incorrect rates (wrong key press within time limit) per trial
incRateOAAw = pd.DataFrame(index=trials) # average incorrect rates per trial for seqAware
incRateOAUn = pd.DataFrame(index=trials) # average incorrect rates per trial for seqUnaware
pattAwOA = pd.DataFrame(index=trials)  # rts of participants who responded aware of a sequence
pattUnOA = pd.DataFrame(index=trials) # rts of participants who responded unaware or unsure of a sequence
rushProbOA = pd.DataFrame(index=trials) # probability of rushed response (rt < 500 ms) per trial
rewardRateOA = pd.DataFrame(index=trials) # probability of reward (ding) per trial
punishRateOA = pd.DataFrame(index=trials) # probability punished per trial
trainSlopesNanOA = [] # learning rate omitting incorrect/miss responses (all awareness groups)
testSlopesNanOA = [] # learning rate omitting incorrect/miss responses (all awareness groups)
## YA Data
yaRTs = pd.DataFrame(index=trials) # for response times (same figure with oaRTs)
corrRateYA = pd.DataFrame(index=trials) # average success rates per trial
missRateYA = pd.DataFrame(index=trials) # average miss rates (RT >= 1000) per trial
missRateYAAw = pd.DataFrame(index=trials) # average miss rates per trial for seqAware
missRateYAUn = pd.DataFrame(index=trials) # average miss rates per trial for seqUnaware
incRateYA = pd.DataFrame(index=trials) # average incorrect rates (wrong key press within time limit) per trial
incRateYAAw = pd.DataFrame(index=trials) # average incorrect rates per trial for seqAware
incRateYAUn = pd.DataFrame(index=trials) # average incorrect rates per trial for seqUnaware
pattAwYA = pd.DataFrame(index=trials)  # rts of participants who responded aware of a sequence
pattUnYA = pd.DataFrame(index=trials) # aggregated rts of participants who responded unaware or unsure of a sequence
rushProbYA = pd.DataFrame(index=trials) # probability of rushed response (rt < 500 ms) per trial
rewardRateYA = pd.DataFrame(index=trials) # probability of reward (ding) per trial
punishRateYA = pd.DataFrame(index=trials) # probability punished per trial
trainSlopesNanYA = [] # learning rate omitting incorrect/miss responses (all awareness groups)
testSlopesNanYA = [] # learning rate omitting incorrect/miss responses (all awareness groups)

# Set empty arrays/dataframes for additional patternAware data analyses
## OA
improveOA = [] # improvement rates (average of T1-3 - average of T28-30)
regressOA = [] # regression (difference of T31-T30/difference of T1-T30)
trainSlopesOA = [] # learning rate (average difference of slopes of T1-5)
testSlopesOA = [] # learning rate (average difference of slopes of T31-35)
stabilizeOA = pd.DataFrame(index=np.arange(1,35)) # when learning stabilizes (i.e. when there is little to no difference in improvement in terms of correct rate); index=each transition
variabilityOA = pd.DataFrame(index=trials) # change in RT variability within each trial
rtAllOA = [] # rt count (histogram)
## YA
improveYA = [] # improvement rates (average of T1-3 - average of T28-30)
regressYA = [] # regression (difference of T31-T30/difference of T1-T30)
trainSlopesYA = [] # learning rate (average difference of slopes of T1-5)
testSlopesYA = [] # learning rate (average difference of slopes of T31-35)
stabilizeYA = pd.DataFrame(index=np.arange(1,35)) # when learning stabilizes (i.e. when there is little to no difference in improvement in terms of correct rate); index=each transition
variabilityYA = pd.DataFrame(index=trials) # change in RT variability within each trial
rtAllYA = [] # rt count (histogram)

# Set empty arrays for stats test dataframes
## General stats test
parIDs = [] # participant ID (each participant will have 35 rows corresponding to each trial)
ageArr = [] # age
genderArr = [] # gender
awareArr = [] # sequence awareness
rtData = [] # rt
corrRates = [] # correct rate
incRates = [] # incorrect rate
missRates = [] # miss rate
rewRates = [] # reward rate
punRates = [] # punishment rate
resVars = [] # response variability
triNum = [] # trial number
rtDataNan = [] # rt data with inc/miss responses converted to NaN
resVarsNan = [] # response variability with inc/miss responses converted to NaN
## Learning rate stats test
parIDsLR = [] # participant ID (each will have 2 rows representing each phase - initial and test)
lr = [] # learning rate
ageLR = [] # age
genderLR = [] # gender
awareLR = [] # sequence awareness
phaseLR = [] # phase (either initial or test)

# Correct rates separating aware and unaware and by age group
corrAwOA = pd.DataFrame(index=trials)
corrUnOA = pd.DataFrame(index=trials)
corrAwYA = pd.DataFrame(index=trials)
corrUnYA = pd.DataFrame(index=trials)
# Correct rates separating awareness (but not age groups)
corrAware = pd.DataFrame(index=trials)
corrUnaware = pd.DataFrame(index=trials)

#############

# Loop through each data file
for file in dirList:
    df = pd.read_csv(file)
    
    # Set file names and save directories
    fileName = file.split('/')[-1:][0].split('\\')[1].split('.')[0]
    saveDir = os.path.join(saveLoc, fileName)
    parID.append(fileName) # save participant id 
        
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
    if surveyData['survey_awareness'] == 'awarenessYes':
        corrAware[fileName] = corrRatePerTrial
        if surveyData['age_dropdown'] >= 65:
            corrAwOA[fileName] = corrRatePerTrial
            missRateOAAw[fileName] = missRatePerTrial
            incRateOAAw[fileName] = incRatePerTrial
        else:
            corrAwYA[fileName] = corrRatePerTrial
            missRateYAAw[fileName] = missRatePerTrial
            incRateYAAw[fileName] = incRatePerTrial
    else:
        corrUnaware[fileName] = corrRatePerTrial
        if surveyData['age_dropdown'] >= 65:
            corrUnOA[fileName] = corrRatePerTrial
            missRateOAUn[fileName] = missRatePerTrial
            incRateOAUn[fileName] = incRatePerTrial
        else:
            corrUnYA[fileName] = corrRatePerTrial
            missRateYAUn[fileName] = missRatePerTrial
            incRateYAUn[fileName] = incRatePerTrial
    
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
            
        # RT Count histograms
        plt.figure() # reset
        plt.hist(rt)
        plt.xlabel('RT')
        plt.ylabel('Count')
        plt.title(f'Frequency of response times of each key press for {fileName}')
        figHist = plt.gcf()
        plt.close()
        figHist.savefig(saveDir + f'/rtCount {fileName}.png', bbox_inches='tight')
        # Store array to appropriate age group array
        if surveyData['age_dropdown'] >= 65:
            rtAllOA = rtAllOA + rt.tolist()
        else:
            rtAllYA = rtAllYA + rt.tolist()
            
    ############
    # Build up stats test dataframes
    
    # General
    ## Participant ID - append fileName 35x
    for aa in np.arange(0,35):
        parIDs.append(fileName)
    ## Age - append 35x
    for bb in np.arange(0,35):
        ageArr.append(surveyData['age_dropdown'])
    ## Gender - append 35x
    for cc in np.arange(0,35):
        genderArr.append(surveyData['gender'])
    ## Sequence Awareness - append 35x
    for dd in np.arange(0,35):
        awareArr.append(surveyData['survey_awareness'])
    ## RT - append list of average RT per trial
    rtData = rtData + aveRTList
    ## Correct rates - append list of average correct rate per trial
    corrRates = corrRates + corrRatePerTrial
    ## Incorrect rates - append list of average incorrect rate per trial
    incRates = incRates + incRatePerTrial
    ## Miss rates - append list of average miss rate per trial
    missRates = missRates + missRatePerTrial
    ## Reward rates - append list of average reward rate per trial
    rewRates = rewRates + dingRates
    ## Punishment rates - append list of average punishment rate per trial
    punRates = punRates + punishRates
    ## Response variability - append list of RT variability per trial
    sdPerTrialAll = [np.std(ff) for ff in rtPerTrial] # create array of standard deviation of each trial
    resVars = resVars + sdPerTrialAll
    ## Trial number - append list of trials (1-35)
    triNum = triNum + trials.tolist() 
    
    # General RT and Response variability, but converting inc/miss RTs to NaN
    ## from accuracyDF dataframe
    nanAccuracyDF = accuracyDF # duplicate
    nanAccuracyDF.loc[nanAccuracyDF['Accuracy'] != 'corr', 'RT'] = np.nan # convert inc/miss RTs to np.nan (from ChatGPT)
    rtListNan = nanAccuracyDF['RT'].tolist() # turn RT column to list
    rtPerTrialNan = list(divide_chunks(rtListNan, howMany)) # Divide into trials
    aveRTListNan = [] # set empty array
    sdPerTrialNan = [] # set empty array
    # loop through each trial and get the mean. If all items in the trial are np.nan, set the mean and sd to np.nan (from ChatGPT)
    for ee in rtPerTrialNan:
        if np.all(np.isnan(ee)):
            aveRTListNan.append(np.nan)
            sdPerTrialNan.append(np.nan)
        else:
            aveRTListNan.append(np.nanmean(ee))
            sdPerTrialNan.append(np.nanstd(ee))
    ## Store into appropriate empty arrays
    rtDataNan = rtDataNan + aveRTListNan
    resVarsNan = resVarsNan + sdPerTrialNan
    
    # LR Stats
    ## Participant ID - append fileName 2x
    parIDsLR.append(fileName)
    parIDsLR.append(fileName)
    ## Learning rate (ave. slopes of first 5 train blocks vs. ave. slopes of test blocks)
    aveTrainSlopesLR = np.nanmean(np.diff(aveRTList[:5])) # Get average slope of first 5 training blocks
    aveTestSlopesLR = np.nanmean(np.diff(aveRTList[-5:])) # Get average slope of test blocks
    lr.append(aveTrainSlopesLR)
    lr.append(aveTestSlopesLR)
    ## Age - append 2x
    ageLR.append(surveyData['age_dropdown'])
    ageLR.append(surveyData['age_dropdown'])
    ## Gender - append 2x
    genderLR.append(surveyData['gender'])
    genderLR.append(surveyData['gender'])
    ## Sequence awareness - append 2x
    awareLR.append(surveyData['survey_awareness'])
    awareLR.append(surveyData['survey_awareness'])
    ## Phase - append initial and test
    phaseLR.append('initial')
    phaseLR.append('test')
    
    ###########
    # Learning rate omitting incorrect and miss responses
    if surveyData['survey_awareness'] == 'awarenessYes':
        ## from aveRTListNan, get slopes
        aveTrainSlopesNan = np.nanmean(np.diff(aveRTListNan[:5])) # Get average slope of first 5 training blocks
        aveTestSlopesNan = np.nanmean(np.diff(aveRTListNan[-5:])) # Get average slope of test blocks
        ## Store into arrays
        if surveyData['age_dropdown'] >= 65:
            trainSlopesNanOA.append(aveTrainSlopesNan)
            testSlopesNanOA.append(aveTestSlopesNan)
        else:
            trainSlopesNanYA.append(aveTrainSlopesNan)
            testSlopesNanYA.append(aveTestSlopesNan)
       
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
plt.title('Average miss rates of last 5 train blocks vs. test blocks')
figMiss = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figMiss.savefig(saveLoc + '/missRatesAll.png', bbox_inches='tight')


# Change in rate of rushed response (rt < 500ms) (per trial, all participants, OA vs. YA)
## Calculate means and SEM
rushProbOA['Mean'] = rushProbOA.mean(axis=1) # Create column taking the mean of each row (trial)
rushProbOA['SEM'] = rushProbOA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
rushProbYA['Mean'] = rushProbYA.mean(axis=1) # Create column taking the mean of each row (trial)
rushProbYA['SEM'] = rushProbYA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
## Plot
plt.figure() # reset
plt.plot(trials, rushProbOA['Mean'].values, color = 'blue', label='older adults')
plt.errorbar(trials, rushProbOA['Mean'].values, yerr = rushProbOA['SEM'].values, fmt='.b', elinewidth=0.5)
plt.plot(trials, rushProbYA['Mean'].values, color = 'red', label='younger adults')
plt.errorbar(trials, rushProbYA['Mean'].values, yerr = rushProbYA['SEM'].values, fmt='.r', elinewidth=0.5)
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Probability')
plt.yticks([0, 0.5, 1.0])
plt.title('Change in rate of rushed response (RT<500ms) in older vs. younger adults')
plt.legend()
figRush = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figRush.savefig(saveLoc + '/rushProbAll.png', bbox_inches='tight')


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
plt.figure() # reset
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


# RT Count histograms (all trials, all participants, separate age groups)
## OA
plt.figure() # reset
plt.hist(rtAllOA)
plt.xlabel('RT')
plt.ylabel('Count')
plt.title('Frequency of response times of each key press in older adults')
figHistOA = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figHistOA.savefig(saveLoc + '/rtCountOA.png', bbox_inches='tight')
# YA
plt.figure() # reset
plt.hist(rtAllYA)
plt.xlabel('RT')
plt.ylabel('Count')
plt.title('Frequency of response times of each key press in younger adults')
figHistYA = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figHistYA.savefig(saveLoc + '/rtCountYA.png', bbox_inches='tight')


# Learning rates, omitting incorrect/miss responses (select trials, all participants, OA vs. YA boxplots)
plt.figure() # reset
## Concatenate data
oaSlopesNan = [np.array(trainSlopesNanOA)[np.isnan(np.array(trainSlopesNanOA))==False], np.array(testSlopesNanOA)[np.isnan(np.array(testSlopesNanOA))==False]]
yaSlopesNan = [np.array(trainSlopesNanYA)[np.isnan(np.array(trainSlopesNanYA))==False], np.array(testSlopesNanYA)[np.isnan(np.array(testSlopesNanYA))==False]]
## Plot data
### from: https://www.geeksforgeeks.org/how-to-create-boxplots-by-group-in-matplotlib/
lrNanTicks = ['initial training', 'test'] # define xticks
### Create separate boxplots for arrays
oaNanPlot = plt.boxplot(oaSlopesNan, positions=np.array(np.arange(len(oaSlopesNan)))*2.0-0.35, widths=0.6)
yaNanPlot = plt.boxplot(yaSlopesNan, positions=np.array(np.arange(len(yaSlopesNan)))*2.0+0.35, widths=0.6)
# setting colors for each groups
define_box_properties(oaNanPlot, '#2C7BB6', 'older adults')
define_box_properties(yaNanPlot, '#D7191C', 'younger adults')
# set the x label values
plt.xticks(np.arange(0, len(lrNanTicks) * 2, 2), lrNanTicks)
# set the limit for x axis
plt.xlim(-2, len(lrNanTicks)*2)
# Set axes labels and title
plt.xlabel('Phase')
plt.ylabel('Learning rate (difference in response times)')
plt.title('Learning rates of older and younger adults, omitting incorrect/miss responses')
figLRNan = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figLRNan.savefig(saveLoc + '/learnRateNanAll.png', bbox_inches='tight')


# Correct rates separating aware and unaware and by age group
plt.figure() # reset
## Get means and SEM
corrAwOA['Mean'] = corrAwOA.mean(axis=1) # Create column taking the mean of each row (trial)
corrAwOA['SEM'] = corrAwOA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
corrUnOA['Mean'] = corrUnOA.mean(axis=1) # Create column taking the mean of each row (trial)
corrUnOA['SEM'] = corrUnOA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
corrAwYA['Mean'] = corrAwYA.mean(axis=1) # Create column taking the mean of each row (trial)
corrAwYA['SEM'] = corrAwYA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
corrUnYA['Mean'] = corrUnYA.mean(axis=1) # Create column taking the mean of each row (trial)
corrUnYA['SEM'] = corrUnYA.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
## Plot data
plt.plot(trials, corrAwOA['Mean'].values, color = 'blue', label = 'older adults aware')
plt.errorbar(trials, corrAwOA['Mean'].values, yerr = corrAwOA['SEM'].values, fmt='.b', elinewidth=0.5)
plt.plot(trials, corrUnOA['Mean'].values, color = 'c', label='older adults unaware')
plt.errorbar(trials, corrUnOA['Mean'].values, yerr = corrUnOA['SEM'].values, fmt='.c', elinewidth=0.5)
plt.plot(trials, corrAwYA['Mean'].values, color = 'red', label = 'younger adults aware')
plt.errorbar(trials, corrAwYA['Mean'].values, yerr = corrAwYA['SEM'].values, fmt='.r', elinewidth=0.5)
plt.plot(trials, corrUnYA['Mean'].values, color = 'm', label='younger adults unaware')
plt.errorbar(trials, corrUnYA['Mean'].values, yerr = corrUnYA['SEM'].values, fmt='.m', elinewidth=0.5)
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Average correct rate')
plt.title('Change in correct rate per trial of older vs. younger adults, aware vs. unaware of pattern')
plt.legend()
figCorrAwAge = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figCorrAwAge.savefig(saveLoc + '/correctAwarenessAge.png', bbox_inches='tight')


# Correct rates separating aware and unaware (but not age groups)
plt.figure() # reset
## Get means and SEM
corrAware['Mean'] = corrAware.mean(axis=1) # Create column taking the mean of each row (trial)
corrAware['SEM'] = corrAware.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
corrUnaware['Mean'] = corrUnaware.mean(axis=1) # Create column taking the mean of each row (trial)
corrUnaware['SEM'] = corrUnaware.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
## Plot data
plt.plot(trials, corrAware['Mean'].values, color = 'blue', label = 'aware')
plt.errorbar(trials, corrAware['Mean'].values, yerr = corrAware['SEM'].values, fmt='.b', elinewidth=0.5)
plt.plot(trials, corrUnaware['Mean'].values, color = 'red', label = 'unaware')
plt.errorbar(trials, corrUnaware['Mean'].values, yerr = corrUnaware['SEM'].values, fmt='.r', elinewidth=0.5)
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Average correct rate')
plt.title('Change in correct rate per trial of aware vs. unaware of pattern')
plt.legend()
figCorrAw = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figCorrAw.savefig(saveLoc + '/correctAwarenessAll.png', bbox_inches='tight')


# Miss and incorrect rates separating age groups and awareness
plt.figure() #reset
## Get means and SEM
missRateOAAw['Mean'] = missRateOAAw.mean(axis=1) # Create column taking the mean of each row (trial)
missRateOAAw['SEM'] = missRateOAAw.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
missRateYAAw['Mean'] = missRateYAAw.mean(axis=1) # Create column taking the mean of each row (trial)
missRateYAAw['SEM'] = missRateYAAw.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
missRateOAUn['Mean'] = missRateOAUn.mean(axis=1) # Create column taking the mean of each row (trial)
missRateOAUn['SEM'] = missRateOAUn.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
missRateYAUn['Mean'] = missRateYAUn.mean(axis=1) # Create column taking the mean of each row (trial)
missRateYAUn['SEM'] = missRateYAUn.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
incRateOAAw['Mean'] = incRateOAAw.mean(axis=1) # Create column taking the mean of each row (trial)
incRateOAAw['SEM'] = incRateOAAw.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
incRateYAAw['Mean'] = incRateYAAw.mean(axis=1) # Create column taking the mean of each row (trial)
incRateYAAw['SEM'] = incRateYAAw.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
incRateOAUn['Mean'] = incRateOAUn.mean(axis=1) # Create column taking the mean of each row (trial)
incRateOAUn['SEM'] = incRateOAUn.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
incRateYAUn['Mean'] = incRateYAUn.mean(axis=1) # Create column taking the mean of each row (trial)
incRateYAUn['SEM'] = incRateYAUn.iloc[:, :-1].sem(axis=1) # Create a column calculating the SEM of each row, not including the Means column
## Plot data
### Miss rates
plt.plot(trials, missRateOAAw['Mean'].values, color = 'blue', label = 'older adults aware')
plt.errorbar(trials, missRateOAAw['Mean'].values, yerr = missRateOAAw['SEM'].values, fmt='.b', elinewidth=0.5)
plt.plot(trials, missRateOAUn['Mean'].values, color = 'c', label='older adults unaware')
plt.errorbar(trials, missRateOAUn['Mean'].values, yerr = missRateOAUn['SEM'].values, fmt='.c', elinewidth=0.5)
plt.plot(trials, missRateYAAw['Mean'].values, color = 'red', label = 'younger adults aware')
plt.errorbar(trials, missRateYAAw['Mean'].values, yerr = missRateYAAw['SEM'].values, fmt='.r', elinewidth=0.5)
plt.plot(trials, missRateYAUn['Mean'].values, color = 'm', label='younger adults unaware')
plt.errorbar(trials, missRateYAUn['Mean'].values, yerr = missRateYAUn['SEM'].values, fmt='.m', elinewidth=0.5)
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Average miss rate')
plt.title('Change in miss rate per trial of older vs. younger adults, aware vs. unaware of pattern')
plt.legend()
figMissAwAge = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figMissAwAge.savefig(saveLoc + '/missAwarenessAge.png', bbox_inches='tight')
### Incorrect rates
plt.plot(trials, incRateOAAw['Mean'].values, color = 'blue', label = 'older adults aware')
plt.errorbar(trials, incRateOAAw['Mean'].values, yerr = incRateOAAw['SEM'].values, fmt='.b', elinewidth=0.5)
plt.plot(trials, incRateOAUn['Mean'].values, color = 'c', label='older adults unaware')
plt.errorbar(trials, incRateOAUn['Mean'].values, yerr = incRateOAUn['SEM'].values, fmt='.c', elinewidth=0.5)
plt.plot(trials, incRateYAAw['Mean'].values, color = 'red', label = 'younger adults aware')
plt.errorbar(trials, incRateYAAw['Mean'].values, yerr = incRateYAAw['SEM'].values, fmt='.r', elinewidth=0.5)
plt.plot(trials, incRateYAUn['Mean'].values, color = 'm', label='younger adults unaware')
plt.errorbar(trials, incRateYAUn['Mean'].values, yerr = incRateYAUn['SEM'].values, fmt='.m', elinewidth=0.5)
plt.xlabel('Trial Number')
plt.xticks(trials)
plt.ylabel('Average incorrect rate')
plt.title('Change in incorrect rate per trial of older vs. younger adults, aware vs. unaware of pattern')
plt.legend()
figIncAwAge = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figIncAwAge.savefig(saveLoc + '/incAwarenessAge.png', bbox_inches='tight')

#############
# Save survey data
surveyDF = surveyDF.transpose() # transpose for easier viewing
surveyDF['Participant ID'] = parID # add participant IDs
surveyDF.to_csv(saveLoc + '/allSurveyData.csv', index=False)

# Create dataframes for statistical tests
## General
generalStats = pd.DataFrame({'Participant ID':parIDs, 'Age':ageArr, 'Gender':genderArr, 'Awareness':awareArr, 'RT':rtData, 'RT w/ NaN':rtDataNan, 'Correct Rate':corrRates, 'Incorrect Rate':incRates, 'Miss Rate':missRates, 'Reward Rate':rewRates, 'Punishment Rate':punRates, 'RT Variability':resVars, 'RT Var. w/ NaN':resVarsNan, 'Trial #':triNum})
generalStats.to_csv(saveLoc + '/generalStatsData.csv', index=False)

## From General, only take each participant's trial 1, 30 and 31 data
### from: https://pandas.pydata.org/docs/dev/getting_started/intro_tutorials/03_subset_data.html
specificStats = generalStats[generalStats['Trial #'].isin([1,30,31])]
specificStats.to_csv(saveLoc + '/specificStatsData.csv', index=False)

## Learning rate
lrStats = pd.DataFrame({'Participant ID':parIDsLR, 'Learning Rate':lr, 'Age':ageLR, 'Gender':genderLR, 'Awareness':awareLR, 'Phase':phaseLR})
lrStats.to_csv(saveLoc + '/lrStatsData.csv', index=False)