# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:26:11 2024

@author: joaquinmtorres

Script to find the accuracy/proportion correct of each aware participant's 
reported sequence, for the 6-item explicit sequence (729183), assuming 
sequenceResponses.csv contains survey data of all patternAware participants,
with each survey_order response being cleaned/standardized.
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Define possible sequences
possible = [729183, 291837, 918372, 183729, 837291, 372918]

# Create empty arrays where each ratio per age group will be stored    
reportPropYA = [] # proportion of sequence reported matching actual sequence
reportPropOA = [] # proportion of sequence reported matching actual sequence

# Define file location (where reported sequences are) and take array of responses
filePath = '/Users/joaqu/OneDrive/Documents/Bates/Kim Lab/dataFiles/20240815 Explicit/'
reportsFile = pd.read_csv(filePath + 'sequenceResponses.csv')

# Get ratios correct
## Loop through each report
for index, row in reportsFile.iterrows():
    ratios = [] # gets ratios for each possible sequence
    # Loop through each possible sequence (from ChatGPT)
    for j in possible:
        report = reportsFile['survey_order'][index] 
        length = min(len(str(report)), len(str(j))) # Ensure report and sequence (j) are the same length for a fair comparison
        matches = sum(1 for k in range(length) if str(report)[k] == str(j)[k]) # Count the number of matching characters
        proportion = matches/len(str(report)) if len(str(report)) > 0 else 0 # Calculate the proportion
        ratios.append(proportion) # add proportion to the ratios array
    # Sorting
    if reportsFile['age_dropdown'][index] >= 65:
        reportPropOA.append(max(ratios)) # appends the highest possible ratio
    else:
        reportPropYA.append(max(ratios)) # appends the highest possible ratio

# Plot data
## Concatenate arrays
proportionsAll = [reportPropOA, reportPropYA]
## Plot
plt.figure() # reset
plt.boxplot(proportionsAll)
plt.xticks([1,2], ['OA', 'YA'])
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.ylabel('Proportion correct')
plt.title('Proportion correct of reported sequences in older vs. younger adults')
figProps = plt.gcf()
plt.show(block=False)
plt.pause(2)
plt.close()
figProps.savefig(filePath + '/proportions.png', bbox_inches='tight')