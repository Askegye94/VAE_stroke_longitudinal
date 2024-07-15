# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:49:14 2022

@author: michi
"""
## PRE STEP STUFF ##    
windowLength = 200 # do not change this.
############################
## Some required settings ##
############################
inputColumns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
latentFeatures = 3 # 2 #  3 /     --> Sina is currently writing the paper visuals for 2 latentfeatures and validation also for 4 latentfeatures. (2 and 4?)
trainModel =  False #True #False

frequency = 50
original_frequency_w = 150
original_frequency_s = 100
original_frequency_l = 100

storingWeights = False #True   # true / false
############################
##### end of settings! #####  20 * 10 seconds = input data 200  --> more gait cycles included. 
############################  50 * 4 seconds = 200 samples. 


if frequency == 20:
    timeLength = 10
    refactorValue = 5 # taking each 5th value for resampling
else:
    timeLength = 4
    refactorValue = 2 


## paths to models / raw data / initialised weights / loaded weights 
pathToDataRelativeAngles =r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\Desktop\New Python Scripts\proximalangles\proximalangles\\'
pathToDataEvents =r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\Desktop\New Python Scripts\Events\Events\\'
pathToCleanData = r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\Desktop\New Python Scripts\3dPreparedData\\'
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.io as spio
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
plt.close('all')


############### STEP 1 ###############################
filename = []
data = []
group = []
#### THE TIMESERIES #####
if 'pathToDataRelativeAngles' in locals():
    for file1 in os.listdir(pathToDataRelativeAngles):
        if file1[0:4] != 'S017': #REMOVE THESE
            if file1[0:4] != 'S022':
                data.append(spio.loadmat(pathToDataRelativeAngles + file1))
                filename.append(file1)
                if file1[0]=='W':
                    group.append(int(file1[1:3]))
                else:
                    group.append(int(file1[1:4]))
            else:
                print(file1[0:4])
        else:
            print(file1[0:4])
            
group = np.array(group)
# get the dict out of the list
for numtrials in range(len(filename)):
    data[numtrials] = (data[numtrials]['prox_relativeangles'])#'markervel'  

#Scaling the data
scaler = MinMaxScaler()
data = [scaler.fit_transform(arr) for arr in data]
    

#%%

# First, find the index of the first file starting with 'W' or 'S'
start_w = None
start_s = None

for indx, name in enumerate(filename):
    if name[0] == 'W' and start_w is None:
        start_w = indx
    elif name[0] == 'S' and start_s is None:
        start_s = indx
    
    if start_w is not None and start_s is not None:
        break

# Calculate refactor_value and new_length based on the file prefix
refactor_value_w = original_frequency_w / frequency
refactor_value_s = original_frequency_s / frequency

# Initialize a list to store downsampled data
downsampled_data = []

# Iterate through data starting from the identified start index
for indx in range(len(data)):
    array = data[indx]  # Get the current array
    filename_prefix = filename[indx][0]  # Get the first character of the filename
    
    if filename_prefix == 'W':
        refactor_value = refactor_value_w
    elif filename_prefix == 'S':
        refactor_value = refactor_value_s
    else:
        # Use default refactor value or handle other cases as needed
        refactor_value = 1.0  # Default case, no downsampling
        
    num_samples = array.shape[0]
    num_columns = array.shape[1]
    new_length = int(num_samples / refactor_value)  # Calculate the new number of samples for this array
    
    if refactor_value != 1.0:
        new_array = np.zeros((new_length, num_columns))
        
        # Perform column-wise interpolation
        for col in range(num_columns):
            x_old = np.linspace(0, num_samples - 1, num_samples)
            x_new = np.linspace(0, num_samples - 1, new_length)
            f = interp1d(x_old, array[:, col], kind='linear')
            new_array[:, col] = f(x_new)
        
        downsampled_data.append(new_array)
    else:
        downsampled_data.append(array)  # Append unchanged data for files that don't need downsampling
    
#%%
###### THE GAIT EVENTS ############
groupE = []
dataE = []
filenameE = []

if 'pathToDataEvents' in locals():
    for file1E in os.listdir(pathToDataEvents):
        if file1E[0:4] != 'S017':
            if file1E[0:4] != 'S022':
                dataE.append(spio.loadmat(pathToDataEvents + file1E))
                filenameE.append(file1E)
                if file1E[0]=='W':
                    groupE.append(int(file1E[1:3]))
                else:
                    groupE.append(int(file1E[1:4]))
            else:
                print(file1E[0:4])
        else:
            print(file1E[0:4])
    #     perturbationType.append(numberPer)
    # numberPer +=1
groupE = np.array(groupE)
# get the dict out of the list
for numtrials in range(len(filenameE)):
    dataE[numtrials] = (dataE[numtrials]['Events'])#'markervel'


############### STEP 2 ###################################
#%%

# Iterate through dataE
for indx in range(len(dataE)):
    filename_prefix = filename[indx][0]  # Get the first character of the filename
    
    if filename_prefix == 'W':
        original_frequency = original_frequency_w
    elif filename_prefix == 'S':
        original_frequency = original_frequency_s
    else:
        # Raise an error if filename does not start with 'W' or 'S'
        raise ValueError(f"Filename '{filename[indx]}' does not start with 'W' or 'S'.")
    
    # Perform rounding based on the adjusted frequency ratio
    dataE[indx] = np.round(dataE[indx] * (frequency / original_frequency), 0)

#%%
start = 0
for indx in range(0,len(filename)): # find the first file with healthy "w"file name
    if filename[indx][0]=='W':
        start = indx
        break    

# overlapping windows
TrialE = []
count = 0
trackGroup = []
trackGroup1 = []
dataAugmented = np.zeros((0, len(inputColumns)))  # Initialize as empty with correct number of columns


for indx in range(len(downsampled_data)):
    if len(downsampled_data[indx]) > windowLength:
        for indx2 in range(len(dataE[indx]) - 2):  # Loop through each index except the last one
            current_index = int(dataE[indx][indx2][0])
            next_index = int(dataE[indx][indx2 + 2][0])
            tempArray = downsampled_data[indx][current_index:next_index]
            if len(tempArray) > 0:  # Ensure there's data between the indices
                # Interpolate tempArray to have window_length rows
                num_columns = tempArray.shape[1]
                new_tempArray = np.zeros((windowLength, num_columns))
                
                for col in range(num_columns):
                    x_old = np.linspace(0, len(tempArray) - 1, len(tempArray))
                    x_new = np.linspace(0, len(tempArray) - 1, windowLength)
                    f = interp1d(x_old, tempArray[:, col], kind='linear', fill_value="extrapolate")
                    new_tempArray[:, col] = f(x_new)
                
                count += 1
                dataAugmented = np.vstack((dataAugmented, new_tempArray[:, inputColumns]))
                trackGroup.append(indx)
                if indx < start:
                    trackGroup1.append('SS' + str(group[indx]))
                    TrialE.append(int(filename[indx][10]))
                else:
                    trackGroup1.append('W' + str(group[indx]))
                    TrialE.append(int(filename[indx][9]))
                    

#%%
######## STEP 3 SPLIT THE DATASET ##################################

y = [0]
# group = []
for indx in range(len(trackGroup)):
    x = filename[trackGroup[indx]].split('FR')
    print(x)
    if x[0][0]=='W':
        # if x[0][1:3].isdigit():
        #     if int(x[0][1:3])<25:
        #         y = np.vstack((y,3))
        #     else:
        #         y = np.vstack((y,2))
        y = np.vstack((y,1))
    else:
        y = np.vstack((y,0))
    # if xx = x[0].split('W')
        # y = np.vstack((y,int(x[1][0])))   
    # group.append(int(filename[1:4]))
    print(y)

y = np.delete(y,(0),axis=0)
# group = np.array(group)
########    Reshape  the X data ############
dataAugmented = dataAugmented.reshape(len(y),200,len(inputColumns))


#%%
#Anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA


num_windows = dataAugmented.shape[0]
window_size = dataAugmented.shape[1]
num_joint_angles = dataAugmented.shape[2]


# Initialize empty arrays to store PCA components and anomaly scores
pca_components = []
anomaly_scores = []

# Perform PCA separately for each joint angle
for j in range(num_joint_angles):
    # Reshape data for the current joint angle to (1000, 200)
    data_joint_angle = dataAugmented[:, :, j]

    # Perform PCA on the current joint angle data
    pca = PCA(n_components=1)  # Choose the number of components as per your data
    principal_components = pca.fit_transform(data_joint_angle)

    # Store principal components and anomaly scores
    pca_components.append(principal_components)
    anomaly_scores.append(pca.components_)

# Create a DataFrame from PCA components
df_pca = pd.DataFrame(data=np.concatenate(pca_components, axis=1))
data_mean = np.mean(dataAugmented, axis=1)

df = pd.DataFrame(data_mean)

anomaly_inputs_x = [0,3,6,9,12,15]
anomaly_inputs_y = [1,4,7,10,13,16]
anomaly_inputs_z = [2,5,8,11,14,17]

# Fit the isolation forest model
model = IsolationForest(contamination=0.08, random_state=42)
model.fit(df[anomaly_inputs_x])

df['anomaly_scores'] = model.decision_function(df[anomaly_inputs_x])
df['anomaly'] = model.predict(df[anomaly_inputs_x])

# outlier_plot(df, 'Isolation Forest')
palette = ['#ff7f0e','#1f77b4']
sns.pairplot(df, vars=anomaly_inputs_x, hue='anomaly', palette=palette)


#%%

externSubjects = ['','','','']#'SS1','SS38','W1','W38'
indexenexternSubjects = []
indexeninternSubjects = []
train_data = []
test_data = []
extern_data = []

## External validation stuff ###
for indx in range(0,len(trackGroup1)):
    if  trackGroup1[indx]!=externSubjects[0] and trackGroup1[indx]!=externSubjects[1] and trackGroup1[indx]!=externSubjects[2] and trackGroup1[indx]!=externSubjects[3]:
        indexeninternSubjects.append(indx)     
    else:
        indexenexternSubjects.append(indx)
        # print(trackGroup1[indx])
extern_data = dataAugmented[indexenexternSubjects,:,:]
other_data = dataAugmented[indexeninternSubjects,:,:] 
y_adapted = y[indexeninternSubjects] # which group (stroke F / NF / healthy etc)

# Make a int list of subjects.
subject = 0
groupsplit=[subject]
for indx in range(1,len(trackGroup1)):
    if trackGroup1[indx]==trackGroup1[indx-1]:
        groupsplit.append(subject)
    else:
        subject+=1
        groupsplit.append(subject)
groupsplit = np.array(groupsplit)
groupsplit =  groupsplit[indexeninternSubjects]

## Group split is a variable which contains a number for each subject. the length is equal to other_data (which needs to be split)

### saving other_data, y_adapted and groupsplit. so next the manuscrit can start over here!

np.save(pathToCleanData + "stored_3D_other_data_withoutS01722_latentfeatures1_" + str(latentFeatures) + "_frequency_" + str(frequency),other_data)
np.save(pathToCleanData + "stored_y_3D_adapted_withoutS01722_latentfeatures1_" + str(latentFeatures) + "_frequency_" + str(frequency),y_adapted)
np.save(pathToCleanData + "stored_3D_groupsplit_withoutS01722_latentfeatures1_" + str(latentFeatures) + "_frequency_" + str(frequency),groupsplit)

