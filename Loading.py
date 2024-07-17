# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:06:33 2024

@author: gib445
"""

import btk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil

# Define the folder containing the .c3d files
input_folder = r'C:\Users\gib445\surfdrive - Larsen, A.G. (Aske Gye)@surfdrive.surf.nl\GitHub\VAE_stroke_longitudinal\Early group\V01\V01_17_met'
output_folder = r'C:\Users\gib445\surfdrive - Larsen, A.G. (Aske Gye)@surfdrive.surf.nl\GitHub\VAE_stroke_longitudinal\Early group\V01\V01_17_met'

# Check if the output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all folders in the input directory
folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]

# Recursively create empty folders in the output directory to match the input structure
for folder_name in folders:
    output_subfolder = os.path.join(output_folder, folder_name)
    os.makedirs(output_subfolder, exist_ok=True)

# Get a list of all .c3d files in the input directory and its subfolders
c3d_files = [os.path.join(root, file) for root, dirs, files in os.walk(input_folder) for file in files if file.endswith('.c3d')]

# Process each .c3d file and save it to the corresponding output folder
for c3d_file in c3d_files:
    try:
        # Create a reader
        reader = btk.btkAcquisitionFileReader()

        # Set the file path
        reader.SetFilename(c3d_file)

        # Read the file
        reader.Update()

        # Get the loaded data
        acq = reader.GetOutput()

        # Initialize lists to store labels and data arrays
        marker_labels = []
        marker_data = []

        # Iterate over points in the acquisition
        for i in range(acq.GetPointNumber()):
            point_label = acq.GetPoint(i).GetLabel()
            trajectories = acq.GetPoint(i).GetValues()

            # Check if the point is a marker
            if trajectories.shape[1] == 3:  # Marker data has 3 columns (X, Y, Z)
                marker_labels.append(point_label)
                marker_data.append(trajectories)

        # Convert lists to numpy arrays
        markers_array_3d = np.array(marker_data)

        # Create a dictionary to store data
        data_dict = {}

        # Iterate over marker labels and data
        for i, label in enumerate(marker_labels):
            data_dict[label + '_X'] = markers_array_3d[i, :, 0]
            data_dict[label + '_Y'] = markers_array_3d[i, :, 1]
            data_dict[label + '_Z'] = markers_array_3d[i, :, 2]

        # Create a DataFrame
        df = pd.DataFrame(data_dict)

        columns_of_interest = ['LHipAngles_X', 'LHipAngles_Y', 'LHipAngles_Z',
                               'LKneeAngles_X', 'LKneeAngles_Y', 'LKneeAngles_Z',
                               'LAnkleAngles_X', 'LAnkleAngles_Y', 'LAnkleAngles_Z',
                               'RHipAngles_X', 'RHipAngles_Y', 'RHipAngles_Z',
                               'RKneeAngles_X', 'RKneeAngles_Y', 'RKneeAngles_Z',
                               'RAnkleAngles_X', 'RAnkleAngles_Y', 'RAnkleAngles_Z']

        df_filtered = df[columns_of_interest]
        df_filtered_0 = df_filtered.loc[~(df_filtered == 0).any(axis=1)]


    #     # Construct the output file path within the output folder
    #     relative_path = os.path.relpath(c3d_file, input_folder)
    #     output_file_path = os.path.join(output_folder, relative_path.replace('.c3d', '_processed.xlsx'))

    #     # Create the directory structure if it doesn't exist
    #     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    #     # Save the DataFrame to an Excel file
    #     df_filtered.to_excel(output_file_path, index=False)

    #     print("Data saved to:", output_file_path)
    except Exception as e:
    #     # If an error occurs, save the file with "_error" suffix instead
    #     output_file_path = output_file_path.replace('_processed.xlsx', '_error.xlsx')
    #     df_filtered.to_excel(output_file_path, index=False)
        # print(f"Error occurred with file {c3d_file}. Data saved to:", output_file_path)
        print("Error message:", str(e))

