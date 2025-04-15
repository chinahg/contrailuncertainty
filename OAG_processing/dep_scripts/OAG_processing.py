import tqdm
import numpy as np
import pandas as pd


"""
Process the OAG data for the year 2022.

Reads the OAG2022.csv file and performs the following operations:
1. Groups the data by Origin Airport, Destination Airport, and Aircraft Type.
2. Calculates the sum of Number of Operations for each group.
3. Adds columns for Annual Distance and Annual Fuelburn.
4. Retrieves the first value of fuel per flight and distance for each group.
5. Calculates the Annual Fuelburn and Annual Distance based on the Number of Operations.
6. Filters out rows where Annual Distance or Annual Fuelburn is zero.
7. Saves the processed data to OAG_processed.csv file.
"""

# Import OAG data
ds_OAG = pd.read_csv("/home/chinahg/GCresearch/contrailuncertainty/OAG_processing/OAG2022.csv", low_memory=False)

# Group the data by Origin Airport, Destination Airport, and Aircraft Type
grouped_OAG = ds_OAG.groupby(['Origin.Airport', 'Destination.Airport', 'Aircraft.Type'])

# Calculate the sum of Number of Operations for each group
ds_OAG_nb_ops = grouped_OAG['Number.of.Operations'].sum().reset_index()

# Add columns for Annual Distance and Annual Fuelburn in a new DataFrame
ds_OAG_processed = ds_OAG_nb_ops
ds_OAG_processed['Annual.Distance'] = 0
ds_OAG_processed['Annual.Fuelburn'] = 0

# Retrieve the first value of fuel per flight and distance for each group
ds_grp_fuelperflight = grouped_OAG['fuel.per.flight(kg)'].first().reset_index()
ds_grp_distance = grouped_OAG['distance(nm)'].first().reset_index()

# Calculate Annual Fuelburn and Annual Distance
ds_OAG_processed['Annual.Fuelburn'] = ds_grp_fuelperflight['fuel.per.flight(kg)'] * ds_OAG_nb_ops['Number.of.Operations']
ds_OAG_processed['Annual.Distance'] = ds_grp_distance['distance(nm)'] * ds_OAG_nb_ops['Number.of.Operations']

# Sum the operations, and annual fuelburn and distance for each aircraft type
ds_OAG_processed = ds_OAG_processed.groupby(['Aircraft.Type']).agg(
    Number_of_Operations=('Number.of.Operations', 'sum'),
    Annual_Distance=('Annual.Distance', 'sum'),
    Annual_Fuelburn=('Annual.Fuelburn', 'sum')
).reset_index()

# Save the processed data to OAG_processed.csv file
ds_OAG_processed.to_csv('OAG_processed.csv', index=False)

print("OAG data processed successfully.")