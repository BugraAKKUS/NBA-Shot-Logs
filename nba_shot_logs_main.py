# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load - the dataset has a CSV file
file_path = "shot_logs.csv"

# Load the latest version using the newer dataset_load() method
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "dansbecker/nba-shot-logs",
    file_path
)

print("First 5 records:", df.head())

##Problems:
#problem1: Short shots incorrectly labeled as 3 Pointers? Vice-versa?
def count_incorrect_three_pointers(dataframe):
    """
    Count number of shots labeled as 3-pointers taken from less than 22 feet
    
    Args:
        dataframe: pandas DataFrame containing shot log data
    Returns:
        int: Number of potentially incorrect 3-point shots
    """
    condition = (dataframe['PTS_TYPE'] == 3) & (dataframe['SHOT_DIST'] < 22)
    count = len(dataframe[condition])
    return count

# Use the function
incorrect_threes = count_incorrect_three_pointers(df)
print(f"Number of 3-point shots taken from less than 22 feet: {incorrect_threes}")
#print(df[(df['PTS_TYPE'] == 3) & (df['SHOT_DIST'] < 22)])