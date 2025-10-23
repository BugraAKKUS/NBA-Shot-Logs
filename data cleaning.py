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

before_cleaning=len(df)
#problem1 Short shots incorrectly labeled as 3 Pointers? Vice-versa? and 2 pointers mislabeled as 3 pointers
# try KNNImputer, IterativeImputer.(hatali labellarin KNNImputer, IterativeImputer ile duzeltilmesi icin diger 
# degidkenlerle iliskili olmasi lazim o ilikilere bak eger yuksekse bunu aciklayarak kullan)
def count_pointers_distants(dataframe, feet=22,p=3 , cond='lower'):
    """
    Count number of shots labeled as 3-pointers taken from less than specified feet
    
    Args:
        dataframe: pandas DataFrame containing shot log data
        feet: distance in feet to check against (default is 22)
    Returns:
        int: Number of  3-point shots distants less than specified feet
    """
    if cond=='lower':
        condition = (dataframe['PTS_TYPE'] == p) & (dataframe['SHOT_DIST'] < feet)
        count = len(dataframe[condition])
    elif cond=='upper':
        condition = (dataframe['PTS_TYPE'] == p) & (dataframe['SHOT_DIST'] > feet)
        count = len(dataframe[condition])
    else:
        raise ValueError("cond must be either 'lower' or 'upper'")
    return count

# Use the function
incorrect_threes = count_pointers_distants(df)
print(f"Number of 3-point shots taken from less than 22 feet: {incorrect_threes}")
incorrect_threes_distant_15 = count_pointers_distants(df, feet=15)
print(f"Number of 3-point shots taken from less than 15 feet: {incorrect_threes_distant_15}")
#so there is some mislabeling going on in three pointers 
#lets check more than 22 feet 2 pointers
print("Number of 2-point shots taken from more than 22 feet:", count_pointers_distants(df, feet=22, p=2, cond='upper'))
print('Total number shots:',len(df))
#total mislabeld shots less then 1%. so we can remove them or keep them as is based on analysis needs

def clean_shot_labels(dataframe, distance_threshold=22):
    """
    Remove mislabeled shots from the DataFrame:
    - 3-pointers taken from less than threshold distance
    - 2-pointers taken from more than threshold distance
    
    Args:
        dataframe: pandas DataFrame containing shot log data
        distance_threshold: distance in feet (default is 22)
    Returns:
        pandas DataFrame: Cleaned dataset with mislabeled shots removed
    """
    # Create mask for correctly labeled shots
    correct_shots = ~(
        ((dataframe['PTS_TYPE'] == 3) & (dataframe['SHOT_DIST'] < distance_threshold)) |
        ((dataframe['PTS_TYPE'] == 2) & (dataframe['SHOT_DIST'] > distance_threshold))
    )
    
    # Apply mask to get clean dataset
    clean_df = dataframe[correct_shots].copy()
    
    # Print summary of removed shots
    removed_shots = len(dataframe) - len(clean_df)
    print(f"Removed {removed_shots} mislabeled shots ({(removed_shots/len(dataframe))*100:.2f}% of total)")
    
    return clean_df

# Clean the dataset
df_clean = clean_shot_labels(df)

# Verify the cleaning worked
print("\nAfter cleaning:")
print("3-point shots < 22 feet:", count_pointers_distants(df_clean, feet=22, p=3, cond='lower'))
print("2-point shots > 22 feet:", count_pointers_distants(df_clean, feet=22, p=2, cond='upper'))
print('Total number shots:',len(df))

# Problem 2: Missing data in SHOT_CLOCK
df.isnull().any()
print("\nBefore cleaning:")
missing_shot_clock = df['SHOT_CLOCK'].isna().sum()
print(f"Number of missing SHOT_CLOCK entries: {missing_shot_clock}")
print(f"Total number of rows: {len(df)}")

# Drop rows with missing shot clock and save to new DataFrame
clean_df = df.dropna(subset=['SHOT_CLOCK'], axis=0)

# Verify the cleaning worked
print("\nAfter cleaning:")
missing_shot_clock = clean_df['SHOT_CLOCK'].isna().sum()
print(f"Number of missing SHOT_CLOCK entries: {missing_shot_clock}")
print(f"Total number of rows: {len(clean_df)}")

# Problem 3: GAME_CLOCK change the numeric value for future analysis
# Changing the type of GAME_CLOCK so that it become an numerical feature  and can be trained in the future.
df['GAME_CLOCK'].str.split(':')
df['GAME_CLOCK'] = df['GAME_CLOCK'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
df.rename(columns={'GAME_CLOCK':'GAME_CLOCK_SEC'}, inplace=True)
print("\nAfter converting GAME_CLOCK to seconds:")
print(df['GAME_CLOCK_SEC'].head())

# Problem 4: convert shot_result to binary
print("\nBefore converting SHOT_RESULT:")
print(df['SHOT_RESULT'].value_counts())

# Convert shot results to binary
df['SHOT_RESULT'] = (df['SHOT_RESULT'] == 'made').astype(int)

# Verify the conversion
print("\nAfter converting SHOT_RESULT to binary:")
print(df['SHOT_RESULT'].value_counts())
print("\nValue counts explanation:")
print("1 = made shots")
print("0 = missed shots")

#check correlation matrix

# Calculate and display correlations
print("\nChecking correlations between numerical columns:")

# Select only numeric columns for correlation
numeric_columns = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_columns].corr()

# Display correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

shot_result_correlations = correlation_matrix['SHOT_RESULT'].sort_values(ascending=False)
print("\nCorrelations with SHOT_RESULT (sorted):")
print(shot_result_correlations)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
#so fgm and shot_Result represent same result we can drop fgm.
#also fgm should represent success shots of the player in the game but our data show missing or made so we can drop it.
df = df.drop('FGM', axis=1)
print("\nAfter dropping FGM column:")
print(df.head())

# Problem 5: Check for negative touch times
print("\nAnalyzing TOUCH_TIME:")
print(f"Minimum touch time: {df['TOUCH_TIME'].min()}")
print(f"Maximum touch time: {df['TOUCH_TIME'].max()}")

negative_touch_times = df[df['TOUCH_TIME'] < 0]
print(f"\nNumber of negative touch times: {len(negative_touch_times)}")

# Remove negative touch times and create new clean dataset
print("\nRemoving negative touch times:")
print(f"Original dataset size: {len(df)}")

# Remove negative touch times
df_clean = df[df['TOUCH_TIME'] >= 0].copy()

# Verify the cleaning
print(f"Dataset size after removing negative touch times: {len(df_clean)}")
print(f"Number of rows removed: {len(df) - len(df_clean)}")
print(f"New touch time range: {df_clean['TOUCH_TIME'].min()} to {df_clean['TOUCH_TIME'].max()} seconds")

# Update the main dataframe
df = df_clean

##after all cleaning print first 5 records
print("\nFirst 5 records of cleaned dataset:", df.head())
print(df.dtypes)
after_cleaning=len(df)
print(f"\nTotal records before cleaning: {before_cleaning}")
print(f"Total records after cleaning: {after_cleaning}")
print(f"Total records removed during cleaning: {before_cleaning - after_cleaning}")