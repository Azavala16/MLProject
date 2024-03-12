import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# file path here
df = pd.read_csv('C:\\Users\\Alexi\\OneDrive\\Desktop\\Machine Learning\\Project\\FallRiskDataSet.csv')

# Dropped rows with any missing values in any column
df.dropna(inplace=True)

# Dropped columns
df = df.drop(columns=['Number', 'Date of incident'])

# get attribute which values that contain '/' or ',' 
contains_slash = df['Presence of companion at time of incident'].str.contains('/')
contains_comma = df['Presence of companion at time of incident'].str.contains(',')

# Standardized 'Presence of companion at the time of incident'
df.loc[contains_slash | contains_comma, 'Presence of companion at time of incident'] = 'yes'

misspelled_mapping = {
    'inpatint units': 'inpatient units',
    'emergency department': 'emergency depart',
    'ob&gyn/birth': 'ob&gynbirth',
    'ob&gyb/birth': 'ob&gynbirth',
    'adult icu': 'adult acu',
    'adult acu': 'adult aci',
    'excotiation': 'excoriation',
    'exam rom': 'exam room',
    'surgical prep adverse event': 'surgical prep aradverse event',
    'deficit motor': 'motor deficit',
    'seious adverse event': 'serious adverse event',
    'female': 'f',
    'male': 'm'
}

# fix inconsistencies
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.lower()
        df[col] = df[col].str.strip()
        for misspelled, correct in misspelled_mapping.items():
            df[col] = df[col].str.replace(misspelled, correct)

# Convert "Age range" column to categories and then codes
df['Age range'] = pd.Categorical(df['Age range of patient'], categories=df['Age range of patient'].unique()).codes

# Dropping original "Age range of patient" column
df.drop(columns=['Age range of patient'], inplace=True)

# Convert all values of attributes with yes and no 
for col in df.columns:
    if df[col].dtype == 'object':
        # (Yes: 1, No: 0)
        if any(df[col].isin(['yes', 'no'])):
            df[col] = df[col].map({'yes': 1, 'no': 0})
        # df[col] = df[col].astype('category').cat.codes
            
    # check values for column are unique (no inconsistencies)
    # print(f'Unique values for Attribute: {col}\n', df[col].unique())
    # print('\n')

# send clean data to new csv file
df.to_csv('cleaned_data.csv', index=False)

# Load the cleaned data
df = pd.read_csv('cleaned_data.csv')

# This function normalizes a feature (column) of a dataset.
# Normalize a feature (column) of a dataset
def normalize_feature(data, feature):
    # encode sring values
    if data[feature].dtype == 'object':
        label_encoder = LabelEncoder()
        data[feature] = label_encoder.fit_transform(data[feature])
    else: # scale values
        scaler = StandardScaler()
        data[feature] = scaler.fit_transform(data[[feature]])
    return data


# Calculates Euclidean distance between two datapoints
def euclidean_distance(dp1, dp2):
    dp1 = np.array(dp1, dtype=float)
    dp2 = np.array(dp2, dtype=float)
    return np.sqrt(np.sum((dp1 - dp2) ** 2))

# Get the k nearest neighbors for a new datapoint
def get_neighbors(X_train, new_dp, k):
    distances = []
    neighbors = []
    for i, dp in enumerate(X_train):
        dist = euclidean_distance(new_dp, dp)  # cal euclidean distance
        distances.append((dp, dist))           # add point-distance tuple to list
    distances.sort(key=lambda x: x[1])         # sort list on distance
    for i in range(k):
        neighbors.append(distances[i][0])      # add k sorted points from distance list to neighbors list
    return neighbors

# determining the class label for the current datapoint
# based on the majority of class labels of its k neighbors.
def predict_dp(neighbors, y_train, X_train):
    class_counter = Counter()
    for neighbor in neighbors:
        # find corresponding label from y_train for each neighbor
        class_label = y_train[np.where((X_train == neighbor).all(axis=1))[0][0]]
        class_counter[class_label] += 1                 # keep track of Class label counter
    prediction = class_counter.most_common(1)[0][0]     # make pred based on majority class  
    return prediction                       

# returns the different X and Y values and predictions
# depending on the data output choice by user (35% of whole dataset, or each year)
def operations_by_choice(X_train, X_test, y_train, y_test):
    # Normalizing the features before doing operations
    for feature in X_test.columns:
        X_train = normalize_feature(X_train, feature)
        X_test = normalize_feature(X_test, feature)
        
    # predicting the class labels of the test set with k = 3
    k = 3
    predictions = []
    for datapoint in X_test.values:
        neighbors = get_neighbors(X_train.values, datapoint, k)
        prediction = predict_dp(neighbors, y_train.values.ravel(), X_train)
        predictions.append(prediction)
    
    return X_test, y_test, predictions

def test_data():
   # Asking the user for choice
    print("Choose an option:")
    print("1. Test on the entire dataset")
    print("2. Test for each year individually")

    choice = int(input("Enter your choice (1 or 2): "))

    if choice == 1:
        # Splitting the data into training and testing sets for 35% of the entire dataset
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Fall risk level']), df['Fall risk level'],
                                                            test_size=0.30, random_state=20)
        
        X_test, y_test, predictions = operations_by_choice(X_train, X_test, y_train, y_test)
        correct = sum([y_true == y_pred for y_true, y_pred in zip(y_test.values.ravel(), predictions)])
        accuracy = (correct / len(y_test)) * 100
        print(f"\nAccuracy for 30% of entire dataset: {accuracy:.2f}%")

        # Writing accuracy results to a text file
        with open('Results.txt', 'a') as f:
            f.write(f"\nAccuracy for 30% of entire dataset: {accuracy:.2f}%  /  number of samples: {len(y_test)} samples")
            f.write(f'\ngeneral statistics:\n{df.describe()}')

        results_df = pd.DataFrame()
        results_df['Actual Fall Risk Level'] = y_test.values.ravel()
        results_df['Predicted Fall Risk Level'] = predictions
        results_df.to_csv('knn_predictions_entire_dataset.csv', index=False)


    elif choice == 2:
        # Getting unique years in the dataset
        unique_years = df['Year'].unique()
        
        # List to store train_test_sets for each year
        train_test_sets = []
        
        # Looping through each year to create train and test sets
        for year in unique_years:
            # Filtering data for the current year
            current_year_data = df[df['Year'] == year]
            X_train, X_test, y_train, y_test = train_test_split(current_year_data.drop(columns=['Fall risk level']), 
                                                                current_year_data['Fall risk level'],
                                                                test_size=0.35, random_state=41)
            train_test_sets.append((X_train, X_test, y_train, y_test))

        # Looping through each train and test sets for each year
        for idx, (X_train, X_test, y_train, y_test) in enumerate(train_test_sets):
            
            X_test, y_test, predictions = operations_by_choice(X_train, X_test, y_train, y_test)
            correct = sum([y_true == y_pred for y_true, y_pred in zip(y_test.values.ravel(), predictions)])
            accuracy = (correct / len(y_test)) * 100
            year_count = 0

            print(f"Accuracy on 35% of data for Year {unique_years[idx]}: {accuracy:.2f}%  /  number of samples: {len(y_test)} samples")

            with open('Results.txt', 'a') as f:
                f.write(f"\nAccuracy on 35% of data for Year {unique_years[idx]}: {accuracy:.2f}%  /  number of samples: {len(y_test)} samples\n")
                f.write(f'general statistics for Year {unique_years[idx]}:\n {df.describe()}')

            results_df = pd.DataFrame()
            results_df['Actual Fall Risk Level'] = y_test.values.ravel()
            results_df['Predicted Fall Risk Level'] = predictions

            year_count += 1
            # results_df.to_csv(f'knn_predictions_{X_test["Year"].iloc[0]}.csv', index=False)

    else:
        print("\n---Invalid choice. Please enter 1 or 2.---")
        test_data()

# # check there are no null values
with open('Results.txt', 'a') as f:
    f.write(f'\n-----Checking for null values for each attribute-----\n {df.isnull().sum()}')
        
while True:
    test_data()
    print("\nWould you like to print the data once again?")
    print_data = int(input("Enter anything for Yes 0 for No:\n"))
    if print_data == 0:
        print('program terminated.')
        break