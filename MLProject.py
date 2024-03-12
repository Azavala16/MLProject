import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# file path here
df = pd.read_csv('FallRiskDataSet.csv')

# Dropped rows with any missing values in any column
df.dropna(inplace=True)

# Dropped columns
df = df.drop(columns=['Number', 'Year', 'Date of incident'])

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
    print(f'Unique values for Attribute: {col}\n', df[col].unique())
    print('\n')

# check there are no null values
print('-----Checking for null values for each attribute-----\n', df.isnull().sum())

# debug print for 1st 5 samples
print('\n-----Printing first 5 samples of the clean datset-----\n', df.head())

# send clean data to new csv file
df.to_csv('cleaned_data.csv', index=False)

# Load the cleaned data
df = pd.read_csv('cleaned_data.csv')

# the target variable
X = df.drop(columns=['Fall risk level'])

# Target variable
y = df['Fall risk level']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=41)

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
def predict_dp(neighbors, y_train):
    class_counter = Counter()
    for neighbor in neighbors:
        # find corresponding label from y_train for each neighbor
        class_label = y_train[np.where((X_train == neighbor).all(axis=1))[0][0]]
        class_counter[class_label] += 1                 # keep track of Class label counter
    prediction = class_counter.most_common(1)[0][0]     # make pred based on majority class  
    return prediction                       

# Normalizing the features before doing operations
for feature in X.columns:
    X_train = normalize_feature(X_train, feature)
    X_test = normalize_feature(X_test, feature)
    
# predicting the class labels of the test set with k = 3
k = 3
predictions = []
for datapoint in X_test.values:
    neighbors = get_neighbors(X_train.values, datapoint, k)
    prediction = predict_dp(neighbors, y_train.values.ravel())
    predictions.append(prediction)

# Calculating and printing the accuracy of predictions
correct = sum([y_true == y_pred for y_true, y_pred in zip(y_test.values.ravel(), predictions)])
accuracy = (correct / len(y_test)) * 100
print(f"Accuracy: {accuracy:.2f}%")

# new DataFrame with original test data and predictions
results_df = pd.DataFrame()
# set the values of actual and predicted fall risk levels to dataframe
results_df['Actual Fall Risk Level'] = y_test.values.ravel()
results_df['Predicted Fall Risk Level'] = predictions

# Saving the results to a new CSV file
results_df.to_csv('knn_predictions.csv', index=False)