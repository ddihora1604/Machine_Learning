import pandas as pd
from pandas import DataFrame

# Uncomment the following line to load the dataset from an external CSV file
data = pd.read_csv('ENJOYSPORT.csv')

#  # Sample dataset defined within the code
# data = pd.DataFrame({
#     'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny', 'Rainy', 'Rainy', 'Sunny'],
#     'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm', 'Cold', 'Cold', 'Warm'],
#     'Humidity': ['Normal', 'High', 'High', 'High', 'Normal', 'Normal', 'High'],
#     'Wind': ['Strong', 'Strong', 'Strong', 'Strong', 'Weak', 'Weak', 'Strong'],
#     'Water': ['Warm', 'Warm', 'Cold', 'Warm', 'Warm', 'Cold', 'Warm'],
#     'Forecast': ['Same', 'Same', 'Change', 'Same', 'Same', 'Change', 'Same'],
#     'EnjoySport': ['Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes']  # Target column
# })

# Check dataset details
columnLength = data.shape[1]
rowlen = data.shape[0]
print(columnLength)
print(rowlen)
print(data.values)
data.head()

hp = []
hn = []
h = ['0'] * (columnLength - 1)

# Split data into positive and negative examples based on target column 'EnjoySport'
for trainingExample in data.values:
    if trainingExample[-1] != 0:  # Changed to match the target value 'Yes'/'No'
        hp.append(list(trainingExample))
    else:
        hn.append(list(trainingExample))

# Generate the Maximally Specific Hypothesis (h)
for i in range(len(hp)):
    for j in range(columnLength - 1):
        if h[j] == '0':
            h[j] = hp[i][j]
        elif h[j] != hp[i][j]:
            h[j] = '?'
        else:
            h[j] = hp[i][j]

# Output the results
print('\nThe positive Hypotheses are:', hp)
print('\nThe negative Hypotheses are:', hn)
print('\nThe Maximally Specific Hypothesis h is:', h)