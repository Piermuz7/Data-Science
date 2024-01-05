import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import tensorflow as tf

# set seed for reproducibility
random.seed(1)

np.random.seed(1)

tf.random.set_seed(1)

filename = 'secret_data.csv'
df = pd.read_csv(filename)

#### PREPROCESSING ####

# replace the NAs from OCCUPATION_TYPE column and drop ID column
df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('jobless')
df = df.drop('ID', axis=1)

# make all labels to predict, numeric
status_replacement = {'X': 7, 'C': 6}
df['status'] = df['status'].replace(status_replacement).astype(int)

# one encode categorical columns
columns_to_one_encode = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=columns_to_one_encode)

# divide the dataset into training/testing
y = df['status'].to_numpy()
x = df.loc[:, df.columns != 'status']

# standardize numerical columns
columns_to_standardize = ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

scaler = StandardScaler()

x[columns_to_standardize] = scaler.fit_transform(x[columns_to_standardize])

#### LOAD BEST MODEL ####

model = tf.keras.models.load_model('final_model.h5')

#### EVALUATION ####

predictions = model.predict(x)
# Convert predictions to class labels
predicted_labels = predictions.argmax(axis=1)

# Compute various metrics
accuracy = accuracy_score(y, predicted_labels)
precision = precision_score(y, predicted_labels, average='weighted')
recall = recall_score(y, predicted_labels, average='weighted')
f1 = f1_score(y, predicted_labels, average='weighted')
# Display the metrics in a table
metrics_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, precision, recall, f1]
})

print("Metrics Table:")
print(metrics_table)
