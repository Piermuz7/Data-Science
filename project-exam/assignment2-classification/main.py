#### SET-UP THE ENVIROMENT ####
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import confusion_matrix

# set seed for reproducibility
random.seed(1)

np.random.seed(1)

tf.random.set_seed(1)

# read the csv

df = pd.read_csv('data.csv')

#### DATA EXPLORATION ####

# display the first few rows of the dataset
print(df.head())

# basic statistics of numerical columns
print(df.describe())

# info about the dataset, including data types and missing values
print(df.info())

# explore categorical variables
# omit OCCUPATION_TYPE because the charts gets messy
categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE',
                    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']

for col in categorical_cols:
    plt.figure(figsize=(15, 6))
    sns.countplot(x=col, data=df, hue='status', palette='viridis')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

# it can be easily seen that class "0" is the predominant one. An address of unbalanced classes

# explore numerical columns
numerical_cols = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS']

# Set up a grid of subplots
fig, axes = plt.subplots(nrows=1, ncols=len(numerical_cols), figsize=(18, 5))
fig.suptitle('Distribution of Numerical Columns')

for i, col in enumerate(numerical_cols):
    # plot histograms for numerical columns
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent title overlap
plt.show()

# correlation heatmap for numerical columns
correlation_matrix = df[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Columns')
plt.show()

# countplot for the target variable "status"
plt.figure(figsize=(10, 6))
sns.countplot(x='status', data=df, palette='viridis')
plt.title('Distribution of Target Variable (status)')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

# as mentioned before, we have a strong imbalance of classes

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# standardize numerical columns
columns_to_standardize = ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

scaler = StandardScaler()

x_train[columns_to_standardize] = scaler.fit_transform(x_train[columns_to_standardize])

x_test[columns_to_standardize] = scaler.transform(x_test[columns_to_standardize])

#### OVERSAMPLING ####

# Find unique values and their counts
unique_labels, counts = np.unique(y_train, return_counts=True)

# Sort the labels based on their counts
sorted_indices = np.argsort(-counts)  # Sort in descending order
sorted_labels = unique_labels[sorted_indices]
sorted_counts = counts[sorted_indices]

# Oversample
smote = SMOTE(sampling_strategy={sorted_labels[4]: sorted_counts[3], sorted_labels[5]: sorted_counts[3],
                                 sorted_labels[6]: sorted_counts[3], sorted_labels[7]: sorted_counts[3]},
              random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

# print the occurrences for each class
unique_values, counts = np.unique(y_train, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"{value}: {count} occurrences")


#### CROSS VALIDATION ####

# Define the function to create the model
def create_model(dropout):
    model = Sequential()
    model.add(Dense(units=256, activation='relu', input_dim=x_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(units=64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(units=64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(units=32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(units=16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(units=8, activation='softmax'))
    model.compile(optimizer=Adam(amsgrad=True), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Create a KerasClassifier for use with GridSearchCV
model = KerasClassifier(model=create_model, epochs=1000, verbose=4)

# Define the parameter grid
param_grid = {
    'batch_size': [512, 1024, 2048],
    'model__dropout': [0.1, 0.2, 0.3]
}

# Use StratifiedKFold for cross-validation if dealing with imbalanced classes
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Create GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=cv, verbose=4)

# Perform the grid search on the data
grid.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid.best_params_
print("Best Hyperparameters:", best_params)

#### TRAINING ####

# create the model
final_model = create_model(dropout=best_params['model__dropout'])

# Further split the training set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

# Define a callback to save the best weights during training
checkpoint = ModelCheckpoint('best_weights.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

history = final_model.fit(x_train, y_train, epochs=20000, batch_size=best_params['batch_size'],
                          validation_data=(x_val, y_val),
                          callbacks=[checkpoint])

# Load the best weights after training
final_model.load_weights('best_weights.h5')

#### EVALUATE THE RESULTS

# Visualize the training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Training and Validation History')
plt.legend()
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = final_model.evaluate(x_test, y_test, verbose=1)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Test Loss: {test_loss:.4f}')

predictions = final_model.predict(x_test)
# Convert predictions to class labels
predicted_labels = predictions.argmax(axis=1)

# Compute various metrics
accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels, average='weighted')
recall = recall_score(y_test, predicted_labels, average='weighted')
f1 = f1_score(y_test, predicted_labels, average='weighted')
# Display the metrics in a table
metrics_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, precision, recall, f1]
})

print("Metrics Table:")
print(metrics_table)

# plot a confusion matrix

class_labels = np.unique(y_test)

conf_matrix = confusion_matrix(y_test, predicted_labels, labels=class_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
disp.plot(cmap='viridis')
plt.show()

#### SAVE THE MODEL, TRAINING, VALIDATION AND TEST DATASETS ####

# save the model
final_model.save('final_model.h5')

# merge features and labels for each dataset
train_data = x_train.copy()
train_data['status'] = y_train
val_data = x_val.copy()
val_data['status'] = y_val
test_data = x_test.copy()
test_data['status'] = y_test

# save the DataFrame to CSV
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

##### CNN CODE####
# this is the code of the cnn architecture we tried, as specified in the report, due to poor results this code was not discarded
''' 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


height = 5
width = 11
x_train = x_train.reshape((x_train.shape[0], height, width, 1))
x_val = x_val.reshape((x_val.shape[0], height, width, 1))

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(5, 11, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='softmax'))


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
history = model.fit(x_train, y_train, epochs=20000, batch_size=512, validation_data=(x_val, y_val))

# Make predictions
predictions = model.predict(x_test)

# Evaluate your model on the original test set (X_test, y_test)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Convert predicted probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Get class labels
class_labels = np.unique(y_test)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_labels, labels=class_labels)

# Display the confusion matrix with class labels
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
disp.plot(cmap='viridis')
'''

#### CLASS WEIGHTS CODE ####
'''
from sklearn.utils.class_weight import compute_class_weight

class_labels = classes=np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y= y_train)
class_weights_dict = dict(zip(class_labels, class_weights))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# Define the model

model = Sequential()
model.add(Dense(units=256, activation='relu', input_dim=x_train.shape[1]))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(units=32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(units=16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(units=8, activation='softmax'))
model.compile(optimizer=Adam(amsgrad=True), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
history = model.fit(x_train, y_train, epochs=20000, batch_size=512, validation_data=(x_val, y_val), class_weight= class_weights_dict)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

'''