import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('/kaggle/input/bank-note-authentication-uci-data/BankNote_Authentication.csv')

# Display the first few rows of the dataset
df.head()

# Display information about the dataset, including data types and missing values
df.info()

# Check for any missing values in the dataset
df.isnull().sum()

# Visualize the distribution of the target variable
ax = sns.countplot(data=df, x='class')
ax.legend(title='[5,9]')

# Visualize the pairwise relationships between the features, with hue representing the target variable
ax = sns.pairplot(df, hue="class", markers=["o", "D"])
ax.fig.suptitle('Pairwise relationship in data, [5,9]', y=1)

# Split the dataset into input features (X) and target variable (Y)
X, Y = df.iloc[:, :4], df.iloc[:, 4]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y)

# Create a decision tree classifier with entropy as the criterion
clf_e = tree.DecisionTreeClassifier(criterion="entropy")

# Train the decision tree classifier on the training data
clf_e.fit(x_train, y_train)

# Make predictions on the test data
output = clf_e.predict(x_test)

# Generate a classification report to evaluate the model performance
report = classification_report(y_test, output)
print(report)

# Generate a confusion matrix to evaluate the model performance
matrix = confusion_matrix(y_test, output)
print(matrix)

# Define the class labels for the confusion matrix visualization
classes = ['Forged', 'Genuine']

# Visualize the confusion matrix
fig = sns.heatmap(matrix, cmap='Blues', fmt='g', annot=True, xticklabels=classes, yticklabels=classes)
fig.set_xlabel('Prediction')
fig.set_ylabel('Actual')

# Plot the decision tree
plt.figure()
plt.figure(figsize=(30, 20))
tree.plot_tree(clf_e, filled=True)
plt.show()
