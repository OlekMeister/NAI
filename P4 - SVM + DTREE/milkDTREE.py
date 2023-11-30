"""
This script demonstrates the use of a Decision Tree classifier for multiclass classification.
It loads data from a CSV file, visualizes input data based on class labels, splits the data into training and testing sets,
trains a Decision Tree model, predicts on the test set, and evaluates classifier performance using classification reports.

Dependencies:
- pandas
- matplotlib
- scikit-learn

REMEMBER TO INSTALL ALL THE DEPENDENCIES BY USING: pip install *Dependencies*

Usage:
1. Adjust the input_file path based on your dataset.
2. Run the script.

Author = Aleksander Guzik

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load input data
input_file = 'data_milk.csv'
data = pd.read_csv(input_file)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Separate input data into three classes based on labels
class_0 = X[y == 'low']
class_1 = X[y == 'medium']
class_2 = X[y == 'high']

# Visualize input data
plt.figure()
plt.scatter(class_0.iloc[:, 0], class_0.iloc[:, 1], s=75, facecolors='black',
            edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1.iloc[:, 0], class_1.iloc[:, 1], s=75, facecolors='white',
            edgecolors='black', linewidth=1, marker='o')
plt.scatter(class_2.iloc[:, 0], class_2.iloc[:, 1], s=75, facecolors='yellow',
            edgecolors='black', linewidth=1, marker='h')
plt.title('Input data')

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

# Decision Trees classifier
params = {'random_state': 0, 'max_depth': 8}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)

# Evaluate classifier performance
class_names = ['Class-0', 'Class-1', 'Class-2']
print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

plt.show()
