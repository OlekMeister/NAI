"""
This script demonstrates the use of Support Vector Machine (SVM) with a radial basis function (RBF) kernel for classification.
It loads data from a CSV file, splits it into training and testing sets, trains an SVM model, predicts on the test set,
generates and displays a confusion matrix.

Dependencies:
- pandas
- matplotlib
- scikit-learn

REMEMBER TO INSTALL ALL THE DEPENDENCIES BY USING: pip install *Dependencies*

Usage:
1. Adjust the input_file path according to your dataset.
2. Run the script.

Author = Aleksander Guzik

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load data from CSV
input_file = 'data_milk.csv'
data = pd.read_csv(input_file)

# Select a subset of columns (adjust the range as needed)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

# Initialize and train SVM classifier
svc = svm.SVC(kernel='rbf', C=1, gamma=100).fit(X_train, y_train)

# Predict on the test set
y_test_pred = svc.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Print the confusion matrix
print(cm)

# Show the plot
plt.show()
