import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("Python is running this file!")
print("Libraries imported successfully!")

# Make fake dataset
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)

print("Classifier accuracy:", clf.score(X_test, y_test))

