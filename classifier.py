import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 1. Generate a toy dataset
X, y = make_classification(
    n_samples=200,      # number of rows
    n_features=2,       # number of features
    n_informative=2,    # useful features
    n_redundant=0,
    n_classes=2,        # binary classification
    random_state=42
)

# Convert to pandas DataFrame
df = pd.DataFrame(X, columns=["feature1", "feature2"])
df["target"] = y

# 2. Inspect data
print(" First 5 rows of the dataset:")
print(df.head())
print("\n Data summary:")
print(df.describe())

# 3. Define features (X) and target (y)
X = df[["feature1", "feature2"]]
y = df["target"]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Evaluate model
accuracy = model.score(X_test, y_test)
print(f"\n Model accuracy: {accuracy:.2f}")

# 7. Detailed classification report
y_pred = model.predict(X_test)
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# 8. Visualization
plt.scatter(X_test["feature1"], X_test["feature2"], c=y_pred, cmap="coolwarm", edgecolor="k")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Predictions")
plt.show()



