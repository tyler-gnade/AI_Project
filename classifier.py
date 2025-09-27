"""
BINARY CLASSIFICATION WITH LOGISTIC REGRESSION
==============================================

Purpose:
    This script demonstrates a complete machine learning workflow for binary
    classification using logistic regression. It generates synthetic data,
    performs model training and evaluation, and visualizes the results.

Features:
    - Synthetic dataset generation with configurable parameters
    - Data exploration and visualization
    - Train/test split with proper random state management
    - Logistic regression model training
    - Comprehensive model evaluation with multiple metrics
    - Enhanced data visualization with decision boundary

Dependencies:
    - pandas: Data manipulation and analysis
    - scikit-learn: Machine learning algorithms and utilities
    - matplotlib: Data visualization
    - numpy: Numerical operations

Author: Enhanced version of original classification script
Date: 2025
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import make_classification
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Dataset parameters
N_SAMPLES = 200
N_FEATURES = 2
N_INFORMATIVE = 2
N_CLASSES = 2
RANDOM_STATE = 42

# Model parameters
TEST_SIZE = 0.2
MODEL_RANDOM_STATE = 42

# Visualization parameters
FIGURE_SIZE = (12, 4)
DPI = 100


# =============================================================================
# DATA GENERATION AND PREPARATION
# =============================================================================

def generate_synthetic_dataset(n_samples=N_SAMPLES, n_features=N_FEATURES,
                               n_informative=N_INFORMATIVE, n_classes=N_CLASSES,
                               random_state=RANDOM_STATE):
    """
    Generate a synthetic binary classification dataset.

    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features
        n_informative (int): Number of informative features
        n_classes (int): Number of classes
        random_state (int): Random state for reproducibility

    Returns:
        tuple: Features (X) and target (y) arrays
    """
    print("Generating synthetic dataset...")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.0,  # Better class separation
        random_state=random_state
    )

    print(f"Dataset generated: {n_samples} samples, {n_features} features")
    return X, y


def prepare_dataframe(X, y):
    """
    Convert arrays to pandas DataFrame for easier manipulation.

    Args:
        X (array): Feature matrix
        y (array): Target vector

    Returns:
        pandas.DataFrame: Combined dataset
    """
    print("Converting to pandas DataFrame...")

    # Create DataFrame with descriptive column names
    feature_columns = [f"feature_{i + 1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_columns)
    df["target"] = y

    print("DataFrame created successfully")
    return df


# =============================================================================
# DATA EXPLORATION AND ANALYSIS
# =============================================================================

def explore_dataset(df):
    """
    Perform basic exploratory data analysis.

    Args:
        df (pandas.DataFrame): Dataset to explore
    """
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n First 5 rows of the dataset:")
    print(df.head())

    print("\n Dataset summary statistics:")
    print(df.describe())

    print(f"\n Dataset shape: {df.shape}")
    print(f" Missing values: {df.isnull().sum().sum()}")
    print(f" Class distribution:")
    print(df['target'].value_counts().sort_index())

    # Check for class balance
    class_balance = df['target'].value_counts()
    balance_ratio = min(class_balance) / max(class_balance)
    print(f"📊 Class balance ratio: {balance_ratio:.2f}")

    if balance_ratio < 0.8:
        print("Warning: Classes are imbalanced. Consider using stratified sampling.")


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split data into training and testing sets.

    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target variable
        test_size (float): Proportion of test set
        random_state (int): Random state for reproducibility

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print(f"\n🔄 Splitting data (test size: {test_size})...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Ensure balanced splits
    )

    print(f" Data split completed:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, random_state=MODEL_RANDOM_STATE):
    """
    Train a logistic regression model.

    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training targets
        random_state (int): Random state for reproducibility

    Returns:
        LogisticRegression: Trained model
    """
    print("\n Training logistic regression model...")

    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000  # Ensure convergence
    )
    model.fit(X_train, y_train)

    print("Model training completed")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and display comprehensive metrics.

    Args:
        model: Trained logistic regression model
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test targets

    Returns:
        tuple: y_pred, accuracy
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n Model Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Detailed classification report
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("\n Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Model coefficients (feature importance)
    print(f"\n Model Coefficients:")
    feature_names = X_test.columns
    for i, coef in enumerate(model.coef_[0]):
        print(f"   {feature_names[i]}: {coef:.4f}")
    print(f"   Intercept: {model.intercept_[0]:.4f}")

    return y_pred, accuracy


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(X_test, y_test, y_pred, model):
    """
    Create comprehensive visualizations of the results.

    Args:
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): True test labels
        y_pred (array): Predicted labels
        model: Trained logistic regression model
    """
    print("\n Creating visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, dpi=DPI)

    # Plot 1: Actual vs Predicted
    scatter1 = axes[0].scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],
                               c=y_test, cmap='coolwarm',
                               edgecolor='black', alpha=0.7, s=50)
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].set_title('Actual Labels (Test Set)')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Predictions with decision boundary
    scatter2 = axes[1].scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],
                               c=y_pred, cmap='coolwarm',
                               edgecolor='black', alpha=0.7, s=50)
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].set_title('Predicted Labels (Test Set)')
    axes[1].grid(True, alpha=0.3)

    # Add decision boundary to second plot
    if X_test.shape[1] == 2:
        h = 0.02  # Step size in the mesh
        x_min, x_max = X_test.iloc[:, 0].min() - 1, X_test.iloc[:, 0].max() + 1
        y_min, y_max = X_test.iloc[:, 1].min() - 1, X_test.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Create prediction mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_proba(mesh_points)[:, 1]
        Z = Z.reshape(xx.shape)

        # Add contour lines
        axes[1].contour(xx, yy, Z, levels=[0.5], colors='red',
                        linestyles='--', linewidths=2)
        axes[1].contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='coolwarm')

    # Add colorbars
    fig.colorbar(scatter1, ax=axes[0], label='Class')
    fig.colorbar(scatter2, ax=axes[1], label='Class')

    plt.tight_layout()
    plt.show()

    print("Visualizations created successfully")


# =============================================================================
# MAIN EXECUTION WORKFLOW
# =============================================================================

def main():
    """
    Main execution function that orchestrates the entire workflow.
    """
    print("🚀 STARTING BINARY CLASSIFICATION WORKFLOW")
    print("=" * 60)

    try:
        # Step 1: Generate dataset
        X, y = generate_synthetic_dataset()

        # Step 2: Prepare DataFrame
        df = prepare_dataframe(X, y)

        # Step 3: Explore data
        explore_dataset(df)

        # Step 4: Prepare features and target
        feature_columns = [col for col in df.columns if col != 'target']
        X = df[feature_columns]
        y = df['target']

        # Step 5: Split data
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Step 6: Train model
        model = train_logistic_regression(X_train, y_train)

        # Step 7: Evaluate model
        y_pred, accuracy = evaluate_model(model, X_test, y_test)

        # Step 8: Create visualizations
        create_visualizations(X_test, y_test, y_pred, model)

        print("\n" + "=" * 60)
        print(f" WORKFLOW COMPLETED SUCCESSFULLY!")
        print(f" Final Model Accuracy: {accuracy:.4f}")
        print("=" * 60)

        return model, accuracy

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    model, accuracy = main()

