# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Step 1: Data Loading and Preparation ---
# Load the dataset from the CSV file
try:
    df = pd.read_csv('uploaded:heart.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: heart.csv not found. Please ensure the file is uploaded correctly.")
    exit()

# Separate features (X) and target (y)
# The target variable is 'target', and all other columns are features
X = df.drop('target', axis=1)
y = df['target']
feature_names = X.columns.tolist()

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 2: Decision Tree Classifier ---
print("\n--- Training Decision Tree Classifier ---")
# 1. Train a Decision Tree Classifier with default settings (no depth limit)
# This model is likely to overfit the training data
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Evaluate the deep tree on both training and testing data
train_pred_dt = dt_classifier.predict(X_train)
test_pred_dt = dt_classifier.predict(X_test)
print(f"Deep Decision Tree - Training Accuracy: {accuracy_score(y_train, train_pred_dt):.4f}")
print(f"Deep Decision Tree - Testing Accuracy: {accuracy_score(y_test, test_pred_dt):.4f}")

# 2. Visualize the Decision Tree
# The full tree can be too large to visualize, so we'll plot a simplified version with a max_depth
print("\nVisualizing a pruned Decision Tree (max_depth=3) for clarity...")
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, max_depth=3, feature_names=feature_names, class_names=['No Disease', 'Disease'], filled=True, rounded=True)
plt.title("Pruned Decision Tree (max_depth=3)")
plt.show()

# --- Step 3: Controlling Overfitting ---
# 3. Train a new Decision Tree with a controlled depth to prevent overfitting
print("\n--- Training a Pruned Decision Tree (max_depth=5) ---")
pruned_dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
pruned_dt_classifier.fit(X_train, y_train)
pruned_test_pred = pruned_dt_classifier.predict(X_test)
print(f"Pruned Decision Tree - Testing Accuracy: {accuracy_score(y_test, pruned_test_pred):.4f}")

# --- Step 4: Random Forest Classifier ---
# 4. Train a Random Forest model
print("\n--- Training Random Forest Classifier ---")
# Random Forest is an ensemble of many decision trees
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions and evaluate
rf_pred = rf_classifier.predict(X_test)
print(f"Random Forest - Testing Accuracy: {accuracy_score(y_test, rf_pred):.4f}")

# Compare accuracies
print("\n--- Model Comparison ---")
print(f"Decision Tree (deep) Accuracy: {accuracy_score(y_test, test_pred_dt):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")

# --- Step 5: Feature Importances ---
# 5. Interpret feature importances from the Random Forest model
print("\n--- Interpreting Feature Importances (from Random Forest) ---")
importances = pd.Series(rf_classifier.feature_importances_, index=feature_names)
importances_sorted = importances.sort_values(ascending=False)

print("Top 5 most important features:")
print(importances_sorted.head())

# Visualize feature importances
plt.figure(figsize=(12, 6))
importances_sorted.plot(kind='bar')
plt.title("Feature Importances in Random Forest")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Step 6: Cross-Validation ---
# 6. Evaluate models using cross-validation for a more robust accuracy estimate
print("\n--- Evaluating Models with 5-fold Cross-Validation ---")
# Evaluate the pruned Decision Tree
dt_cv_scores = cross_val_score(pruned_dt_classifier, X, y, cv=5)
print(f"Pruned Decision Tree CV Accuracy: {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std() * 2:.4f})")

# Evaluate the Random Forest
rf_cv_scores = cross_val_score(rf_classifier, X, y, cv=5)
print(f"Random Forest CV Accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")

