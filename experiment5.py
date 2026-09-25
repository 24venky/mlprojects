import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
x = iris.data
y = iris.target

# FIX 1: Fixed the unpacking order (x_train, x_test, y_train, y_test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize and train classifier
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predict
y_pred = classifier.predict(x_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

# FIX 2: Corrected the f-string formatting syntax to :.2f
print(f"accuracy: {accuracy:.2f}")
print("\nclassification report:")
print(report)
