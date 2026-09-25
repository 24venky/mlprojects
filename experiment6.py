import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create dataset
data = pd.DataFrame({
         'feature1':[2,4,4,6,6,8,10,12],
     'feature2':[4,2,4,2,6,6,8,10],
     'label': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b']
})
print("dataset :\n", data)

x = data[['feature1', 'feature2']]
y = data['label']

# Split data (Note: with 8 rows and test_size=0.3, x_test gets exactly 3 rows)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# FIX: Corrected spelling from n_nieghbors to n_neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# Predict
y_pred = knn.predict(x_test)

# Print metrics
print("\n confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\n classification report:\n", classification_report(y_test, y_pred, zero_division=0))
print("\n accuracy :", accuracy_score(y_test, y_pred))
