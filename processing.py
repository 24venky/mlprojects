import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
 
data = {
    'name': ['john', 'anna', 'peter', 'linda', 'james', np.nan],
    'age': [28, 22, np.nan, 32, 45, 36],
    'salary': [50000, 54000, 58000, np.nan, 62000, 60000],
    'department': ['HR', 'IT', 'FINANCE', 'HR', np.nan, np.nan] 
}

 
df = pd.DataFrame(data)
print("Original dataset:\n", df)

 
df['age'] = df['age'].fillna(df['age'].mean())
df['name'] = df['name'].fillna('deepu')

 
df['salary'] = df['salary'].fillna(df['salary'].mean())

 
df['department'] = df['department'].fillna('Unknown')
print("\nAfter handling missing values:\n", df)

        
label_encoder = LabelEncoder()
df['dept_label'] = label_encoder.fit_transform(df['department'])
print("\nAfter label encoding:\n", df)

  
df = pd.get_dummies(df, columns=['department'])
print("\nAfter one hot encoding:\n", df)

 
scaler = MinMaxScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])

 
std_scaler = StandardScaler()
df[['age', 'salary']] = std_scaler.fit_transform(df[['age', 'salary']])
print("\nAfter feature scaling:\n", df)

 
X = df.drop(['name'], axis=1)
y = df['name']

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

 
print("\nTraining features:\n", X_train)
print("\nTesting features:\n", X_test)
print("\nTraining labels:\n", y_train)
print("\nTesting labels:\n", y_test)
