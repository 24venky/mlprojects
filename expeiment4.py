import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
     'student': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
     'MATH': [88, 92, 80, 89, 100, 67, 78, 85],
     'science': [90, 85, 88, 95, 70, 75, 80, 92],
     'english': [70, 78, 85, 89, 60, 65, 72, 80] 
}

df = pd.DataFrame(data)

# 1. Line Plot
plt.figure(figsize=(8, 5))

plt.plot(df['student'], df['MATH'], marker='o', label='math')
plt.plot(df['student'], df['science'], marker='s', label='science')
plt.plot(df['student'], df['english'], marker='^', label='english')
plt.title("Line plot of student scores")
plt.xlabel("Students")
plt.ylabel("Scores")
plt.legend()
plt.show()

# 2. Bar Chart
plt.figure(figsize=(8, 5))
plt.bar(df['student'], df['MATH'], color='skyblue')
plt.title("Bar chart - Math scores")
plt.xlabel("Student")
plt.ylabel("Scores")

plt.show()

# 3. Histogram
plt.figure(figsize=(8, 5))
plt.hist(df['MATH'], bins=5, color='lightgreen', edgecolor='yellow')
plt.title("Histogram of math scores")
plt.xlabel("Score range")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(df['MATH'], df['science'], color='red')
plt.title("Scatter plot - Math vs Science")
plt.xlabel("Math score")
plt.ylabel("Science score")
plt.show()

# 5. Box Plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[['MATH', 'science', 'english']])
plt.title("Box plot of subject scores")
plt.ylabel("Score")
plt.show()

# 6. Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df[['MATH', 'science', 'english']].corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap correlation between subjects")
plt.show()
