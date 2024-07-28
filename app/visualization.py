import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data
file_path = 'D:\\movie-recommendation-system\\data\\preprocessed_movies.csv'
data = pd.read_csv(file_path)

# Basic Data Exploration
print(data.head())
print(data.info())
print(data.describe())

# Histogram of IMDB Ratings
plt.figure(figsize=(10, 6))
sns.histplot(data['IMDB_Rating'], bins=20, kde=True)
plt.title('Distribution of IMDB Ratings')
plt.xlabel('IMDB Rating')
plt.ylabel('Frequency')
plt.show()

# Bar Chart of Movies by Release Year
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='Released_Year', order=data['Released_Year'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number of Movies Released by Year')
plt.xlabel('Released Year')
plt.ylabel('Number of Movies')
plt.show()

# Top 10 Directors by Number of Movies
top_directors = data['Director'].value_counts().head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_directors.values, y=top_directors.index, palette='viridis')
plt.title('Top 10 Directors by Number of Movies')
plt.xlabel('Number of Movies')
plt.ylabel('Director')
plt.show()

# Scatter Plot of Ratings by Released Year
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='Released_Year', y='IMDB_Rating', alpha=0.6)
plt.title('IMDB Ratings by Released Year')
plt.xlabel('Released Year')
plt.ylabel('IMDB Rating')
plt.show()
