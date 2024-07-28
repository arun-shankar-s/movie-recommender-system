import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import hstack
import joblib

# Load the preprocessed data
file_path = 'D:\\movie-recommendation-system\\data\\preprocessed_movies.csv'
data = pd.read_csv(file_path)

# Fill missing values if any (specific strategy can be applied here)
data['Released_Year'].fillna(data['Released_Year'].median(), inplace=True)
data['IMDB_Rating'].fillna(data['IMDB_Rating'].median(), inplace=True)
data['Director'].fillna('Unknown', inplace=True)

# TF-IDF Vectorization for Overviews
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Overview'])

# Scaling the Released Year and IMDB Rating
scaler = StandardScaler()
scaled_numeric_features = scaler.fit_transform(data[['Released_Year', 'IMDB_Rating']])

# One-hot Encoding for Director
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
director_encoded = onehot_encoder.fit_transform(data[['Director']])

# Extract 'Series_Title' (which is the movie title) as it is without transformation
titles = data['Series_Title'].values

# Extract 'Genre' for emotion detection
genres = data['Genre'].values

# Combining all features into a single matrix
combined_features = hstack([tfidf_matrix, scaled_numeric_features, director_encoded])

# Save combined features and other necessary data for model building
combined_features_path = 'D:\\movie-recommendation-system\\data\\combined_features.pkl'
joblib.dump(combined_features, combined_features_path)

# Save other necessary information for later use
metadata = {
    'titles': titles,
    'genres': genres,
    'tfidf_vectorizer': tfidf_vectorizer,
    'scaler': scaler,
    'onehot_encoder': onehot_encoder
}
metadata_path = 'D:\\movie-recommendation-system\\data\\metadata.pkl'
joblib.dump(metadata, metadata_path)

print("Feature engineering completed and saved.")
