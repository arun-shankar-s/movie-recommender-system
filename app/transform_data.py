import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import hstack

# Load the preprocessed data
file_path = 'D:\\movie-recommendation-system\\data\\preprocessed_movies.csv'
data = pd.read_csv(file_path)

# Load combined features and metadata
combined_features_path = 'D:\\movie-recommendation-system\\data\\combined_features.pkl'
metadata_path = 'D:\\movie-recommendation-system\\data\\metadata.pkl'

combined_features = joblib.load(combined_features_path)
metadata = joblib.load(metadata_path)

# Extract the necessary information from the metadata
titles = metadata['titles']
tfidf_vectorizer = metadata['tfidf_vectorizer']
scaler = metadata['scaler']
onehot_encoder = metadata['onehot_encoder']

# Example usage: Transform new data using the loaded models
def transform_new_data(new_data):
    new_data['Released_Year'].fillna(new_data['Released_Year'].median(), inplace=True)
    new_data['IMDB_Rating'].fillna(new_data['IMDB_Rating'].median(), inplace=True)
    new_data['Director'].fillna('Unknown', inplace=True)
    
    new_tfidf_matrix = tfidf_vectorizer.transform(new_data['Overview'])
    new_scaled_numeric_features = scaler.transform(new_data[['Released_Year', 'IMDB_Rating']])
    new_director_encoded = onehot_encoder.transform(new_data[['Director']])
    
    new_combined_features = hstack([new_tfidf_matrix, new_scaled_numeric_features, new_director_encoded])
    return new_combined_features

# Example of transforming a new dataset (assuming new_data is a DataFrame with the same structure)
# new_combined_features = transform_new_data(new_data)
new_data = pd.DataFrame({
    'Series_Title': ['New Movie 1', 'New Movie 2'],
    'Released_Year': [2022, 2023],
    'IMDB_Rating': [8.5, 7.4],
    'Overview': ['An exciting new adventure.', 'A thrilling new drama.'],
    'Director': ['Director A', 'Director B']
})

# Transform the new data using the function
new_combined_features = transform_new_data(new_data)

print(new_combined_features)
