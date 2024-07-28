import pandas as pd

def load_movie_dataset(filepath):
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for enc in encodings:
        try:
            data = pd.read_csv(filepath, encoding=enc)
            print(f"Successfully read the file with {enc} encoding")
            return data
        except UnicodeDecodeError as e:
            print(f"Failed to read the file with {enc} encoding: {e}")
    raise ValueError("Unable to read the file with the provided encodings")

def preprocess_data(data):
    # Check for missing values
    print("Missing values before preprocessing:")
    print(data.isnull().sum())
    
    # Convert IMDB_Rating to numeric, if not already
    data['IMDB_Rating'] = pd.to_numeric(data['IMDB_Rating'], errors='coerce')
    
    # Drop rows with missing values in the relevant columns
    data = data.dropna(subset=['Series_Title', 'Released_Year', 'IMDB_Rating', 'Overview', 'Director', 'Genre'])
    
    # Keep only the relevant columns
    data = data[['Series_Title', 'Released_Year', 'IMDB_Rating', 'Overview', 'Director', 'Genre']]
    
    print("Missing values after preprocessing:")
    print(data.isnull().sum())
    
    return data

# Path to your dataset
dataset_path = 'D:/movie-recommendation-system/data/imdb_top_1000.csv'  # Adjust the path as needed
movie_data = load_movie_dataset(dataset_path)
movie_data = preprocess_data(movie_data)

print(movie_data.head())  # Print the cleaned data

def save_preprocessed_data(data, output_filepath):
    """
    Save the preprocessed dataset to a CSV file.
    """
    data.to_csv(output_filepath, index=False)
    print(f"Data saved to {output_filepath}")

output_filepath = 'D:\\movie-recommendation-system\\data\\preprocessed_movies.csv'
save_preprocessed_data(movie_data, output_filepath)
