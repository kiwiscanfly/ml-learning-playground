import pandas as pd

# Load the dataset
file_path = 'IMDB_Dataset.csv'
separator = ','
column_names = ['review', 'sentiment']
data = pd.read_csv(file_path, sep=separator, names=column_names)

def remove_longest_reviews(data, percent):
    # Calculate the length of each review
    data['review_length'] = data['review'].apply(len)
    
    # Determine the threshold length
    threshold = data['review_length'].quantile(1 - percent / 100)
    
    # Filter out the longest reviews
    filtered_data = data[data['review_length'] <= threshold]
    
    # Drop the review_length column
    filtered_data = filtered_data.drop(columns=['review_length'])
    
    return filtered_data

# Set the percentage of reviews to remove
percent_to_remove = 85

# Remove the longest reviews
filtered_data = remove_longest_reviews(data, percent_to_remove)

# Display the first few rows of the filtered dataframe
print(filtered_data.head())

# Optionally, save the filtered data to a new file
filtered_data.to_csv('imdb_sentiment_reduced.csv', sep=separator, index=False, header=False, encoding='utf-8', lineterminator='\n', quoting=1, columns=column_names)