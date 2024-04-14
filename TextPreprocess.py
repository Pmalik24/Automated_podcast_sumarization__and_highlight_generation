import pandas as pd
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load podcast datasets
all_podcast = pd.read_csv('final_df_raw.csv')


# Function to download audio data
def download_audio(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad status codes
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading audio from {url}: {e}")
        return None

# Function to tokenize text
def tokenize_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

# Tokenize transcript data in the all_podcast dataset
tokenized_transcripts = []

for index, row in all_podcast.iterrows():
    transcript = row['transcript']
    if isinstance(transcript, str):  # Check if transcript is a valid string
        # Perform tokenization on transcript
        tokens = tokenize_text(transcript)
        # Append tokens to list
        tokenized_transcripts.append(tokens)
    else:
        tokenized_transcripts.append([]) 

# Add tokenized transcripts to the all_podcast dataset
all_podcast['tokenized_transcript'] = tokenized_transcripts

# Save the updated dataset to a new CSV file
all_podcast.to_csv('all_podcast_with_tokens.csv', index=False)

