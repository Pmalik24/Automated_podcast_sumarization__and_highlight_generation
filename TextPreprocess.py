import pandas as pd
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Load podcast datasets
joe_rogan_podcast_data = pd.read_csv('joe_rogan_podcast_dataset.csv')
TAL_podcast_data = pd.read_csv('TAL_podcast_dataset.csv')
ben_shapiro_podcast_data = pd.read_csv('ben_shapiro_podcast_dataset.csv')

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

# Process Joe Rogan podcast dataset
for index, row in joe_rogan_podcast_data.iterrows():
    audio_data = download_audio(row['download_url'])
    if audio_data:
        # Perform tokenization on transcript
        tokens = tokenize_text(row['title'])

# Process TAL podcast dataset
for index, row in TAL_podcast_data.iterrows():
    audio_data = download_audio(row['download_url'])
    if audio_data:
        # Perform tokenization on transcript
        tokens = tokenize_text(row['title'])

# Process Ben Shapiro podcast dataset
for index, row in ben_shapiro_podcast_data.iterrows():
    audio_data = download_audio(row['download_url'])
    if audio_data:
        # Perform tokenization on transcript
        tokens = tokenize_text(row['title'])
