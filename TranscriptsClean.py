import os
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Define the directory paths for input and output
input_dir = "/Users/meagandyer/Desktop/NLPClass/Project/Transcripts"
output_dir = "/Users/meagandyer/Desktop/NLPClass/Project/Cleaned"

# Defines clean text
def clean_text(text):

    text = re.sub(r'\b\d+\s+\(\d+s\):\s*', '', text) 
    text = re.sub(r'\b\d+\s+\dm\s+\ds\b', '', text)  
    text = re.sub(r'\b\d+\s+\ds\b', '', text)  
    
    # Remove sections with questions and time markers
    text = re.sub(r'\b\d+\s+\dm\s+\ds\b.*?questions.*?\d+\s+\dm\s+\ds\b', '', text, flags=re.DOTALL)
    
    # Remove long time markers
    text = re.sub(r'\b\d+\s+\dm\s+\ds\b', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalnum()]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# List all folders in the input directory
folders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]

# Iterate over each folder
for folder in folders:
    # Create a corresponding folder in the output directory
    output_folder = os.path.join(output_dir, folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # List all files in the current folder
    files = os.listdir(os.path.join(input_dir, folder))
    
    # Initialize tqdm to track progress
    pbar = tqdm(files, desc=f"Processing {folder}", total=len(files))
    
    # Iterate over each file in the folder
    for filename in pbar:
        # Construct the full path to the current file
        file_path = os.path.join(input_dir, folder, filename)
        
        # Check if the current item is a file
        if os.path.isfile(file_path):
            # Read the content of the file
            with open(file_path, 'r') as file:
                text = file.read()
                
            # Clean the text
            cleaned_text = clean_text(text)
            
            # Write the cleaned text to a new file in the output directory
            with open(os.path.join(output_folder, filename), 'w') as file:
                file.write(cleaned_text)
            
            # Update the progress bar
            pbar.set_postfix_str(f"Processed: {filename}")
