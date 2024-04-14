# %%
import pandas as pd
pd.options.display.max_rows = 300

# %% [markdown]
# INCLUSIVE
# # Joe index 0 to 38
#
# # Ben index 39 to 186
#
# # TAL index 187 to 344
#
# # Huberman index 345 to 543
#  


# %%

df1 = pd.read_csv('final_df_with_cleaned_transcripts.csv')
df1.rename(columns = {'Unnamed: 0':'Index'}, inplace = True) 




# %% [markdown]
# ### **NO NEED FOR TRAINING AND TESTING, AUDIO TO SPEECH, BECAUSE WE ARE USING PRETRAINED MODELS FOR INFERENCE WHEN DOING AUDIO TO SPEECH, WE ARE NOT RE-TRAINING OR FINETUNING AUDIO TO SPEECH TRANSFORMER, WE ARE JUST USING IT FOR INFERENCE**. 
# 

# %%
# use pre trained model transformers
from pydub import AudioSegment
import requests
import tempfile
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model
import torch
import os
import gc
"""
RUN IN TERMINAL / WINDOWS COMMAND SHELL
pip install spacy
python -m spacy download en_core_web_sm
"""


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# device = torch.device("cpu")  # Use "mps" for Apple Silicon
model.to(device)

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    """
    Convert an MP3 audio file to WAV format.

    Args:
        mp3_file_path (str): The path to the MP3 audio file.
        wav_file_path (str): The path to save the converted WAV audio file.
    """
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="wav")

def download_and_preprocess_audio(url):
    """
    Download an audio file from a given URL, convert it to WAV format, and preprocess it.

    Args:
        url (str): The URL of the audio file.

    Returns:
        audio (ndarray): The preprocessed audio as a NumPy array.
    """
    response = requests.get(url)
    if response.status_code != 200:
        return None  # Handle error appropriately in your real scenario

    # Save the MP3 audio file temporarily
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3_file:
        tmp_mp3_file.write(response.content)
        tmp_mp3_file_path = tmp_mp3_file.name

    # Convert MP3 to WAV
    tmp_wav_path = tmp_mp3_file_path.replace('.mp3', '.wav')
    convert_mp3_to_wav(tmp_mp3_file_path, tmp_wav_path)

    # Load and resample the WAV audio file using librosa
    audio, sample_rate = librosa.load(tmp_wav_path, sr=16000)
    os.remove(tmp_mp3_file_path)  # Delete the temporary MP3 file
    os.remove(tmp_wav_path)  # Delete the temporary WAV file after processing
    return audio

def test_audio_to_text(audio):
    """
    Convert audio to text using the pre-trained Wav2Vec2 model.

    Args:
        audio (ndarray): The preprocessed audio as a NumPy array.

    Returns:
        full_transcription (str): The transcribed text.
    """
    chunk_size = 16000 * 240  # 240 seconds; adjust based on your needs and system capability
    transcriptions = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        transcriptions.append(transcription.strip())

    # Combine chunk transcriptions, handling potential repetitions at chunk boundaries
    full_transcription = " ".join(transcriptions)
    return full_transcription


# Test with a small part of an audio file or a dummy audio input


def process_single_url(url):
    """
    Process a single audio file from a given URL.

    Args:
        url (str): The URL of the audio file.

    Returns:
        text (str): The transcribed text.
    """
    audio = download_and_preprocess_audio(url)
    if audio is None:
        print("Failed to process audio.")
        return 'FAILED'

    print("Audio downloaded and processed. Converting to text...")
    text = test_audio_to_text(audio)
    return text

def write_text_file(index, text):
    """
    Write the transcribed text to a text file.

    Args:
        index (int): The index of the audio file.
        text (str): The transcribed text.
    """
    folder_path = 'TAL_GENERATED_TRANS' # EDIT FOLDER PATH
    os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist
    file_path = os.path.join(folder_path, f'{index}.txt')
    with open(file_path, 'w') as file:
        file.write(text)
    print(f"Text for {index} written to {file_path}")

# %%
def process_urls(df, strt_index, end_indx):
    """
    Process a range of audio files from a DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the audio file information.
        strt_index (int): The starting index of the range (inclusive).
        end_indx (int): The ending index of the range (exclusive).
    """
    # Iterate over the DataFrame within the specified range, inclusive of both ends
    for index in range(strt_index, end_indx):  # end_indx is exclusive, so it processes up to (end_indx - 1)
        print(f"Processing index {index}")
        url = df.at[index, 'download_url']  # Using .at for direct access by index
        text = process_single_url(url)
        if text != 'FAILED':
            write_text_file(index, text)
        else:
            print(f"Failed to process URL at index {index}")

# %%
process_urls(df1, 187, 345) # TAL index +1 in end for range

# %%