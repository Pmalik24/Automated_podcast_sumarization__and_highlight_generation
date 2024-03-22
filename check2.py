from pydub import AudioSegment
import requests
import tempfile
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model
import torch
import os
"""
RUN IN TERMINAL / WINDOWS COMMAND SHELL
pip install spacy
python -m spacy download en_core_web_sm
"""

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
device = torch.device("cpu")  # Use "mps" for Apple Silicon
model.to(device)

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="wav")

def download_and_preprocess_audio(url):
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
    chunk_size = 16000 * 120  # 120 seconds; adjust based on your needs and system capability
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
    audio = download_and_preprocess_audio(url)
    if audio is None:
        print("Failed to process audio.")
        return
    
    print("Audio downloaded and processed. Converting to text...")
    text = test_audio_to_text(audio)
    print("Converted Text:\n", text)


def process_single_file(filepath):
    print("Loading audio file...")
    # Load and resample the WAV audio file using librosa
    audio, sample_rate = librosa.load(filepath, sr=16000)
    
    if audio is None:
        print("Failed to load audio.")
        return
    
    print("Audio loaded. Converting to text...")
    text = test_audio_to_text(audio)
    print("Converted Text:\n", text)


test_url = 'https://claritaspod.com/measure/arttrk.com/p/24FDE/verifi.podscribe.com/rss/p/pfx.vpixl.com/2jSe3/prfx.byspotify.com/e/dts.podtrac.com/redirect.mp3/mgln.ai/e/121/injector.simplecastaudio.com/01514e65-f508-4e0c-99d9-aad07cea61ff/episodes/c3d0405a-7fb4-4710-a02d-6915b41482e6/audio/128/default.mp3?aid=rss_feed&awCollectionId=01514e65-f508-4e0c-99d9-aad07cea61ff&awEpisodeId=c3d0405a-7fb4-4710-a02d-6915b41482e6&feed=C0fPpQ64'
process_single_url(test_url)