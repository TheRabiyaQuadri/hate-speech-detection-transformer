import torch
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import wave
import os
import numpy as np
import sounddevice as sd
from gtts import gTTS
from pydub import AudioSegment
import numpy as np
import tempfile
import wave

import speech_recognition as sr

import torch
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import seaborn as sns
import nltk
from nltk.corpus import stopwords
    
nltk.download('stopwords')

# Torch ML libraries
from transformers import BertModel, RobertaModel
from torch import nn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore')
sns.set()

import logging
from transformers import logging as transformers_logging

# Set logging level to ERROR to suppress warnings and info messages
logging.basicConfig(level=logging.ERROR)
transformers_logging.set_verbosity_error()

# pip install --upgrade transformers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HateClassifier(nn.Module):
    
    # Constructor class 
    def __init__(self, n_classes, MODEL_NAME):
        super(HateClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME, return_dict=False)
        self.roberta = RobertaModel.from_pretrained(MODEL_NAME, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size + self.roberta.config.hidden_size, n_classes)
    
    # Forward propagation method
    def forward(self, inp_ids, att_mask):
        _, bert_pooled_output = self.bert(input_ids=inp_ids, attention_mask=att_mask)
        _, roberta_pooled_output = self.roberta(input_ids=inp_ids, attention_mask=att_mask)
        
        # Concatenate or combine the pooled outputs
        combined_output = torch.cat((bert_pooled_output, roberta_pooled_output), dim=1)
        
        # Add a dropout layer 
        output = self.drop(combined_output)
        
        # Pass through a linear layer for classification
        return self.out(output)


inappropriate_words = {
    "abuse", "annoying", "ass", "bastard", "bitch", "bloody", "bozo", "crap", "fuck",
    "dumb", "fool", "freak", "hate", "hell", "idiot", "jerk", "kill", 
    "loser", "moron", "nonsense", "pervert", "pig", "retard", "rude", "scum", 
    "stupid", "trash", "ugly", "vulgar", "witch", "worthless", "slut", "whore", 
    "trash", "sicko", "psycho", "douche", "prick", "creep", "scumbag", 
    "dirtbag", "nutcase", "weirdo", "doofus", "airhead", "coward", 
    "deadbeat", "filthy", "greedy", "gross", "pathetic", "perv", "selfish", 
    "snake", "two-faced", "wimp", "bigot", "racist", "sexist", "homophobic", 
    "misogynist", "terrorist", "bully", "backstabber", "hypocrite", "ignorant", 
    "clown", "jerkoff", "scumbucket", "screw-up", "loser", "waste", "maniac", 
    "jerkwad", "knucklehead", "meanie", "blockhead", "dimwit", "bonehead", 
    "lowlife", "sleazebag", "twit", "brainless", "ignoramus", "imbecile", 
    "nitwit", "nincompoop", "halfwit", "jerkface", "mouthbreather", "punk", 
    "slob", "thug", "vermin", "wacko", "dirtbag", "fraud", "leech", "loserface"
}

def generate_sinusoidal_signal(duration=0.5, frequency=2000, sample_rate=44100, amplitude=32767):
    """
    Generate a sinusoidal wave of specified frequency and duration.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    return signal

def save_signal_to_file(signal, filename="files/beep.wav", sample_rate=44100):
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal.tobytes())

# Generate and save the signal
signal = generate_sinusoidal_signal()
save_signal_to_file(signal)

def play_signal(signal, sample_rate=44100):
    """
    Play the sinusoidal signal using the sounddevice library.
    """
    
    sd.play(signal, samplerate=sample_rate)
    sd.wait()

def speech_to_text():
    """
    Record audio from the microphone and convert it to text using SpeechRecognition.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Please speak into the microphone.")
        audio_data = recognizer.listen(source)
        print("Processing...")
        try:
            text = recognizer.recognize_google(audio_data)
            print("Recognized Text:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
    return ""

def tts_to_numpy(word, lang="en"):
    # Generate TTS for the word
    tts = gTTS(text=word, lang=lang, slow=False)
    
    # Save TTS audio to a temporary MP3 file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        tts.save(temp_file.name)
        temp_path = temp_file.name

    # Convert MP3 to WAV and NumPy format
    audio = AudioSegment.from_mp3(temp_path)
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)

    # Clean up temporary file
    os.remove(temp_path)
    
    return samples, audio.frame_rate

# Process text for inappropriate words
def process_text(input_text):
    """
    Process the input text, replacing inappropriate words with a beep sound.
    """
    audio_list = []
    words = input_text.split()
    for word in words:
        if word.lower() in inappropriate_words:
            print(f"Found inappropriate word: {word}. Playing 2kHz signal.")
            audio_list.append(signal)
        else:
            print(f"Generating TTS for word: {word}")
            samples, frame_rate = tts_to_numpy(word)
            audio_list.append(samples)

    merged_audio = np.concatenate(audio_list)

    # Save the merged audio as a WAV file
    output_file = "files/merged_audio.wav"
    with wave.open(output_file, "w") as wav_file:
        n_channels = 1  # Mono audio
        sampwidth = 2   # Sample width in bytes (16-bit audio)
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(merged_audio.tobytes())

    print(f"Merged audio saved to {output_file}")
    os.system("afplay files/merged_audio.wav")

def prediction_model(text):


    MAX_LEN = 130

    MODEL_NAME = 'bert-base-cased'

    class_names = ['Identity Hate','Insult','No Toxicity','Obscene','Severe Toxic','Threat','Toxic']

    print("Loading the model...")

    #Loading Saved Models
    bert_tokenizer = torch.load(r"bert/BertTokenizer.pth")
    loaded_model = HateClassifier(len(class_names), MODEL_NAME).to(device)
    loaded_model.load_state_dict(torch.load(r"bert/RobertaHateDetectionModel.pth", map_location=device))

    print("Model loaded successfully")

    encoded_review = bert_tokenizer.encode_plus( text, max_length=MAX_LEN, add_special_tokens=True,
        return_token_type_ids=False, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        
    inp_ids = encoded_review['input_ids'].to(device)
    att_mask = encoded_review['attention_mask'].to(device)

    op = loaded_model(inp_ids, att_mask)
    _, pred = torch.max(op, dim=1)

    print(f'Given text: {text}')
    print(f'Detected sentiment  : {class_names[pred]}')

    return class_names[pred]

if __name__ == "__main__":

    prediction_model(speech_to_text())