import numpy as np
import pickle
import librosa
from keras.models import load_model
from collections import Counter

# Replace this with an import from the preprocess_data module once you convert it.

def cut_song(song):
    """
    Cut each song in pieces of 100.000 before doing anything else
    """
    
    start = 0
    end = len(song)
    
    song_pieces = []
    
    while start + 100000 < end:
        song_pieces.append(song[start:start+100000])
        start += 100000
    
    return song_pieces


def prepare_song(song_path):
    """
    Get song spectrograms.
    """
    
    list_matrices = []
    y,sr = librosa.load(song_path, sr=22050)
    song_pieces = cut_song(y)
    for song_piece in song_pieces:
        melspect = librosa.feature.melspectrogram(song_piece)
        list_matrices.append(melspect)

    return list_matrices


def process_audio(file_path):
    X_test = prepare_song(file_path)
    folder_name = "train_test_dataset"

    with open(f'./{folder_name}/label_encoder.pkl', 'rb') as file:
        loaded_label_encoder = pickle.load(file)

    model = load_model('Raaga_Identification_Trained_Model')
    pred_labels = []

    for snippet in X_test:
        snippet = snippet.reshape(1, snippet.shape[0], snippet.shape[1])
        prediction = model.predict(snippet, verbose=0)
        label_index = np.argmax(prediction)
        original_label_name = loaded_label_encoder.inverse_transform([label_index])[0]
        pred_labels.append(original_label_name)

    # Count the frequency of each element
    frequency = Counter(pred_labels)

    # Sort items by frequency (highest first)
    # Note: items() returns pairs (element, count), and lambda defines the sort key
    sorted_items = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

    # Extracting just the elements in sorted order
    sorted_elements = [item[0] for item in sorted_items]

    return sorted_elements
