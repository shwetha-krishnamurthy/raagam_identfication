import librosa
import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import utils


def collate_mp3_json_files(root_dir):
    """
    Traverses through all directories and subdirectories starting from 'root_dir'
    and returns a list of paths to all the MP3 and JSON files found in a list of dictionaries.
    """
    
    mp3_json_files = []
    for root, dirs, files in os.walk(root_dir):
        curr_dict = {}
        for file in files:
            if file.endswith('.mp3'):
                mp3_file = os.path.join(root, file)
                curr_dict['mp3_file'] = mp3_file
                
            if file.endswith('.json'):
                json_file = os.path.join(root, file)
                curr_dict['json_file'] = json_file

        if curr_dict:
            mp3_json_files.append(curr_dict)
    return mp3_json_files


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

def create_tuples(file_paths_list):
    tuple_list = []

    for file in file_paths_list:
        file_path = file['json_file']

        with open(file_path, 'r') as f:
            data = json.load(f)

        if data['raaga'] and 'mp3_file' in file:
            raaga_name = data['raaga'][0]['name']
            song_matrix = prepare_song(file['mp3_file'])
        
            for snippet in song_matrix:
                tuple_list.append((snippet, raaga_name))
    
    return tuple_list


def create_dataframe(file_paths_list):
    """
    Create a pandas dataframe with the x and y variables.
    x: song_matrix containing the spectrogram
    y: raaga name
    """
    
    # Define your column names
    columns = ['x', 'y']

    # Create an empty DataFrame with these columns
    df = pd.DataFrame(columns = columns)

    for file in file_paths_list:
        file_path = file['json_file']
        # Open the file and read the JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        if data['raaga'] and 'mp3_file' in file:
            raaga_name = data['raaga'][0]['name']
            song_matrix = prepare_song(file['mp3_file'])
        
            for snippet in song_matrix:
                df.loc[len(df)] = [snippet, raaga_name]

    return df


def create_train_test_data_from_tuple(tuple_list):
    """
    1. Encode the Categorical Variable: Since y is categorical, it should be one-hot
        encoded if it's not numerical.
    2. Reshape Input Data for LSTM: LSTM models in libraries like
        Keras expect input to be in a 3D array format of [samples, time steps, features].
    3. Split Data into Training and Testing Sets: Typically, data is
        divided into training and testing sets to evaluate the model's performance.
    4. Normalize the Data: Depending on the range of your independent
        variable x, you might need to normalize it, especially if you're dealing with varying scales.
    """
    # Separate the independent and dependent variables
    X = [item[0] for item in tuple_list if item[1] != "Rāgamālika"]  # Independent variables (2D matrices)
    y = [item[1] for item in tuple_list if item[1] != "Rāgamālika"]  # Dependent variables (strings)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    
    # One-hot encoding
    onehot_encoded = utils.to_categorical(integer_encoded)
    
    # padded_data = pad_sequences(X, padding='post')
    X = np.array(X)  # Convert column to list and then to numpy array
    
    print("Here 1")
    # X = np.array(X)

    # Example: Reshape X if necessary (depends on your data)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    print(X.shape)
    X = X.astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(X, onehot_encoded, test_size=0.2, random_state=0)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test, label_encoder


def create_train_test_data(df):
    """
    1. Encode the Categorical Variable: Since y is categorical, it should be one-hot
        encoded if it's not numerical.
    2. Reshape Input Data for LSTM: LSTM models in libraries like
        Keras expect input to be in a 3D array format of [samples, time steps, features].
    3. Split Data into Training and Testing Sets: Typically, data is
        divided into training and testing sets to evaluate the model's performance.
    4. Normalize the Data: Depending on the range of your independent
        variable x, you might need to normalize it, especially if you're dealing with varying scales.
    """

    # Filtering out Ragamalika songs since they have multiple raagas
    df = df[df['y'] != "Rāgamālika"]
    
    # Label Encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df['y'])
    
    # One-hot encoding
    onehot_encoded = utils.to_categorical(integer_encoded)

    padded_data = pad_sequences(df['x'], padding='post')
    X = np.array(padded_data.tolist())  # Convert column to list and then to numpy array

    # Example: Reshape X if necessary (depends on your data)
    print(X.shape)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    X_train, X_test, y_train, y_test = train_test_split(X, onehot_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder
