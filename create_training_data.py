import os
import pickle
import numpy as np
import datetime
import pandas as pd
import preprocess_utils


def main():
    # Read CSV from folder
    folder_name = "train_test_dataset"
    file_path = os.path.join(folder_name, 'tuple_list.pkl')
    with open(file_path, 'rb') as file:
        tuple_list = pickle.load(file)

    # Get training and test arrays
    x_train, x_test, y_train, y_test, label_encoder = preprocess_utils.create_train_test_data_from_tuple(tuple_list)

    # Store train and test datasets in a folder
    np.save(f'./{folder_name}/x_train.npy', x_train)
    np.save(f'./{folder_name}/x_test.npy', x_test)
    np.save(f'./{folder_name}/y_train.npy', y_train)
    np.save(f'./{folder_name}/y_test.npy', y_test)

    # Save the LabelEncoder to a file
    with open(f'./{folder_name}/label_encoder.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)

    now = datetime.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    with open("./data_preprocessing_completion.txt", 'w') as file:
        file.write(f"Data preprocessing complete at {formatted_now}")


if __name__ == "__main__":
    main()
