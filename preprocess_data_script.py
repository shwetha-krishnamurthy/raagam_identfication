import os
import pickle
import numpy as np
import datetime
import preprocess_utils

def main():
    # Replace this with the path to the top-level directory
    root_directory = './carnatic'
    folder_name = "train_test_dataset"

    # Call the function and get the list of MP3 and JSON files
    file_list = preprocess_utils.collate_mp3_json_files(root_directory)

    # Create dataframe
    tuples = preprocess_utils.create_tuples(file_list)

    file_path = os.path.join(folder_name, 'tuple_list.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(tuples, file)
        
    print(f"Tuple list saved to {file_path}")

    x_train, x_test, y_train, y_test, label_encoder = preprocess_utils.create_train_test_data_from_tuple(tuples)

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

    # # Save the DataFrame to a CSV file
    # file_path = os.path.join(folder_name, 'my_dataframe.csv')
    # dataframe.to_csv(file_path, index=False)

    # print(f"DataFrame saved to {file_path}")


if __name__ == "__main__":
    main()
