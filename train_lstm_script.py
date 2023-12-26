import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # Import Data

    folder_name = "train_test_dataset"

    X_train = np.load(f'./{folder_name}/x_train.npy')
    y_train = np.load(f'./{folder_name}/y_train.npy')

    X_test = np.load(f'./{folder_name}/x_test.npy')
    y_test = np.load(f'./{folder_name}/y_test.npy')

    print(X_train.shape)
    print(y_train.shape)

    model = Sequential()
    model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=True))
    model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(units=y_train.shape[1], activation="softmax"))

    print("Compiling ...")
    # Keras optimizer defaults:
    # Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
    # RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
    # SGD    : lr=0.01,  momentum=0.,                             decay=0.
    opt = Adam(lr = 0.005)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train, epochs=150, batch_size=100)

    model.save('Raaga_Identification_Trained_Model')

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    plt.savefig("Model_Performance.png", format = "png")

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")

    with open(f'./{folder_name}/label_encoder.pkl', 'rb') as file:
        loaded_label_encoder = pickle.load(file)

    pred_labels = model.predict(X_test)
    pred_labels = np.argmax(pred_labels, axis=1)
    pred_label_names = []
    for one_hot_label in pred_labels:
        label_index = np.argmax(one_hot_label)
        original_label_name = loaded_label_encoder.inverse_transform([label_index])[0]
        pred_label_names.append(original_label_name)

    # Actual labels
    actual_labels = np.argmax(y_test, axis=1)
    actual_label_names = []
    for one_hot_label in actual_labels:
        label_index = np.argmax(one_hot_label)
        original_label_name = loaded_label_encoder.inverse_transform([label_index])[0]
        actual_label_names.append(original_label_name)

    print(confusion_matrix(actual_label_names, pred_label_names))
    print(classification_report(actual_label_names, pred_label_names))


if __name__ == "__main__":
    main()
