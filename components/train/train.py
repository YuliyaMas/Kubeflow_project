"""Training a model to classify authors based on text data"""

import argparse
import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Embedding, LSTM, Dense, Bidirectional
from keras.api.optimizers import Adam
import joblib
import os


def build_author_classifier(vocab_size, num_authors, maxlen):
    # Build the classification model with embedding and bidirectional LSTM layers
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=50, input_length=maxlen),
        Bidirectional(LSTM(128)),
        Dense(64, activation='relu'),
        Dense(num_authors, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_author_classifier(data_path, metadata_path, output_model_path):
    # Load preprocessed data and metadata
    x = np.load(f"{data_path}/x.npy")
    y_auteurs = np.load(f"{data_path}/y_auteurs.npy")
    label_encoder = joblib.load(f"{metadata_path}/label_encoder.pkl")
    char_indices = joblib.load(f"{metadata_path}/char_indices.pkl")

    vocab_size = len(char_indices)
    num_authors = len(label_encoder.classes_)
    maxlen = 100

    # Build and train the model
    model = build_author_classifier(vocab_size, num_authors, maxlen)
    model.fit(x, y_auteurs, epochs=10, batch_size=128, validation_split=0.2)

    # Save the trained model
    os.makedirs(output_model_path, exist_ok=True)
    model.save(f"{output_model_path}/author_model.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--metadata-path", type=str, required=True)
    parser.add_argument("--output-model-path", type=str, required=True)
    args = parser.parse_args()
    train_author_classifier(args.data_path, args.metadata_path, args.output_model_path)
