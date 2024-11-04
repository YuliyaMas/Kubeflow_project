"""This script loads raw text data, encodes authors and characters, and prepares data for training"""

import argparse
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def preprocess_data(input_path, output_data_path, output_metadata_path):
    # Load raw text files organized by author
    auteurs_textes = {}
    for fichier in glob.glob(f"{input_path}/*"):
        auteur = fichier.split("_")[0].split("\\")[-1]
        with open(fichier, "r", encoding="utf-8") as fichier_input:
            texte = fichier_input.read()
            if auteur in auteurs_textes:
                auteurs_textes[auteur].append(texte[:1000])
            else:
                auteurs_textes[auteur] = [texte[:1000]]

    # Encode authors and characters
    auteurs = list(auteurs_textes.keys())
    textes = [texte for liste_textes in auteurs_textes.values() for texte in liste_textes]
    label_encoder = LabelEncoder()
    auteurs_encodes = label_encoder.fit_transform(auteurs)

    # Save metadata
    os.makedirs(output_metadata_path, exist_ok=True)
    joblib.dump(label_encoder, f"{output_metadata_path}/label_encoder.pkl")

    # Prepare sequences for training
    all_text = ' '.join(textes)
    chars = sorted(list(set(all_text)))
    char_indices = {c: i for i, c in enumerate(chars)}
    indices_char = {i: c for i, c in enumerate(chars)}

    maxlen = 100
    step = 5
    phrases, auteur_labels = [], []
    for i, auteur in enumerate(auteurs):
        for text in auteurs_textes[auteur]:
            for char_index in range(0, len(text) - maxlen, step):
                phrases.append(text[char_index: char_index + maxlen])
                auteur_labels.append(i)

    # Encode sequences
    x = np.zeros((len(phrases), maxlen), dtype=np.int32)
    y_auteurs = np.array(auteur_labels)

    for i, phrase in enumerate(phrases):
        for t, char in enumerate(phrase):
            x[i, t] = char_indices.get(char, 0)

    # Save processed data
    os.makedirs(output_data_path, exist_ok=True)
    np.save(f"{output_data_path}/x.npy", x)
    np.save(f"{output_data_path}/y_auteurs.npy", y_auteurs)
    joblib.dump(char_indices, f"{output_metadata_path}/char_indices.pkl")
    joblib.dump(indices_char, f"{output_metadata_path}/indices_char.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-data-path", type=str, required=True)
    parser.add_argument("--output-metadata-path", type=str, required=True)
    args = parser.parse_args()
    preprocess_data(args.input_path, args.output_data_path, args.output_metadata_path)
