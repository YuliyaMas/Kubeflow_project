"""Loading a model to predict the author of a text and generates new text in the author's style"""

import argparse
import numpy as np
from keras.api.models import load_model
import joblib

def generate_text(model, input_text, char_indices, indices_char, maxlen, length=200):
    # Prepare input text for prediction
    x_pred = np.zeros((1, maxlen), dtype=np.int32)
    for t, char in enumerate(input_text[-maxlen:]):
        x_pred[0, t] = char_indices.get(char, 0)

    generated_text = input_text
    for _ in range(length):
        char_probs = model.predict(x_pred, verbose=0)
        next_index = np.argmax(char_probs[0])
        next_char = indices_char[next_index]

        generated_text += next_char

        # Update x_pred with the new character
        x_pred = np.roll(x_pred, -1)
        x_pred[0, -1] = next_index

    return generated_text

def predict_and_generate(model_path, metadata_path, input_text):
    # Load the trained model and metadata
    model = load_model(model_path)
    char_indices = joblib.load(f"{metadata_path}/char_indices.pkl")
    indices_char = joblib.load(f"{metadata_path}/indices_char.pkl")

    maxlen = 100  # Sequence length used during training
    generated_text = generate_text(model, input_text, char_indices, indices_char, maxlen)

    print("Generated Text:", generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--metadata-path", type=str, required=True)
    parser.add_argument("--input-text", type=str, required=True)
    args = parser.parse_args()
    predict_and_generate(args.model_path, args.metadata_path, args.input_text)
