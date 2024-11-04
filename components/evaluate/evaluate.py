"""Loading the trained model and evaluates it on test data"""

import argparse
import numpy as np
from keras.api.models import load_model
from sklearn.model_selection import train_test_split

def evaluate_model(data_path, model_path):
    # Load preprocessed data and trained model
    x = np.load(f"{data_path}/x.npy")
    y_auteurs = np.load(f"{data_path}/y_auteurs.npy")
    model = load_model(model_path)

    # Split data and evaluate model on the test set
    _, x_test, _, y_test = train_test_split(x, y_auteurs, test_size=0.2, random_state=42)
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()
    evaluate_model(args.data_path, args.model_path)

