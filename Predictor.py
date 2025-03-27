import argparse
import numpy as np
import pandas as pd
import tensorflow as tf


def load_csv_features(csv_file):
    """
    Loads the CSV file and returns the feature matrix.
    Assumes the first column is the timestamp, which is dropped.
    """
    data = pd.read_csv(csv_file)
    # Drop the timestamp column (first column) if present.
    features = data.iloc[:, 1:].values.astype(np.float32)
    return features


def preprocess_features(features, expected_frames=100):
    """
    Preprocesses the feature matrix to match the expected number of frames.
    If the number of frames is less than expected, pads with zeros.
    If more, truncates the extra frames.

    Returns the features reshaped as (1, expected_frames, num_features)
    """
    num_frames, num_features = features.shape
    if num_frames < expected_frames:
        pad_width = expected_frames - num_frames
        pad = np.zeros((pad_width, num_features), dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)
    elif num_frames > expected_frames:
        features = features[:expected_frames, :]

    # Add batch dimension: (1, expected_frames, num_features)
    features = np.expand_dims(features, axis=0)
    return features


def load_trained_model(model_path):
    """
    Loads the saved Keras CNN model from the specified path.
    """
    model = tf.keras.models.load_model(model_path)
    return model


def predict_speech_match(model, features, threshold=0.5):
    """
    Runs inference using the model on the preprocessed features.
    Returns a boolean indicating a match (True if the predicted probability
    is above the threshold) along with the raw confidence score.
    """
    prediction = model.predict(features)
    # For binary classification, the model outputs a probability in [0,1].
    confidence = prediction[0][0]
    match = confidence >= threshold
    return match, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for speech matching using a trained CNN model.")
    parser.add_argument("model_path", help="Path to the saved CNN model (e.g., model.h5).")
    parser.add_argument("csv_file", help="Path to the CSV file with extracted audio features.")
    parser.add_argument("--expected_frames", type=int, default=100,
                        help="Number of frames expected by the model (default is 100).")
    args = parser.parse_args()

    # Load the trained model.
    model = load_trained_model(args.model_path)

    # Load and preprocess the CSV feature data.
    features = load_csv_features(args.csv_file)
    features = preprocess_features(features, expected_frames=args.expected_frames)

    # Perform inference.
    match, confidence = predict_speech_match(model, features)

    # Output the result.
    if match:
        print(f"Speech MATCHES the trained person (confidence: {confidence:.4f}).")
    else:
        print(f"Speech DOES NOT match the trained person (confidence: {confidence:.4f}).")
