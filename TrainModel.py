import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization


def create_cnn_model(input_shape):
    """
    Creates a CNN model to learn a reduced feature representation from audio feature sequences.

    Parameters:
        input_shape (tuple): Shape of each input sample, e.g., (num_frames, num_features).
                             For example, if each sample has 100 frames and 26 features, input_shape = (100, 26).

    Returns:
        model: A compiled Keras model.
    """
    model = Sequential()

    # First convolution: learn local patterns in time
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Second convolution: deeper features
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Third convolution: further abstraction
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Global average pooling reduces the time dimension,
    # yielding a fixed-length output regardless of input length.
    model.add(GlobalAveragePooling1D())

    # A dense layer to combine features into a lower-dimensional representation.
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization

    # Final classification layer: sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model using binary cross-entropy loss and an Adam optimizer.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Example usage:
# Assume each sample is a fixed-length sequence of 100 frames, each with 26 features.
input_shape = (100, 26)
model = create_cnn_model(input_shape)
model.summary()
