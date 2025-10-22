import tensorflow as tf
import gin

@gin.configurable
def lstm_model(input_shape, n_classes, lstm_units=128, dense_units=64, dropout_rate=0.5):
    """
    Defines a simple LSTM-based model.

    Parameters:
        input_shape (tuple): Input shape of the neural network (e.g., (250, 6)).
        n_classes (int): Number of output classes.
        lstm_units (int): Number of units in the LSTM layer.
        dense_units (int): Number of units in the dense layer.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        keras.Model: A Keras model object.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Add LSTM layer
    x = tf.keras.layers.LSTM(lstm_units, return_sequences=False)(inputs)

    # Fully connected layer with dropout
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='lstm_model')


@gin.configurable
def gru_model(input_shape, n_classes, gru_units=128, dense_units=64, dropout_rate=0.5):
    """
    Defines a GRU-based model for human activity recognition.

    Parameters:
        input_shape (tuple): Input shape of the neural network (e.g., (250, 6)).
        n_classes (int): Number of output classes.
        gru_units (int): Number of units in the GRU layer.
        dense_units (int): Number of units in the dense layer.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        keras.Model: A Keras model object.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Add GRU layer
    x = tf.keras.layers.GRU(gru_units, return_sequences=False)(inputs)

    # Fully connected layer with dropout
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='gru_model')

@gin.configurable
def cnn_1d_model(input_shape, n_classes, conv_filters=64, kernel_size=3, pool_size=2, dense_units=64, dropout_rate=0.5):
    """
    Defines a 1D CNN-based model for human activity recognition.

    Parameters:
        input_shape (tuple): Input shape of the neural network (e.g., (250, 6)).
        n_classes (int): Number of output classes.
        conv_filters (int): Number of filters in the convolutional layer.
        kernel_size (int): Kernel size for the convolutional layer.
        pool_size (int): Pool size for the max-pooling layer.
        dense_units (int): Number of units in the dense layer.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        keras.Model: A Keras model object.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Add 1D Convolution and pooling layers
    x = tf.keras.layers.Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(x)
    x = tf.keras.layers.Flatten()(x)

    # Fully connected layer with dropout
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn_1d_model')