"""
Carries out hyperparameter tuning
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import ParameterGrid, KFold

from CAMELS_LOADER import CAMELS_Dataset


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(6, activation='linear')  # Output layer with 6 units for regression
    ])
    return model


def train_model(config, X_train, y_train, X_val, y_val):
    # Define model
    model = build_model(config['input_shape'])

    # Compile model
    model.compile(optimizer=config['optimizer'],
                  loss=config['loss'])

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['patience'], verbose=1)

    # Train model with early stopping
    history = model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'],
                        validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

    # Evaluate model
    loss = model.evaluate(X_val, y_val)

    return loss


def hyperparameter_tuning(config_space, X, y, num_folds=3):
    best_loss = float('inf')
    best_config = None

    kf = KFold(n_splits=num_folds)

    for config in config_space:
        loss_sum = 0

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            loss = train_model(config, X_train, y_train, X_val, y_val)
            loss_sum += loss

        mean_loss = loss_sum / num_folds

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_config = config

    return best_config, best_loss


# Generate data
dataset = CAMELS_Dataset(['B'])
dataset.generate_dataset()
dataset.normalise()
# Example data
X = dataset.train_x
y = dataset.train_y

# Example configuration space
config_space = ParameterGrid({
    'input_shape': [(256, 256, 1)],
    'optimizer': ['adam', 'sgd'],
    'loss': ['mean_squared_error'],
    'epochs': [20],
    'batch_size': [32, 64],
    'patience': [5]
})

best_config, best_loss = hyperparameter_tuning(config_space, X, y, num_folds=3)
print("Best Configuration:", best_config)
print("Best Loss:", best_loss)
