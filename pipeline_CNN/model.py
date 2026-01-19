"""
src/model.py
son role =>> définir un CNN baseline simple (from scratch) pour classifier les 43 panneaux.
On l’utilisera dans le script d’entrainement pour compiler/entrainer/évaluer.
Exécution avec ==>> ce fichier n’est pas lancé seul, il est importé par train.py.
"""

import tensorflow as tf


def build_baseline_cnn(input_shape=(32, 32, 3), num_classes=43) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


