"""
src/train.py
Role == entrainer le CNN baseline sur train_split.csv et val_split.csv, puis sauvegarder le modèle.
Execution avec : python -m src.train
"""

from pathlib import Path
import tensorflow as tf

from src.data import make_tf_dataset
from src.model import build_baseline_cnn

import json
from pathlib import Path


def main():
    img_size = (32, 32)
    batch_size = 64
    num_classes = 43
    epochs = 10

    train_csv = Path("data/train_split.csv")
    val_csv = Path("data/val_split.csv")

    train_ds = make_tf_dataset(train_csv, img_size=img_size, batch_size=batch_size, shuffle=True)
    val_ds = make_tf_dataset(val_csv, img_size=img_size, batch_size=batch_size, shuffle=False)

    model = build_baseline_cnn(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("models/baseline_cnn.keras", monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    


    history_dict = history.history
    Path("reports").mkdir(exist_ok=True)

    with open("reports/training_history.json", "w") as f:
        json.dump(history_dict, f)

    print(" Historique d'entraînement sauvegardé")

        
    

    # Sauvegarde finale (poids best déjà sauvegardés par checkpoint)
    Path("models").mkdir(exist_ok=True)
    model.save("models/baseline_cnn_last.keras")
    print(" Saved models/baseline_cnn.keras (best) and models/baseline_cnn_last.keras (last)")


if __name__ == "__main__":
    main()
