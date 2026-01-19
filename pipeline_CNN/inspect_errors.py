"""
src/inspect_errors.py
Role : retrouver les exemples mal classés sur val_split.csv et sauvegarder une planche d'images
(montrant vrai label vs prédiction) pour analyser visuellement les confusions.
Execution =>>> python -m src.inspect_errors
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data import make_tf_dataset

def main():
    img_size = (32, 32)
    batch_size = 64

    val_csv = Path("data/val_split.csv")
    df = pd.read_csv(val_csv)

    model = tf.keras.models.load_model("models/baseline_cnn.keras")
    val_ds = make_tf_dataset(val_csv, img_size=img_size, batch_size=batch_size, shuffle=False)

    # prédictions pour toute la val
    y_true_list, y_pred_list = [], []
    for Xb, yb in val_ds:
        probs = model.predict(Xb, verbose=0)
        y_pred_list.append(np.argmax(probs, axis=1))
        y_true_list.append(yb.numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    wrong_idx = np.where(y_true != y_pred)[0]
    print(f"Nb erreurs total: {len(wrong_idx)} / {len(y_true)}")

    if len(wrong_idx) == 0:
        print("Aucune erreur à afficher.")
        return

    # on prend jusqu'à 25 erreurs
    n_show = min(25, len(wrong_idx))
    idxs = wrong_idx[:n_show]

    # recharger les images correspondant à ces index (via df Path)
    # Attention: df et dataset sont dans le même ordre car shuffle=False
    paths = df.loc[idxs, "Path"].values
    true_labels = y_true[idxs]
    pred_labels = y_pred[idxs]

    imgs = []
    for p in paths:
        full_path = Path("data") / p
        img = tf.io.read_file(str(full_path))
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        imgs.append(img.numpy())

    Path("figures").mkdir(exist_ok=True)

    cols = 5
    rows = int(np.ceil(n_show / cols))
    plt.figure(figsize=(14, 3 * rows))

    for i in range(n_show):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imgs[i])
        plt.title(f"T:{true_labels[i]}  P:{pred_labels[i]}")
        plt.axis("off")

    plt.tight_layout()
    out_path = Path("figures") / "misclassified_examples.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved {out_path}")

if __name__ == "__main__":
    main()
