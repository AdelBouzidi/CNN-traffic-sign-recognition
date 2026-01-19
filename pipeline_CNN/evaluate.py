"""
src/evaluate.py
son role es d'evaluer le modèle sauvegardé sur val_split.csv, calculer accuracy + classification report
et générer une matrice de confusion (image) + un CSV des confusions principales.
Exécution par =>> python -m src.evaluate
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report

from src.data import make_tf_dataset


def main():
    img_size = (32, 32)
    batch_size = 64

    val_csv = Path("data/val_split.csv")
    val_ds = make_tf_dataset(val_csv, img_size=img_size, batch_size=batch_size, shuffle=False)

    model_path = Path("models/baseline_cnn.keras")  # meilleur checkpoint
    model = tf.keras.models.load_model(model_path)
    print(f" Loaded model: {model_path}")

    # ---------
    # Récupérer y_true et y_pred
    # ---------------------------------
    y_true_list = []
    y_pred_list = []

    for Xb, yb in val_ds:
        probs = model.predict(Xb, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true_list.append(yb.numpy())
        y_pred_list.append(preds)

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    acc = (y_true == y_pred).mean()
    print(f"\n Validation accuracy (recomputed) = {acc:.6f}")

    # ---------
    # Classification report
    # ---------
    report = classification_report(y_true, y_pred, digits=4)
    Path("reports").mkdir(exist_ok=True)
    (Path("reports") / "classification_report.txt").write_text(report, encoding="utf-8")
    print(" Saved reports/classification_report.txt")

    # ---------
    # Confusion matrix
    # ---------
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(43))

    Path("figures").mkdir(exist_ok=True)
    fig_path = Path("figures") / "confusion_matrix.png"

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Validation)")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f" Saved {fig_path}")

    # ---------
    # Top confusions (hors diagonale)
    # ---------
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)

    pairs = []
    for i in range(cm_off.shape[0]):
        for j in range(cm_off.shape[1]):
            if cm_off[i, j] > 0:
                pairs.append((i, j, int(cm_off[i, j])))

    pairs.sort(key=lambda t: t[2], reverse=True)
    top = pairs[:30]

    df_top = pd.DataFrame(top, columns=["true_class", "pred_class", "count"])
    df_top.to_csv("reports/top_confusions.csv", index=False)
    print(" Saved reports/top_confusions.csv")

    # Affiche juste un aperçu console
    print("\nTop 10 confusions :")
    print(df_top.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
