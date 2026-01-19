"""
génére les courbes d'entrainement du modèle CNN :
(Accuracy et val_accuracy)
(Loss et val_loss)

Le script charge l'historique d'entrainement sauvegardé
et produit deux figures ( utilisées dans le rapport final ).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import json

# =========================
# CONFIGURATION
# =========================
HISTORY_PATH = Path("reports/training_history.json")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

ACC_FIG = FIG_DIR / "train_val_accuracy.png"
LOSS_FIG = FIG_DIR / "train_val_loss.png"


def main():
    # Charger l'historique
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)

    # ================================================= Accuracy =====
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], label="Train accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Époque")
    plt.ylabel("Accuracy")
    plt.title("Évolution de l'accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ACC_FIG, dpi=200)
    plt.close()

    # ================== Loss =========================================
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Époque")
    plt.ylabel("Loss")
    plt.title("Évolution de la loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_FIG, dpi=200)
    plt.close()

    print(" Figures générées :")
    print(f" - {ACC_FIG}")
    print(f" - {LOSS_FIG}")

if __name__ == "__main__":
    main()
