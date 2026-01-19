"""
Inférence qualitative sur plusieurs images du dossier data/Test.
- Charge le modèle entrainé (models/baseline_cnn.keras)
- Prend N images aléatoires de data/Test (les images de tests)
- Prédit la classe + proba
- Affiche une grille si possible, sinon sauvegarde automatiquement l'image
- Affiche aussi un récap terminal (preuve reproductible pour le rapport)
"""

import random
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# CONFIGURATION=>>
MODEL_PATH = Path("models/baseline_cnn.keras")
TEST_DIR = Path("data/Test")
OUT_DIR = Path("outputs")
OUT_IMAGE = OUT_DIR / "infer_demo_grid.png"

IMG_SIZE = (32, 32)
N_IMAGES = 12  # 8 à 16 recommandé


# UTILS==>
def load_and_preprocess_image(img_path: Path) -> np.ndarray:
    """Charge et prépare une image exactement comme à l'entraînement"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def can_show_figures() -> bool:
    """
    Détecte si l'environnement permet un affichage plt.show().
    Si non (WSL/SSH), on sauvegarde uniquement.
    """
    backend = matplotlib.get_backend().lower()
    # backends non-interactifs fréquents : agg, pdf, svg, etc.
    non_interactive = ("agg" in backend) or ("pdf" in backend) or ("svg" in backend)
    return not non_interactive


# MAIN==>>
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Charger le modèle
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Modèle chargé : {MODEL_PATH}")

    #  2) Lister les images de test
    images = sorted(TEST_DIR.glob("*.png"))
    if len(images) == 0:
        raise RuntimeError("❌ Aucune image *.png trouvée dans data/Test")

    # 3) Sélection aléatoire
    selected = random.sample(images, min(N_IMAGES, len(images)))

    #  4) le batch
    batch = np.stack([load_and_preprocess_image(p) for p in selected], axis=0)

    #  5) Prédictions
    preds = model.predict(batch, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    pred_probs = np.max(preds, axis=1)

    # 6) log terminal (preuve pour rapport)
    print("\n===== Résumé prédictions (démo) =====")
    for p, c, pr in zip(selected, pred_classes, pred_probs):
        print(f"{p.name:35s} -> classe={int(c):2d} | proba={pr:.3f}")

    # 7) Grille
    cols = 4
    rows = int(np.ceil(len(selected) / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, img_path in enumerate(selected):
        img = Image.open(img_path).convert("RGB")
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"Prédit: {int(pred_classes[i])}\nP={pred_probs[i]:.3f}",
            fontsize=10
        )

    plt.tight_layout()

    # 8) Sauvegarde systématique (robuste)
    plt.savefig(OUT_IMAGE, dpi=200)
    print(f"\n Grille sauvegardée : {OUT_IMAGE}")

    # 9) Affichage si possible
    if can_show_figures():
        plt.show()
    else:
        plt.close()
        print(" Environnement non-interactif détecté : pas d'affichage, sauvegarde seulement")

if __name__ == "__main__":
    main()
