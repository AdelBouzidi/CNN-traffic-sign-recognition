# src/data.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

# Chemins robustes : basés sur la racine du projet (dossier parent de src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "Train.csv"



def load_train_sample(
    n_samples: int = 64,
    img_size: tuple[int, int] = (32, 32),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    charge un petit échantillon du dataset d'entrainement (baseline).
    - lire Train.csv
    - charge n_samples images
    - Redimensionne à img_size
    - Normalise les pixels dans [0, 1]
    Retourne:
      X: (n_samples, H, W, 3) float32
      y: (n_samples,) int64
    """
    df = pd.read_csv(TRAIN_CSV)

    # Mélanger pour obtenir un échantillon aléatoire reproductible
    df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)

    X_list = []
    y_list = []

    for _, row in df.iterrows():
        img_path = DATA_DIR / row["Path"]
        label = int(row["ClassId"])

        # Lecture image (OpenCV lit en BGR)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Image illisible: {img_path}")

        # Conversion BGR -> RGB (plus standard pour ML)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Redimensionnement
        img_resized = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_AREA)

        # Normalisation
        img_resized = img_resized.astype(np.float32) / 255.0

        X_list.append(img_resized)
        y_list.append(label)

    X = np.stack(X_list, axis=0)  # (N, H, W, 3)
    y = np.array(y_list, dtype=np.int64)

    return X, y


def load_image_with_roi(row: pd.Series, img_size: tuple[int, int] = (32, 32)) -> np.ndarray:
    """
    Charge une image, applique un crop ROI (panneau), redimensionne, normalise.
    Retour: (H, W, 3) float32 dans [0,1]
    """
    img_path = DATA_DIR / row["Path"]

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image illisible: {img_path}")

    # ROI crop (attention: slicing [y1:y2, x1:x2])
    x1, y1, x2, y2 = int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"])
    crop = img_bgr[y1:y2, x1:x2]

    # Si ROI vide (rare), fallback sur image entière
    if crop.size == 0:
        crop = img_bgr

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, img_size, interpolation=cv2.INTER_AREA)
    crop_resized = crop_resized.astype(np.float32) / 255.0
    return crop_resized


def load_dataset_from_csv(
    csv_path: Path,
    n_samples: int | None = None,
    img_size: tuple[int, int] = (32, 32),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge un dataset complet (ou un sous-échantillon) depuis un CSV split.
    Applique ROI crop + resize + normalisation.
    Retour: X (N,H,W,3), y (N,)
    """
    df = pd.read_csv(csv_path)

    if n_samples is not None:
        df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)

    X_list, y_list = [], []
    for _, row in df.iterrows():
        X_list.append(load_image_with_roi(row, img_size=img_size))
        y_list.append(int(row["ClassId"]))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y



def make_tf_dataset(
    csv_path: Path,
    img_size: tuple[int, int] = (32, 32),
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Construit un tf.data.Dataset à partir d'un CSV (train_split ou val_split).
    Utilise OpenCV via tf.py_function pour appliquer ROI crop + resize.
    """
    df = pd.read_csv(csv_path)
    paths = (DATA_DIR / df["Path"]).astype(str).values
    labels = df["ClassId"].astype("int64").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 5000), seed=seed, reshuffle_each_iteration=True)

    def _load(path, label):
        def _py_load(p):
        # p est un EagerTensor (tf.string) -> bytes -> str
            p = p.numpy().decode("utf-8")

            img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Image illisible: {p}")

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_AREA)
            img_resized = img_resized.astype(np.float32) / 255.0
            return img_resized

        img = tf.py_function(func=_py_load, inp=[path], Tout=tf.float32)
        img.set_shape([img_size[1], img_size[0], 3])  # (H,W,3)
        return img, label
    

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

