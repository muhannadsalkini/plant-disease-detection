"""
src/data_pipeline.py
--------------------
Data loading, stratified splitting, augmentation, and generator creation
for the PlantVillage dataset.

All images are loaded from the directory structure:
    data/PlantVillage/<ClassName>/<image>.jpg
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Allow importing config from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe(data_dir: str = config.DATA_DIR) -> pd.DataFrame:
    """
    Walk the dataset directory and return a DataFrame with columns:
        filepath  (str)  – absolute path to each image file
        label     (str)  – class name (folder name)

    Returns
    -------
    df : pd.DataFrame
    class_names : list[str]  – sorted list of unique class names
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Run  python download_dataset.py  first."
        )

    records = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append({"filepath": str(img_path), "label": class_dir.name})

    if not records:
        raise ValueError(
            f"No images found under {data_dir}. "
            "Check that the dataset was extracted correctly."
        )

    df = pd.DataFrame(records)
    class_names = sorted(df["label"].unique().tolist())
    print(f"[DataPipeline] Total images : {len(df):,}")
    print(f"[DataPipeline] Total classes: {len(class_names)}")
    return df, class_names


# ─────────────────────────────────────────────────────────────────────────────
# 2. STRATIFIED SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_SPLIT,
    val_ratio:   float = config.VAL_SPLIT,
    seed:        int   = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train / val / test DataFrames using stratified sampling.

    Returns
    -------
    df_train, df_val, df_test
    """
    test_ratio = 1.0 - train_ratio - val_ratio  # e.g. 0.15

    df_train, df_temp = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df["label"],
        random_state=seed,
    )
    # val and test from the remaining (df_temp)
    relative_val = val_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(
        df_temp,
        train_size=relative_val,
        stratify=df_temp["label"],
        random_state=seed,
    )

    print(f"[DataPipeline] Train : {len(df_train):,}  "
          f"Val : {len(df_val):,}  "
          f"Test : {len(df_test):,}")
    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASS WEIGHTS (for imbalanced dataset)
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(df_train: pd.DataFrame, class_names: list) -> dict:
    """
    Compute inverse-frequency class weights to handle class imbalance.

    Returns
    -------
    class_weight_dict : {class_index: weight}
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels = df_train["label"].values
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(class_names),
        y=labels,
    )
    return {i: w for i, w in enumerate(weights)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. IMAGE DATA GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def make_generators(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    batch_size: int = config.BATCH_SIZE,
    image_size: tuple = config.IMAGE_SIZE,
):
    """
    Build Keras ImageDataGenerators with augmentation on train set only.

    Augmentations (from paper §IV.E):
        - Random horizontal & vertical flip
        - Rotation ±20°
        - Zoom ±20%
        - Width / height shift ±10%
        - Brightness ±20%

    Returns
    -------
    train_gen, val_gen, test_gen
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.10,
        height_shift_range=0.10,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    def _flow(gen, df, shuffle, seed=config.RANDOM_SEED):
        return gen.flow_from_dataframe(
            dataframe=df,
            x_col="filepath",
            y_col="label",
            target_size=image_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=shuffle,
            seed=seed,
        )

    train_gen = _flow(train_datagen, df_train, shuffle=True)
    val_gen   = _flow(eval_datagen,  df_val,   shuffle=False)
    test_gen  = _flow(eval_datagen,  df_test,  shuffle=False)

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONVENIENCE FUNCTION — load everything at once
# ─────────────────────────────────────────────────────────────────────────────

def load_data(batch_size: int = config.BATCH_SIZE):
    """
    One-call wrapper that returns all generators and metadata needed for training.

    Returns
    -------
    train_gen, val_gen, test_gen, class_names, class_weights, df_test
    """
    df, class_names = build_dataframe()
    df_train, df_val, df_test = stratified_split(df)
    class_weights = compute_class_weights(df_train, class_names)
    train_gen, val_gen, test_gen = make_generators(
        df_train, df_val, df_test, batch_size=batch_size
    )
    return train_gen, val_gen, test_gen, class_names, class_weights, df_test


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_gen, val_gen, test_gen, class_names, cw, df_test = load_data()
    batch_x, batch_y = next(iter(train_gen))
    print(f"\nSample batch — X: {batch_x.shape}  Y: {batch_y.shape}")
    print(f"First 5 classes: {class_names[:5]}")
