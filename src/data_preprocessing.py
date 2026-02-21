import os
import cv2
import random

IMG_SIZE = 224
SPLIT_RATIO = (0.8, 0.1, 0.1)

RAW_DIR = "data/raw/PetImages"
PROCESSED_DIR = "data/processed"

def create_dirs():
    for split in ["train", "val", "test"]:
        for label in ["Cat", "Dog"]:
            os.makedirs(os.path.join(PROCESSED_DIR, split, label), exist_ok=True)

def preprocess_and_split():
    create_dirs()

    for label in ["Cat", "Dog"]:
        folder = os.path.join(RAW_DIR, label)
        images = os.listdir(folder)
        random.shuffle(images)

        total = len(images)
        train_end = int(total * SPLIT_RATIO[0])
        val_end = int(total * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, split_images in splits.items():
            for img_name in split_images:
                src_path = os.path.join(folder, img_name)
                dst_path = os.path.join(PROCESSED_DIR, split, label, img_name)

                img = cv2.imread(src_path)
                if img is None:
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(dst_path, img)

if __name__ == "__main__":
    preprocess_and_split()