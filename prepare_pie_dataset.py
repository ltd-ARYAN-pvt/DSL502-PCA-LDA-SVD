from pathlib import Path
import numpy as np
from PIL import Image

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATASET_PATH = Path("data/raw/PCA_PIE_Data")
OUTPUT_PATH = Path("data/processed")

PERSONS = [
    "Person1",
    "Person2",
    "Person9",
    "Person19",
    "Person46"
]

CONDITIONS = ["lights", "nolights"]

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Function to load and flatten images
# --------------------------------------------------

def load_images(condition):

    images = []
    labels = []
    flash_labels = []

    for person in PERSONS:

        person_path = DATASET_PATH / condition / person

        image_files = sorted(person_path.glob("flash_*.jpg"))

        for image_path in image_files:

            with Image.open(image_path) as image:

                image = image.convert("L")

                image_array = np.asarray(
                    image,
                    dtype=np.float64
                )

                # Flatten 100 x 100 into 10000
                flattened_image = image_array.flatten()

                images.append(flattened_image)

                labels.append(person)

                flash_number = int(
                    image_path.stem.split("_")[1]
                )

                flash_labels.append(flash_number)

    X = np.stack(images)

    y = np.array(labels)

    flash = np.array(flash_labels)

    return X, y, flash


# --------------------------------------------------
# 3. Construct both datasets
# --------------------------------------------------

datasets = {}

for condition in CONDITIONS:

    X, y, flash = load_images(condition)

    datasets[condition] = X

    print(f"\nCondition: {condition}")
    print("Matrix shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Flash labels shape:", flash.shape)
    print("Data type:", X.dtype)

    assert X.shape == (105, 10000)

    assert len(y) == 105

    assert len(flash) == 105


# --------------------------------------------------
# 4. Save processed datasets
# --------------------------------------------------

np.savez_compressed(
    OUTPUT_PATH / "pie_dataset.npz",
    X_lights=datasets["lights"],
    X_nolights=datasets["nolights"],
)

print("\nDataset matrices successfully created!")

print("Saved at:", OUTPUT_PATH / "pie_dataset.npz")