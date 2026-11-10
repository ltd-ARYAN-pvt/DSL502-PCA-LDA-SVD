from pathlib import Path
from PIL import Image
import numpy as np

# Dataset location
DATASET_PATH = Path("data/raw/PCA_PIE_Data")

lighting_conditions = ["lights", "nolights"]

print("=" * 60)
print("PIE FACE DATASET INSPECTION")
print("=" * 60)

for condition in lighting_conditions:
    print(f"\nLighting condition: {condition}")

    condition_path = DATASET_PATH / condition

    for person_path in sorted(condition_path.iterdir()):
        image_files = sorted(person_path.glob("*.jpg"))

        print(f"\nPerson: {person_path.name}")
        print(f"Number of images: {len(image_files)}")

        if not image_files:
            continue

        # Inspect first image
        image_path = image_files[0]

        with Image.open(image_path) as image:
            image_array = np.array(image)

            print(f"Sample image: {image_path.name}")
            print(f"Image dimensions: {image.size}")
            print(f"Image mode: {image.mode}")
            print(f"Array shape: {image_array.shape}")
            print(f"Pixel minimum: {image_array.min()}")
            print(f"Pixel maximum: {image_array.max()}")
            print(f"Data type: {image_array.dtype}")

print("\nDataset inspection completed.")