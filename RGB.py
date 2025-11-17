import os
from PIL import Image

# Allowed image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def convert_to_rgb(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTS:
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    rgb = img.convert("RGB")   # convert to 3-channel RGB
                    rgb.save(file_path)        # overwrite original
                    print(f"Converted to RGB: {file_path}")
                except Exception as e:
                    print(f"Error converting {file_path}: {e}")

if __name__ == "__main__":
    target_dir = input("Enter the directory path: ").strip()
    convert_to_rgb(target_dir)
    print("Done!")
