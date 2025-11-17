import os
from PIL import Image

def resize_images(root_dir, size=(256, 256)):
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

    for path, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(supported_ext):
                img_path = os.path.join(path, file)
                try:
                    img = Image.open(img_path)
                    img = img.resize(size, Image.LANCZOS)  # High-quality resize
                    img.save(img_path)  # overwrite original
                    print(f"Resized: {img_path}")
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")

if __name__ == "__main__":
    folder = input("Enter the directory to scan: ")
    resize_images(folder)
