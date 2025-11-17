import cv2
import os
import glob
import shutil

def merge_and_move_masks(mask_folder):
    final_folder = os.path.join(mask_folder, "final")
    os.makedirs(final_folder, exist_ok=True)

    # List all mask files ending with _mask.png or _mask_1.png
    mask_files = glob.glob(os.path.join(mask_folder, "*_mask.png"))
    mask1_files = glob.glob(os.path.join(mask_folder, "*_mask_1.png"))

    # Extract base filename before _mask or _mask_1
    def get_base(filename):
        if filename.endswith("_mask.png"):
            return filename[:-9]  # remove '_mask.png'
        elif filename.endswith("_mask_1.png"):
            return filename[:-11]  # remove '_mask_1.png'
        else:
            return None

    mask_map = {get_base(os.path.basename(f)): f for f in mask_files}
    mask1_map = {get_base(os.path.basename(f)): f for f in mask1_files}

    all_keys = set(mask_map.keys()).union(mask1_map.keys())

    for key in all_keys:
        mask_path = mask_map.get(key)
        mask1_path = mask1_map.get(key)

        if mask_path and mask1_path:
            # Both masks exist, merge
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or mask1 is None:
                print(f"Error reading masks for {key}")
                continue
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            _, mask1_bin = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
            merged = cv2.bitwise_or(mask_bin, mask1_bin)

            save_path = os.path.join(final_folder, f"{key}_mask.png")
            cv2.imwrite(save_path, merged)
            print(f"Merged and saved: {save_path}")

        elif mask_path:
            # Only mask exists, copy it
            shutil.copy(mask_path, os.path.join(final_folder, os.path.basename(mask_path)))
            print(f"Copied unique mask: {mask_path}")

        elif mask1_path:
            # Only mask_1 exists, copy it
            shutil.copy(mask1_path, os.path.join(final_folder, os.path.basename(mask1_path)))
            print(f"Copied unique mask: {mask1_path}")

# Replace 'path_to_masks_folder' with the your actual masks folder path
merge_and_move_masks("/home/wraith/work/G19/dataset/original/malignant/masks")
