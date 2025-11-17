import os

def remove_mask_suffix(folder):
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        # Skip if it's a directory
        if os.path.isdir(filepath):
            continue

        # Split name and extension
        name, ext = os.path.splitext(filename)

        # Check if name ends with "_mask"
        if name.endswith("_mask"):
            new_name = name[:-5] + ext   # remove "_mask" (5 chars)
            new_path = os.path.join(folder, new_name)

            print(f"Renaming: {filename} → {new_name}")
            os.rename(filepath, new_path)

# -------- RUN --------
folder_path = "/home/wraith/work/G19/dataset/original/normal/masks"   # change this
remove_mask_suffix(folder_path)
