import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import argparse
from datetime import datetime
import pickle
import csv

# --- Constants ---
DICE_SMOOTH = 1e-6

# --- Reproducibility ---
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
seed_everything()

# Paths & Params
DATASET_FOLDER = "./dataset/original"
AUGMENTED_FOLDER = "./dataset/augmented"
PRED_MASKS_FOLDER = "./predicted_masks"
MODEL_CHECKPOINTS = "./model_checkpoints"
REPORT_FILE = "classification_performance_report.csv"
SEG_EVAL_FILE = os.path.join(MODEL_CHECKPOINTS, "segmentation_evaluation.csv")
DATA_STATE_PATH = "data"
os.makedirs(MODEL_CHECKPOINTS, exist_ok=True)
os.makedirs(PRED_MASKS_FOLDER, exist_ok=True)
os.makedirs(DATA_STATE_PATH, exist_ok=True)

BATCH_SIZE_SEG = 12
BATCH_SIZE_CLS = 24
SEG_EPOCHS = 25
CLS_EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 256
NUM_AUGMENTATIONS = 3
NUM_WORKERS = min(4, os.cpu_count() or 1)  # Auto-detect, max 4

def print_with_timestamp(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# --- AUGMENTATION UTILITIES ---
def get_augmentation_pipeline():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    ])

def augment_and_save(image_path, mask_path, output_img_dir, output_mask_dir, prefix):
    try:
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # Validate dimensions
        if image.shape[:2] != mask.shape[:2]:
            print_with_timestamp(f"Warning: Image and mask dimensions don't match for {image_path}")
            return None, None
        
        augmentation = get_augmentation_pipeline()
        augmented = augmentation(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # Force PNG for lossless storage
        new_img_name = f"{prefix}_{base_name}.png"
        new_mask_name = f"{prefix}_{base_name}.png"
        Image.fromarray(aug_image).save(os.path.join(output_img_dir, new_img_name))
        Image.fromarray(aug_mask).save(os.path.join(output_mask_dir, new_mask_name))
        return os.path.join(output_img_dir, new_img_name), os.path.join(output_mask_dir, new_mask_name)
    except Exception as e:
        print_with_timestamp(f"Error augmenting {image_path}: {str(e)}")
        return None, None

def check_and_generate_augmentations(dataset_folder, augmented_folder, num_augmentations):
    if os.path.exists(augmented_folder):
        classes = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
        all_exist = True
        for cls_name in classes:
            aug_img_folder = os.path.join(augmented_folder, cls_name, 'images')
            if not os.path.exists(aug_img_folder) or len(os.listdir(aug_img_folder)) == 0:
                all_exist = False
                break
        if all_exist:
            print_with_timestamp(f"Augmented dataset found at {augmented_folder}. Skipping augmentation.")
            return
    print_with_timestamp("Augmented dataset not found. Generating augmentations...")
    classes = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))])
    for cls_name in classes:
        print_with_timestamp(f"Augmenting class: {cls_name}")
        img_folder = os.path.join(dataset_folder, cls_name, 'images')
        mask_folder = os.path.join(dataset_folder, cls_name, 'masks')
        output_img_folder = os.path.join(augmented_folder, cls_name, 'images')
        output_mask_folder = os.path.join(augmented_folder, cls_name, 'masks')
        os.makedirs(output_img_folder, exist_ok=True)
        os.makedirs(output_mask_folder, exist_ok=True)
        image_files = sorted([f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        for image_file in tqdm(image_files, desc=f"Augmenting {cls_name}", leave=False):
            image_path = os.path.join(img_folder, image_file)
            mask_path = os.path.join(mask_folder, image_file)
            if not os.path.exists(mask_path):
                print_with_timestamp(f"Warning: No mask found for {image_path}, skipping.")
                continue
            original_img = Image.open(image_path)
            original_mask = Image.open(mask_path)
            resized_img = original_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            resized_mask = original_mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            resized_img.save(os.path.join(output_img_folder, image_file))
            resized_mask.save(os.path.join(output_mask_folder, image_file))
            for aug_idx in range(num_augmentations):
                prefix = f"aug{aug_idx+1}"
                augment_and_save(image_path, mask_path, output_img_folder, output_mask_folder, prefix)
    print_with_timestamp(f"Augmentation complete. Augmented dataset saved to {augmented_folder}")

# --- DATA LOADING ---
def get_classes_and_labels(dataset_path):
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def find_image_mask_label_tuples(dataset_path, class_to_idx):
    data = []
    for cls_name, label in class_to_idx.items():
        img_folder = os.path.join(dataset_path, cls_name, 'images')
        mask_folder = os.path.join(dataset_path, cls_name, 'masks')
        if not os.path.exists(img_folder):
            print_with_timestamp(f"Warning: {img_folder} does not exist, skipping.")
            continue
        
        if not os.path.exists(mask_folder):
            print_with_timestamp(f"Warning: {mask_folder} does not exist, skipping class {cls_name}.")
            continue
            
        image_files = sorted([f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(image_files) == 0:
            print_with_timestamp(f"Warning: No images found in {img_folder}, skipping class {cls_name}.")
            continue
            
        for image_file in image_files:
            image_path = os.path.join(img_folder, image_file)
            mask_path = os.path.join(mask_folder, image_file)
            
            if not os.path.exists(mask_path):
                # Try with .png extension
                base_name = os.path.splitext(image_file)[0]
                mask_path = os.path.join(mask_folder, base_name + '.png')
                if not os.path.exists(mask_path):
                    print_with_timestamp(f"Warning: No mask found for image {image_path}, skipping.")
                    continue
            
            try:
                mask_img = Image.open(mask_path).convert("L")
                image_img = Image.open(image_path).convert("RGB")
                
                # Validate dimensions match
                if mask_img.size != image_img.size:
                    print_with_timestamp(f"Warning: Image and mask dimensions don't match for {image_path}, skipping.")
                    continue
                
                mask_np = np.array(mask_img) / 255.0
                mask_np = mask_np.astype(np.float32)
                data.append((image_path, mask_np, label))
            except Exception as e:
                print_with_timestamp(f"Error loading {image_path} or {mask_path}: {str(e)}, skipping.")
                continue
    
    if len(data) == 0:
        raise ValueError(f"No valid image-mask pairs found in {dataset_path}")
    
    return data

# --- DATASET CLASSES ---
class SegmentationDataset(Dataset):
    def __init__(self, data_tuples, transform=None, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
        self.data = data_tuples
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, mask_np, _ = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.fromarray((mask_np * 255).astype(np.uint8))
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)
        image = np.array(image)
        mask = np.array(mask) / 255.0
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, torch.tensor(mask).unsqueeze(0).float()

class ClassificationDataset(Dataset):
    def __init__(self, data_tuples, pred_mask_folder, transform=None):
        self.data = data_tuples
        self.pred_mask_folder = pred_mask_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, orig_mask_np, label = self.data[idx]
        
        # Load predicted mask from disk instead of memory
        mask_filename = f"train_mask_{idx}.png" if "train" in self.pred_mask_folder else f"val_mask_{idx}.png"
        pred_mask_path = os.path.join(self.pred_mask_folder, mask_filename)
        pred_mask_np = np.array(Image.open(pred_mask_path).convert("L")) / 255.0
        
        image = np.array(Image.open(image_path).convert("RGB"))
        combined = np.concatenate([
            image,
            orig_mask_np[..., None],
            pred_mask_np[..., None]],
            axis=2).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=combined)
            combined = augmented['image']
        return combined, label

# --- TRANSFORMS ---
segmentation_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

classification_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.Normalize(
        mean=(0.485, 0.456, 0.406, 0, 0),
        std=(0.229, 0.224, 0.225, 1, 1)
    ),
    ToTensorV2()
])

# --- MODEL FACTORIES ---
def get_segmentation_model(encoder_name="inceptionresnetv2"):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )

class InceptionV3Wrapper(nn.Module):
    """Wrapper for Inception V3 that handles 299x299 input requirement"""
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.requires_resize = True
        
    def forward(self, x):
        if self.requires_resize and x.size(2) != 299:
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return self.model(x)

def get_classification_model(name, num_classes=3):
    if name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.conv1 = nn.Conv2d(5, 64, 7, 2, 3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.conv1 = nn.Conv2d(5, 64, 7, 2, 3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.features[0][0] = nn.Conv2d(5, 32, 3, 2, 1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        model.features[0][0] = nn.Conv2d(5, 48, 3, 2, 1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.features[0][0] = nn.Conv2d(5, 32, 3, 2, 1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'inception_v3':
        base_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=False)
        base_model.Conv2d_1a_3x3.conv = nn.Conv2d(5, 32, 3, 2, 0, bias=False)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        # Wrap model to handle 299x299 requirement cleanly
        model = InceptionV3Wrapper(base_model)
    else:
        raise ValueError(f"Model {name} not supported")
    return model

# --- TRAINING UTILITIES ---
def train_segmentation(model, train_loader, val_loader, epochs, device, model_name):
    from segmentation_models_pytorch.losses import DiceLoss
    criterion = DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)
    best_val_loss = float('inf')
    best_model_path = os.path.join(MODEL_CHECKPOINTS, f"best_{model_name}.pth")

    for epoch in tqdm(range(epochs), desc=f"Training {model_name}"):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        tqdm.write(f"[{model_name}] Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            tqdm.write(f"Saved best {model_name} model with val loss: {val_loss:.4f}")

    try:
        model.load_state_dict(torch.load(best_model_path))
    except Exception as e:
        print_with_timestamp(f"Error loading best model from {best_model_path}: {str(e)}")
        raise
    return model, best_val_loss

def generate_predicted_masks_and_save(model, dataset, device, save_folder, split_name):
    """Generate masks in batches and save to disk without storing all in memory"""
    model.eval()
    os.makedirs(save_folder, exist_ok=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
    
    mask_index = 0
    for imgs, _ in tqdm(loader, desc=f"Generating masks ({split_name})"):
        imgs = imgs.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = (masks > 0.5).astype(np.uint8)  # Changed to uint8 to save memory
            
            # Save immediately instead of storing in list
            for mask_np in masks.squeeze(1):
                mask_img = Image.fromarray(mask_np * 255)
                mask_img.save(os.path.join(save_folder, f"{split_name}_mask_{mask_index}.png"))
                mask_index += 1
    
    print_with_timestamp(f"Saved {mask_index} masks to {save_folder}")

def train_classification(model, train_loader, val_loader, epochs, device, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)
    best_acc = 0
    best_model_path = os.path.join(MODEL_CHECKPOINTS, f"best_{model_name}.pth")

    for epoch in tqdm(range(epochs), desc=f"Training {model_name}"):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        acc, f1 = evaluate_classification(model, val_loader, device)
        tqdm.write(f"[{model_name}] Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Acc: {acc:.4f} Val F1: {f1:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            tqdm.write(f"Saved best {model_name} model with val acc: {acc:.4f}")

    try:
        model.load_state_dict(torch.load(best_model_path))
    except Exception as e:
        print_with_timestamp(f"Error loading best model from {best_model_path}: {str(e)}")
        raise
    return best_acc

def evaluate_classification(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1

# --- SEGMENTATION EVALUATION ---
def dice_coefficient(y_true, y_pred, smooth=DICE_SMOOTH):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum()
    return (2. * intersection + smooth) / (union + smooth)

def evaluate_segmentation():
    print_with_timestamp("STEP 7: Evaluating segmentation model on validation set...")

    with open(os.path.join(DATA_STATE_PATH, "train_val_data.pkl"), "rb") as f:
        data = pickle.load(f)
    data_tuples = data['data']
    train_data, val_data = train_test_split(data_tuples, test_size=0.2,
                                           stratify=[d[2] for d in data_tuples], random_state=42)

    # Load the best model name dynamically
    best_model_name_file = os.path.join(MODEL_CHECKPOINTS, "best_seg_model_name.txt")
    if os.path.exists(best_model_name_file):
        with open(best_model_name_file, "r") as f:
            best_seg_model_name = f.read().strip()
    else:
        print_with_timestamp("Warning: best_seg_model_name.txt not found, defaulting to inceptionresnetv2")
        best_seg_model_name = 'inceptionresnetv2'
    
    print_with_timestamp(f"Evaluating segmentation model: {best_seg_model_name}")
    best_seg_model = get_segmentation_model(best_seg_model_name)
    model_path = os.path.join(MODEL_CHECKPOINTS, "best_segmentation_model.pth")
    try:
        best_seg_model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print_with_timestamp(f"Error: Model file not found at {model_path}")
        raise
    except Exception as e:
        print_with_timestamp(f"Error loading segmentation model: {str(e)}")
        raise
    best_seg_model.to(DEVICE)
    best_seg_model.eval()

    norm_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_seg_ds_noaug = SegmentationDataset(val_data, transform=norm_transform)
    val_loader = DataLoader(val_seg_ds_noaug, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Evaluating segmentation"):
            imgs = imgs.to(DEVICE)
            outputs = best_seg_model(imgs)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            gt_masks = masks.numpy() > 0.5
            for pred_mask, gt_mask in zip(preds, gt_masks):
                pred_flat = pred_mask.flatten()
                gt_flat = gt_mask.flatten()
                dice_scores.append(dice_coefficient(gt_flat, pred_flat))
                iou_scores.append(jaccard_score(gt_flat, pred_flat))

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    print_with_timestamp(f"Mean Dice Coefficient: {avg_dice:.4f}")
    print_with_timestamp(f"Mean IoU: {avg_iou:.4f}")

    with open(SEG_EVAL_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Mean Dice Coefficient", avg_dice])
        writer.writerow(["Mean IoU", avg_iou])
    print_with_timestamp(f"Segmentation evaluation report saved to: {SEG_EVAL_FILE}")

# --- PIPELINE STEPS ---
def step_augment_data():
    print_with_timestamp("STEP 1: Checking for augmented dataset...")
    check_and_generate_augmentations(DATASET_FOLDER, AUGMENTED_FOLDER, NUM_AUGMENTATIONS)

def step_load_data():
    print_with_timestamp("STEP 2: Loading augmented dataset...")
    classes, class_to_idx = get_classes_and_labels(AUGMENTED_FOLDER)
    data = find_image_mask_label_tuples(AUGMENTED_FOLDER, class_to_idx)
    print_with_timestamp(f"Loaded {len(data)} samples across {len(classes)} classes.")
    with open(os.path.join(DATA_STATE_PATH, "train_val_data.pkl"), "wb") as f:
        pickle.dump({'classes': classes, 'class_to_idx': class_to_idx, 'data': data}, f)

def step_train_segmentation():
    with open(os.path.join(DATA_STATE_PATH, "train_val_data.pkl"), "rb") as f:
        data = pickle.load(f)
    classes = data['classes']
    data_tuples = data['data']

    train_data, val_data = train_test_split(data_tuples, test_size=0.2,
                                           stratify=[d[2] for d in data_tuples], random_state=42)
    print_with_timestamp(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

    train_seg_ds = SegmentationDataset(train_data, transform=segmentation_transform)
    val_seg_ds = SegmentationDataset(val_data, transform=segmentation_transform)

    train_seg_loader = DataLoader(train_seg_ds, batch_size=BATCH_SIZE_SEG, shuffle=True, num_workers=NUM_WORKERS)
    val_seg_loader = DataLoader(val_seg_ds, batch_size=BATCH_SIZE_SEG, shuffle=False, num_workers=NUM_WORKERS)

    seg_models_to_test = {
        'inceptionv4': get_segmentation_model('inceptionv4'),
        'inceptionresnetv2': get_segmentation_model('inceptionresnetv2')
    }
    seg_results = {}
    trained_seg_models = {}

    for model_name, model in seg_models_to_test.items():
        print_with_timestamp(f"Training segmentation model: {model_name}")
        trained_model, val_loss = train_segmentation(model, train_seg_loader, val_seg_loader,
                                                     SEG_EPOCHS, DEVICE, model_name)
        seg_results[model_name] = val_loss
        trained_seg_models[model_name] = trained_model
        print_with_timestamp(f"{model_name} final validation loss: {val_loss:.4f}")

    best_seg_model_name = min(seg_results, key=seg_results.get)
    best_seg_model = trained_seg_models[best_seg_model_name]
    print_with_timestamp(f"Best segmentation model: {best_seg_model_name} with val loss: {seg_results[best_seg_model_name]:.4f}")
    torch.save(best_seg_model.state_dict(), os.path.join(MODEL_CHECKPOINTS, "best_segmentation_model.pth"))
    
    # Save the best model name for later use
    with open(os.path.join(MODEL_CHECKPOINTS, "best_seg_model_name.txt"), "w") as f:
        f.write(best_seg_model_name)
    
    with open(os.path.join(DATA_STATE_PATH, "classes.pkl"), "wb") as f:
        pickle.dump(classes, f)

def step_generate_masks():
    # NOTE: Potential data leakage - the segmentation model was trained on train_data
    # and now generates masks for the same training data. Consider using a separate
    # validation fold for segmentation if this impacts classification performance.
    with open(os.path.join(DATA_STATE_PATH, "train_val_data.pkl"), "rb") as f:
        data = pickle.load(f)
    data_tuples = data['data']
    train_data, val_data = train_test_split(data_tuples, test_size=0.2,
                                           stratify=[d[2] for d in data_tuples], random_state=42)
    
    # Load the best model name dynamically
    best_model_name_file = os.path.join(MODEL_CHECKPOINTS, "best_seg_model_name.txt")
    if os.path.exists(best_model_name_file):
        with open(best_model_name_file, "r") as f:
            best_seg_model_name = f.read().strip()
    else:
        print_with_timestamp("Warning: best_seg_model_name.txt not found, defaulting to inceptionresnetv2")
        best_seg_model_name = 'inceptionresnetv2'
    
    print_with_timestamp(f"Loading best segmentation model: {best_seg_model_name}")
    best_seg_model = get_segmentation_model(best_seg_model_name)
    model_path = os.path.join(MODEL_CHECKPOINTS, "best_segmentation_model.pth")
    try:
        best_seg_model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print_with_timestamp(f"Error: Model file not found at {model_path}. Run step 3 first.")
        raise
    except Exception as e:
        print_with_timestamp(f"Error loading segmentation model: {str(e)}")
        raise
    best_seg_model.to(DEVICE)

    norm_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_seg_ds_noaug = SegmentationDataset(train_data, transform=norm_transform)
    val_seg_ds_noaug = SegmentationDataset(val_data, transform=norm_transform)

    # Just save masks, don't return them
    generate_predicted_masks_and_save(
        best_seg_model, train_seg_ds_noaug, DEVICE, os.path.join(PRED_MASKS_FOLDER, "train"), "train")
    generate_predicted_masks_and_save(
        best_seg_model, val_seg_ds_noaug, DEVICE, os.path.join(PRED_MASKS_FOLDER, "val"), "val")
    
    # No need to pickle masks anymore, we load them on-the-fly

def step_train_classification():
    with open(os.path.join(DATA_STATE_PATH, "train_val_data.pkl"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(DATA_STATE_PATH, "classes.pkl"), "rb") as f:
        classes = pickle.load(f)

    train_data, val_data = train_test_split(data['data'], test_size=0.2,
                                           stratify=[d[2] for d in data['data']], random_state=42)

    # Pass folder paths instead of loaded masks
    train_cls_ds = ClassificationDataset(train_data, os.path.join(PRED_MASKS_FOLDER, "train"), transform=classification_transform)
    val_cls_ds = ClassificationDataset(val_data, os.path.join(PRED_MASKS_FOLDER, "val"), transform=classification_transform)

    train_cls_loader = DataLoader(train_cls_ds, batch_size=BATCH_SIZE_CLS, shuffle=True, num_workers=NUM_WORKERS)
    val_cls_loader = DataLoader(val_cls_ds, batch_size=BATCH_SIZE_CLS, shuffle=False, num_workers=NUM_WORKERS)

    model_names = ['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b4', 'mobilenet_v2', 'inception_v3']
    report = []

    for name in model_names:
        print_with_timestamp(f"Training classifier {name}")
        cls_model = get_classification_model(name, num_classes=len(classes))
        best_acc = train_classification(cls_model, train_cls_loader, val_cls_loader, CLS_EPOCHS, DEVICE, name)
        report.append({'Model': name, 'Best_Val_Accuracy': best_acc})

    df = pd.DataFrame(report)
    df.to_csv(REPORT_FILE, index=False)
    with open(os.path.join(DATA_STATE_PATH, "classification_report.pkl"), "wb") as f:
        pickle.dump(df, f)

def step_report():
    print_with_timestamp("Final Classification Results:")
    with open(os.path.join(DATA_STATE_PATH, "classes.pkl"), "rb") as f:
        classes = pickle.load(f)
    with open(os.path.join(DATA_STATE_PATH, "classification_report.pkl"), "rb") as f:
        df = pickle.load(f)
    print(f"Classes: {classes}")
    print("Classification Performance Report:")
    print(df.to_string(index=False))
    print(f"Report saved to: {REPORT_FILE}")

def step_all():
    print_with_timestamp("Running full pipeline")
    step_augment_data()
    step_load_data()
    step_train_segmentation()
    step_generate_masks()
    step_train_classification()
    step_report()
    evaluate_segmentation()

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline steps modularly")
    parser.add_argument('--step', type=str, required=True,
                        help="Step to run: all|1|2|3|4|5|6|7\n"
                             "1: Augment Data\n"
                             "2: Load Data\n"
                             "3: Train Segmentation\n"
                             "4: Generate Masks\n"
                             "5: Train Classification\n"
                             "6: Report\n"
                             "7: Evaluate Segmentation\n"
                             "all: Run entire pipeline")
    args = parser.parse_args()

    if args.step.lower() == 'all' or args.step == '0':
        step_all()
    elif args.step == '1':
        step_augment_data()
    elif args.step == '2':
        step_load_data()
    elif args.step == '3':
        step_train_segmentation()
    elif args.step == '4':
        step_generate_masks()
    elif args.step == '5':
        step_train_classification()
    elif args.step == '6':
        step_report()
    elif args.step == '7':
        evaluate_segmentation()
    else:
        print_with_timestamp("Invalid step.")
