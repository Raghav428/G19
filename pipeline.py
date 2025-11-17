import os
import numpy as np
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm


# --- Reproducibility ---
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
seed_everything()


# Paths & Parameters
DATASET_FOLDER = "./dataset/original"
AUGMENTED_FOLDER = "./dataset/augmented"
BATCH_SIZE_SEG = 12
BATCH_SIZE_CLS = 24
SEG_EPOCHS = 25
CLS_EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 256
NUM_AUGMENTATIONS = 3  # Number of augmented versions per image


# --- AUGMENTATION UTILITIES ---
def get_augmentation_pipeline():
    """Returns augmentation pipeline that resizes first, then augmentations"""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),  # Resize first
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    ])


def augment_and_save(image_path, mask_path, output_img_dir, output_mask_dir, prefix):
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    
    augmentation = get_augmentation_pipeline()
    augmented = augmentation(image=image, mask=mask)
    aug_image = augmented['image']
    aug_mask = augmented['mask']
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]
    
    new_img_name = f"{prefix}_{base_name}{ext}"
    new_mask_name = f"{prefix}_{base_name}{ext}"
    
    Image.fromarray(aug_image).save(os.path.join(output_img_dir, new_img_name))
    Image.fromarray(aug_mask).save(os.path.join(output_mask_dir, new_mask_name))
    return os.path.join(output_img_dir, new_img_name), os.path.join(output_mask_dir, new_mask_name)


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
            print(f"Augmented dataset found at {augmented_folder}. Skipping augmentation.")
            return
    
    print(f"Augmented dataset not found. Generating augmentations...")
    
    classes = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))])
    
    for cls_name in classes:
        print(f"Augmenting class: {cls_name}")
        img_folder = os.path.join(dataset_folder, cls_name, 'images')
        mask_folder = os.path.join(dataset_folder, cls_name, 'masks')
        
        output_img_folder = os.path.join(augmented_folder, cls_name, 'images')
        output_mask_folder = os.path.join(augmented_folder, cls_name, 'masks')
        
        os.makedirs(output_img_folder, exist_ok=True)
        os.makedirs(output_mask_folder, exist_ok=True)
        
        image_files = sorted([f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for image_file in tqdm(image_files, desc=f"Processing {cls_name}"):
            image_path = os.path.join(img_folder, image_file)
            mask_path = os.path.join(mask_folder, image_file)
            
            if not os.path.exists(mask_path):
                print(f"Warning: No mask found for {image_path}, skipping.")
                continue
            
            # Resize and save original images/masks to augmented folder 
            original_img = Image.open(image_path)
            original_mask = Image.open(mask_path)
            resized_img = original_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            resized_mask = original_mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            resized_img.save(os.path.join(output_img_folder, image_file))
            resized_mask.save(os.path.join(output_mask_folder, image_file))
            
            # Generate augmentations
            for aug_idx in range(num_augmentations):
                prefix = f"aug{aug_idx+1}"
                augment_and_save(image_path, mask_path, output_img_folder, output_mask_folder, prefix)
    
    print(f"Augmentation complete. Augmented dataset saved to {augmented_folder}")


# --- DATA LOADING HELPERS ---
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
            print(f"Warning: {img_folder} does not exist, skipping.")
            continue
            
        image_files = sorted([f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

        for image_file in image_files:
            image_path = os.path.join(img_folder, image_file)
            mask_path = os.path.join(mask_folder, image_file)

            if not os.path.exists(mask_path):
                print(f"Warning: No mask found for image {image_path}, skipping.")
                continue

            mask_img = Image.open(mask_path).convert("L")
            mask_np = np.array(mask_img) / 255.0
            mask_np = mask_np.astype(np.float32)

            data.append((image_path, mask_np, label))
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

        return image, mask.unsqueeze(0).float()


class ClassificationDataset(Dataset):
    def __init__(self, data_tuples, pred_masks, transform=None):
        """
        data_tuples: list of (image_path, original_mask_np, label)
        pred_masks: predicted masks from selected segmentation model
        """
        self.data = data_tuples
        self.pred_masks = pred_masks
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, orig_mask_np, label = self.data[idx]
        pred_mask_np = self.pred_masks[idx].astype(np.float32)

        image = np.array(Image.open(image_path).convert("RGB"))
        # Concatenate: RGB (3) + original mask (1) + predicted mask (1) = 5 channels
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


def get_classification_model(name, num_classes=3):
    if name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(5, 64, 7, 2, 3, bias=False)  # 5 channels: RGB + 2 masks
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.conv1 = nn.Conv2d(5, 64, 7, 2, 3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.features[0][0] = nn.Conv2d(5, 32, 3, 2, 1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'efficientnet_b4':
        model = models.efficientnet_b4(pretrained=True)
        model.features[0][0] = nn.Conv2d(5, 48, 3, 2, 1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        model.features[0][0] = nn.Conv2d(5, 32, 3, 2, 1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'inception_v3':
        model = models.inception_v3(pretrained=True, aux_logits=False)
        model.Conv2d_1a_3x3.conv = nn.Conv2d(5, 32, 3, 2, 0, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model {name} not supported")
    return model


# --- TRAINING UTILITIES ---
def train_segmentation(model, train_loader, val_loader, epochs, device, model_name="segmentation"):
    from segmentation_models_pytorch.losses import DiceLoss

    criterion = DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    best_val_loss = float('inf')
    best_model_path = f"best_{model_name}.pth"

    for epoch in range(epochs):
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best {model_name} model with val loss: {val_loss:.4f}")

        print(f"[{model_name}] Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

    model.load_state_dict(torch.load(best_model_path))
    return model, best_val_loss


def generate_predicted_masks(model, dataset, device):
    model.eval()
    preds = []
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = (masks > 0.5).astype(np.float32)
            preds.extend(masks.squeeze(1))
    return preds


def train_classification(model, train_loader, val_loader, epochs, device, model_name="classification"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    best_acc = 0
    best_model_path = f"best_{model_name}.pth"

    for epoch in range(epochs):
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
        print(f"[{model_name}] Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Acc: {acc:.4f} Val F1: {f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best {model_name} model with val acc: {acc:.4f}")

    model.load_state_dict(torch.load(best_model_path))
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


# --- MAIN PIPELINE ---
def main():
    print("=" * 80)
    print("STEP 1: Checking for augmented dataset...")
    print("=" * 80)
    check_and_generate_augmentations(DATASET_FOLDER, AUGMENTED_FOLDER, NUM_AUGMENTATIONS)
    
    print("\n" + "=" * 80)
    print("STEP 2: Loading augmented dataset...")
    print("=" * 80)
    classes, class_to_idx = get_classes_and_labels(AUGMENTED_FOLDER)
    data = find_image_mask_label_tuples(AUGMENTED_FOLDER, class_to_idx)

    train_data, val_data = train_test_split(data, test_size=0.2, stratify=[d[2] for d in data], random_state=42)

    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

    train_seg_ds = SegmentationDataset(train_data, transform=segmentation_transform)
    val_seg_ds = SegmentationDataset(val_data, transform=segmentation_transform)
    train_seg_loader = DataLoader(train_seg_ds, batch_size=BATCH_SIZE_SEG, shuffle=True, num_workers=4)
    val_seg_loader = DataLoader(val_seg_ds, batch_size=BATCH_SIZE_SEG, shuffle=False, num_workers=4)

    print("\n" + "=" * 80)
    print("STEP 3: Training and comparing segmentation models...")
    print("=" * 80)
    
    seg_models_to_test = {
        'inceptionv4': get_segmentation_model('inceptionv4'),
        'inceptionresnetv2': get_segmentation_model('inceptionresnetv2')
    }
    
    seg_results = {}
    trained_seg_models = {}
    
    for model_name, model in seg_models_to_test.items():
        print(f"\nTraining segmentation model: {model_name}")
        trained_model, val_loss = train_segmentation(model, train_seg_loader, val_seg_loader, SEG_EPOCHS, DEVICE, model_name)
        seg_results[model_name] = val_loss
        trained_seg_models[model_name] = trained_model
        print(f"{model_name} final validation loss: {val_loss:.4f}")
    
    best_seg_model_name = min(seg_results, key=seg_results.get)
    best_seg_model = trained_seg_models[best_seg_model_name]
    print(f"\n{'='*80}")
    print(f"Best segmentation model: {best_seg_model_name} with validation loss: {seg_results[best_seg_model_name]:.4f}")
    print(f"{'='*80}")

    print("\n" + "=" * 80)
    print("STEP 4: Generating predicted masks using best segmentation model...")
    print("=" * 80)
    
    norm_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_seg_ds_noaug = SegmentationDataset(train_data, transform=norm_transform)
    val_seg_ds_noaug = SegmentationDataset(val_data, transform=norm_transform)

    train_masks_pred = generate_predicted_masks(best_seg_model, train_seg_ds_noaug, DEVICE)
    val_masks_pred = generate_predicted_masks(best_seg_model, val_seg_ds_noaug, DEVICE)

    print("\n" + "=" * 80)
    print("STEP 5: Training classification models...")
    print("=" * 80)
    
    train_cls_ds = ClassificationDataset(train_data, train_masks_pred, transform=classification_transform)
    val_cls_ds = ClassificationDataset(val_data, val_masks_pred, transform=classification_transform)
    train_cls_loader = DataLoader(train_cls_ds, batch_size=BATCH_SIZE_CLS, shuffle=True, num_workers=4)
    val_cls_loader = DataLoader(val_cls_ds, batch_size=BATCH_SIZE_CLS, shuffle=False, num_workers=4)

    model_names = ['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b4', 'mobilenet_v2', 'inception_v3']
    report = []

    for name in model_names:
        print(f"\n=== Training classifier {name} ===")
        cls_model = get_classification_model(name, num_classes=len(classes))
        best_acc = train_classification(cls_model, train_cls_loader, val_cls_loader, CLS_EPOCHS, DEVICE, name)
        report.append({'Model': name, 'Best_Val_Accuracy': best_acc})

    print("\n" + "=" * 80)
    print("STEP 6: Final Results")
    print("=" * 80)
    
    print(f"\nBest Segmentation Model: {best_seg_model_name}")
    print(f"Validation Loss: {seg_results[best_seg_model_name]:.4f}\n")
    
    df = pd.DataFrame(report)
    print("Classification Performance Report:")
    print(df.to_string(index=False))
    df.to_csv("classification_performance_report.csv", index=False)
    print("\nReport saved to: classification_performance_report.csv")


if __name__ == "__main__":
    main()
