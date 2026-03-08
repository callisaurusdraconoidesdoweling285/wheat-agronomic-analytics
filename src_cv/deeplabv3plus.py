import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import time

# =====================
# 1. CONFIGURATION
# =====================
NUM_CLASSES = 4
BATCH_SIZE = 4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
IMAGE_SIZE = (512, 512)

CLASS_INFO = {
    0: {"name": "background", "color": [0, 0, 0]},
    1: {"name": "head", "color": [132, 255, 50]},
    2: {"name": "stem", "color": [255, 132, 50]},
    3: {"name": "leaf", "color": [50, 255, 214]}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 2. DATASET
# =====================
class WheatSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths, self.mask_paths = [], []
        self.transform = transform
        self.resize = transforms.Resize(IMAGE_SIZE)

        for folder in os.listdir(image_dir):
            img_folder = os.path.join(image_dir, folder)
            msk_folder = os.path.join(mask_dir, folder)
            if not os.path.isdir(img_folder):
                continue

            for fname in os.listdir(img_folder):
                if fname.endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(img_folder, fname))
                    self.mask_paths.append(os.path.join(msk_folder, fname))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image = self.resize(image)
        mask = self.resize(mask)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, info in CLASS_INFO.items():
        rgb[mask == class_id] = info["color"]
    return rgb

# ---- Metrics helpers ----
def compute_global_pixel_accuracy(preds, labels):
    """Global pixel accuracy across all classes."""
    return (preds == labels).sum().item(), labels.numel()  # return counts

def update_per_class_counts(preds, labels, class_correct, class_total, num_classes=NUM_CLASSES):
    """Accumulate per-class correct/total counts."""
    for cls in range(num_classes):
        cls_mask = (labels == cls)
        total = cls_mask.sum().item()
        if total == 0:
            continue
        correct = ((preds == labels) & cls_mask).sum().item()
        class_correct[cls] += correct
        class_total[cls] += total

def compute_balanced_accuracy_from_counts(class_correct, class_total):
    """Mean per-class accuracy from accumulated counts."""
    accs = []
    for c, t in zip(class_correct, class_total):
        if t > 0:
            accs.append(c / t)
    return float(np.mean(accs)) if accs else 0.0

def compute_mIoU(preds, labels, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

# =====================
# 3. EARLY STOPPING CLASS
# =====================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_weights = model.state_dict()
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_weights = model.state_dict()
            self.counter = 0

# =====================
# 4. PREPARE DATA
# =====================
image_root = "./images"
mask_root = "./masks_grayscale"

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = WheatSegmentationDataset(image_root, mask_root, transform=transform)

train_len = int(0.7 * len(dataset))
val_len = int(0.15 * len(dataset))
test_len = len(dataset) - train_len - val_len

train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

# =====================
# 5. MODEL & EARLY STOPPING
# =====================
# Note: Depending on torchvision version, the weights arg may differ.
model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=NUM_CLASSES)
# Ensure classifier head outputs NUM_CLASSES
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)

# =====================
# 6. TRAINING LOOP
# =====================
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # ---- Validation (with balanced accuracy) ----
    model.eval()
    val_loss = 0.0
    val_iou = 0.0

    # Accumulators for accurate set-level metrics
    val_total_correct = 0
    val_total_pixels = 0
    val_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    val_class_total = np.zeros(NUM_CLASSES, dtype=np.int64)

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1)

            # Loss / IoU
            val_loss += loss.item()
            val_iou += compute_mIoU(preds, masks, NUM_CLASSES)

            # Global pixel accuracy counts
            correct, total = compute_global_pixel_accuracy(preds, masks)
            val_total_correct += correct
            val_total_pixels += total

            # Per-class counts
            update_per_class_counts(preds, masks, val_class_correct, val_class_total, NUM_CLASSES)

    n = len(val_loader)
    avg_val_loss = val_loss / max(1, n)
    avg_val_iou = val_iou / max(1, n)
    val_global_acc = val_total_correct / max(1, val_total_pixels)
    val_balanced_acc = compute_balanced_accuracy_from_counts(val_class_correct, val_class_total)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Train Loss: {running_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Global Acc: {val_global_acc:.4f} | "
        f"Val Balanced Acc: {val_balanced_acc:.4f} | "
        f"Val mIoU: {avg_val_iou:.4f}"
    )

    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

# Load the best model weights before testing
if early_stopping.best_model_weights:
    model.load_state_dict(early_stopping.best_model_weights)
    print("Loaded best model weights.")

# =====================
# 7. TEST EVALUATION
# =====================
model.eval()
test_loss = 0.0
test_iou = 0.0

test_total_correct = 0
test_total_pixels = 0
test_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
test_class_total = np.zeros(NUM_CLASSES, dtype=np.int64)

with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        preds = torch.argmax(outputs, dim=1)
        test_loss += loss.item()
        test_iou += compute_mIoU(preds, masks, NUM_CLASSES)

        correct, total = compute_global_pixel_accuracy(preds, masks)
        test_total_correct += correct
        test_total_pixels += total

        update_per_class_counts(preds, masks, test_class_correct, test_class_total, NUM_CLASSES)

n = len(test_loader)
avg_test_loss = test_loss / max(1, n)
avg_test_iou = test_iou / max(1, n)
test_global_acc = test_total_correct / max(1, test_total_pixels)
test_balanced_acc = compute_balanced_accuracy_from_counts(test_class_correct, test_class_total)

print(
    f"Test Results -> Loss: {avg_test_loss:.4f} | "
    f"Global Acc: {test_global_acc:.4f} | "
    f"Balanced Acc: {test_balanced_acc:.4f} | "
    f"mIoU: {avg_test_iou:.4f}"
)

# (Optional) print per-class accuracies for inspection
for cls in range(NUM_CLASSES):
    if test_class_total[cls] > 0:
        print(f"  Class {cls} ({CLASS_INFO[cls]['name']}): "
              f"Acc = {test_class_correct[cls]/test_class_total[cls]:.4f} "
              f"({test_class_correct[cls]}/{test_class_total[cls]})")
    else:
        print(f"  Class {cls} ({CLASS_INFO[cls]['name']}): no pixels in ground truth.")

# =====================
# 8. VISUALIZATION
# =====================
output_dir = "deeplab1_images"
os.makedirs(output_dir, exist_ok=True)  # Create output directory

sample_images, sample_masks = next(iter(test_loader))
sample_images = sample_images.to(device)
with torch.no_grad():
    sample_preds = torch.argmax(model(sample_images)['out'], dim=1).cpu()

for i in range(min(3, sample_images.size(0))):
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_images[i].cpu().permute(1, 2, 0))
    plt.title("Input Image")
    plt.subplot(1, 3, 2)
    plt.imshow(mask_to_rgb(sample_masks[i].numpy()))
    plt.title("Ground Truth")
    plt.subplot(1, 3, 3)
    plt.imshow(mask_to_rgb(sample_preds[i].numpy()))
    plt.title("Prediction")
    plt.tight_layout()

    image_filename = os.path.join(output_dir, f"segmentation_example_{i}.png")
    plt.savefig(image_filename)
    print(f"Saved visualization to {image_filename}")
    plt.close()

# =====================
# 9. TOTAL RUNTIME
# =====================
end_time = time.time()  # <-- End timer
print(f"Total script execution time: {end_time - start_time:.2f} seconds")

