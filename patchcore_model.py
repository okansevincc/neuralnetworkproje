import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Eğitim ve test görüntülerini okuma ve işleme
# Görüntüleri normalize eder, kenarları vurgulayan bir maske ile birleştirir

def load_and_preprocess_images(root_dir):
    image_paths, processed_images = [], []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img = cv2.resize(img, (256, 256))
                img = img.astype(np.float32) / 255.0

                gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                _, dark_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                dark_mask = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

                gauss = cv2.getGaussianKernel(256, 75)
                gauss_map = gauss @ gauss.T
                gauss_map = (gauss_map - gauss_map.min()) / (gauss_map.max() - gauss_map.min())
                gauss_map = np.stack([gauss_map] * 3, axis=-1)

                dark_mask = dark_mask * (gauss_map ** 0.1)

                combined = (img * 0.20 + dark_mask * 0.80).astype(np.float32)
                processed_images.append(combined)
                image_paths.append(img_path)

    return processed_images, image_paths

# Dataset sınıfı
class PatchCoreDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = torch.tensor(img).permute(2, 0, 1)  # HWC -> CHW
        return img

# ResNet34 modelinden sadece ilk layer'ı kullanan backbone
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )

    def forward(self, x):
        return self.feature_extractor(x)

# Özellik çıkarımı
def extract_features(dataloader, model, device):
    model.eval()
    all_feats = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            features = model(batch)
            all_feats.append(features.cpu())
    return torch.cat(all_feats)

# Özellik haritasını düzleştir
def flatten_features(features):
    N, C, H, W = features.shape
    return features.permute(0, 2, 3, 1).reshape(-1, C), H, W

# Anomali haritası hesapla (cosine benzerliğe dayalı)
def compute_anomaly_map(test_patch_feats, train_patch_feats, h, w):
    dists = torch.cdist(test_patch_feats.unsqueeze(0), train_patch_feats.unsqueeze(0)).squeeze(0)
    min_dists, _ = torch.min(dists, dim=1)
    anomaly_map = min_dists.reshape(h, w)
    anomaly_map = F.interpolate(anomaly_map.unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
    return anomaly_map.squeeze().numpy()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = "Wood_dataset/wood/train/good"
    test_defect_dir = "Wood_dataset/wood/test/defect"
    test_good_dir = "Wood_dataset/wood/test/good"
    gt_mask_dir = "Wood_dataset/wood/ground_truth/defect"

    train_imgs, _ = load_and_preprocess_images(train_dir)
    test_defect_imgs, defect_paths = load_and_preprocess_images(test_defect_dir)
    test_good_imgs, good_paths = load_and_preprocess_images(test_good_dir)

    test_imgs = test_good_imgs + test_defect_imgs
    test_paths = good_paths + defect_paths

    train_loader = DataLoader(PatchCoreDataset(train_imgs), batch_size=16, shuffle=False)

    model = ResNetBackbone().to(device)
    print("Extracting train features...")
    train_feats = extract_features(train_loader, model, device)
    train_patch_feats, H, W = flatten_features(train_feats)

    print("Running on test images...")
    all_scores = []
    all_labels = []
    iou_scores = []

    for img_tensor, path in zip(test_imgs, test_paths):
        img_tensor = torch.tensor(img_tensor).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor).cpu()
        test_patch_feats, _, _ = flatten_features(feat)

        anomaly_map = compute_anomaly_map(test_patch_feats, train_patch_feats, H, W)
        anomaly_score = np.percentile(anomaly_map, 99)
        all_scores.append(anomaly_score)
        is_defect = "defect" in path
        all_labels.append(1 if is_defect else 0)

        if is_defect:

            fname = os.path.basename(path).replace(".jpg", "_mask.jpg")
            mask_path = os.path.join(gt_mask_dir, fname)
            if os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = cv2.resize(gt_mask, (256, 256))
                gt_mask = (gt_mask > 127).astype(np.uint8)
                pred_mask = (anomaly_map > 1.356).astype(np.uint8)
                iou = np.sum((gt_mask & pred_mask)) / (np.sum((gt_mask | pred_mask)) + 1e-8)
                iou_scores.append(iou)

    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    f1s = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1s)] if len(thresholds) > 0 else 0.5
    y_pred = [1 if s > best_thresh else 0 for s in all_scores]

    f1 = f1_score(all_labels, y_pred)
    auc = roc_auc_score(all_labels, all_scores)
    cm = confusion_matrix(all_labels, y_pred)
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0

    print(f"\n En iyi eşik: {best_thresh:.3f}")
    print(f" F1 Score: {f1:.4f}")
    print(f" ROC AUC: {auc:.4f}")
    print(f" Confusion Matrix:\n{cm}")
    print(f" Ortalama IoU: {mean_iou:.4f}")

    # Örnek görsellerin gösterildiği bölüm
    print("\n Örnek sonuçlar:")
    shown = 0
    for img_tensor, path in zip(test_imgs, test_paths):
        if shown >= 3:
            break
        if "defect" not in path:
            continue

        # Test görselini modele vermek için uygun tensora çeviriyoruz
        img_tensor_show = torch.tensor(img_tensor).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor_show).cpu()
        test_patch_feats_show, _, _ = flatten_features(feat)

        # Test görselinden anomaly map hesaplanıyor
        anomaly_map_show = compute_anomaly_map(test_patch_feats_show, train_patch_feats, H, W)

         # Belirlenen eşik değeri ile predicted mask oluşturuluyor
        pred_mask = (anomaly_map_show > best_thresh).astype(np.uint8)

         # Ground truth maskesi dosya adından alınarak yükleniyor
        fname = os.path.basename(path).replace(".jpg", "_mask.jpg")
        gt_path = os.path.join(gt_mask_dir, fname)
        if os.path.exists(gt_path):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, (256, 256))
        else:
            gt_mask = np.zeros((256, 256), dtype=np.uint8)

        # 4'lü görsel karşılaştırma hazırlanıyor

        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(img_tensor_show.squeeze().permute(1, 2, 0).cpu().numpy())
        plt.title("Orijinal")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(anomaly_map_show, cmap="jet")
        plt.title("Anomaly Map")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(gt_mask, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.suptitle(os.path.basename(path))
        plt.tight_layout()
        plt.show()
        shown += 1