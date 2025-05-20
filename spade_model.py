# ==== GEREKLİ KÜTÜPHANELER ====
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, confusion_matrix, jaccard_score
import matplotlib.pyplot as plt
from PIL import Image

# ==== ÖZEL DATASET SINIFI ====
class WoodDataset(Dataset):
    def __init__(self, image_list):
        self.images = image_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Görseli tensor'a çevirip [H,W,C] → [C,H,W] yap
        img = self.images[idx]
        img = torch.tensor(img).permute(2, 0, 1)
        return img

# ==== GÖRSEL YÜKLEME VE ÖN İŞLEME ====
def load_and_preprocess_images(root_dir):
    image_paths, processed_images = [], []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Görseli sabit boyuta getir ve normalize et
                img = cv2.resize(img, (256, 256))
                img = img.astype(np.float32) / 255.0

                # Koyu bölgeleri bulmak için griye çevir ve threshold uygula
                gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                _, dark_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                dark_mask = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

                # Gaussian merkez vurgulu ağırlık maskesi oluştur
                gauss = cv2.getGaussianKernel(256, 75)
                gauss_map = gauss @ gauss.T
                gauss_map = (gauss_map - gauss_map.min()) / (gauss_map.max() - gauss_map.min())
                gauss_map = np.stack([gauss_map] * 3, axis=-1)

                # Maskeye Gaussian etki uygula
                dark_mask = dark_mask * (gauss_map ** 0.1)

                # Orijinal görsel + maske harmanla
                combined = (img * 0.20 + dark_mask * 0.80).astype(np.float32)
                processed_images.append(combined)
                image_paths.append(img_path)

    return processed_images, image_paths

# ==== GROUND TRUTH MASKELERİ YÜKLE ====
def load_ground_truth_masks(mask_dir, image_paths):
    masks = []
    for path in image_paths:
        fname = os.path.basename(path).replace(".jpg", "_mask.jpg")
        mask_path = os.path.join(mask_dir, fname)
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).resize((256, 256)))
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = np.zeros((256, 256), dtype=np.uint8)
        masks.append(mask)
    return masks

# ==== ÖZELLİK ÇIKARICI MODEL ====
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Backbone olarak sadece ilk katmanları alıyoruz → çıkış boyutu 32x32
        self.backbone = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1  # Buraya kadar alındığında feature map boyutu 32x32 olur
        )

    def forward(self, x):
        return self.backbone(x)

# ==== ÖZELLİK (FEATURE) ÇIKAR ====
def extract_features(dataloader, model, device):
    model.eval()
    features = []

    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            feat = model(x)  # Feature map al
            features.append(feat.cpu())

    return torch.cat(features)  # Bütün batch’leri birleştir

# ==== ANOMALY MAP HESAPLAMA ====
def compute_anomaly_map(test_feat, train_feats, power=1.8):
    N, C, H, W = train_feats.shape

    # Test görselini tüm train görselleriyle kıyaslamak için N kez tekrarla
    test_feat = test_feat.unsqueeze(0).repeat(N, 1, 1, 1)

    # Cosine benzerliğini hesapla
    similarity = F.cosine_similarity(test_feat, train_feats, dim=1)

    # Benzerlikten 1 çıkarılarak "anomalilik skoru" elde edilir
    anomaly_map = 1 - similarity.mean(dim=0)

    # Normalize edip güç uygula (heatmap’te kontrast artırır)
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    anomaly_map = anomaly_map ** power

    # Anomaly skor olarak en yüksek %1 değer alınır
    anomaly_score = np.percentile(anomaly_map.detach().cpu().numpy(), 99)
    return anomaly_map.detach().cpu().numpy(), anomaly_score

# ==== ANA PROGRAM ====
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Veri yolları ===
    train_dir = "Wood_dataset/wood/train/good"
    test_good_dir = "Wood_dataset/wood/test/good"
    test_defect_dir = "Wood_dataset/wood/test/defect"
    gt_mask_dir = "Wood_dataset/wood/ground_truth/defect"

    print("Loading images...")
    train_imgs, _ = load_and_preprocess_images(train_dir)
    test_imgs, test_paths = load_and_preprocess_images(test_good_dir)
    defect_imgs, defect_paths = load_and_preprocess_images(test_defect_dir)

    # Good + Defect testlerini birleştir
    test_imgs += defect_imgs
    test_paths += defect_paths

    # === Eğitim verisini yükle ===
    train_loader = DataLoader(WoodDataset(train_imgs), batch_size=16, shuffle=False)
    model = FeatureExtractor().to(device)

    print("Extracting features...")
    train_features = extract_features(train_loader, model, device)

    # === Sınıflandırma ve segmentasyon skorları ===
    all_scores, all_labels, iou_scores = [], [], []
    visuals = []

    # Test görselleri üzerinde dön
    for img_tensor, path in zip(test_imgs, test_paths):
        img_tensor = torch.tensor(img_tensor).permute(2, 0, 1).unsqueeze(0).to(device)
        test_feat = model(img_tensor)[0]

        # Anomaly heatmap + skor hesapla
        anomaly_map, anomaly_score = compute_anomaly_map(test_feat, train_features)
        anomaly_map_up = cv2.resize(anomaly_map, (256, 256), interpolation=cv2.INTER_CUBIC)

        all_scores.append(anomaly_score)
        label = 1 if "defect" in path else 0
        all_labels.append(label)

        # Segmentasyon yapılacaksa
        if label == 1:
            gt_mask = load_ground_truth_masks(gt_mask_dir, [path])[0]
            pred_mask = (anomaly_map_up > 0.4).astype(np.uint8)

            # Gürültü azaltma: açma-kapama morfoloji işlemleri
            kernel = np.ones((3, 3), np.uint8)
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            iou = jaccard_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=0)
            iou_scores.append(iou)

            # İlk 3 görseli görselleştirmek için sakla
            if len(visuals) < 3:
                visuals.append((img_tensor[0].permute(1, 2, 0).cpu().numpy(), anomaly_map_up, pred_mask, gt_mask))

    # === Sınıflandırma metrikleri ===
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    f1s = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1s)] if len(thresholds) else 0.5

    # En iyi eşiğe göre tahmin yap
    y_pred = [1 if s > best_thresh else 0 for s in all_scores]

    f1_cls = f1_score(all_labels, y_pred)
    auc_cls = roc_auc_score(all_labels, all_scores)
    cm = confusion_matrix(all_labels, y_pred)
    mean_iou = np.mean(iou_scores)

    # === Görselleştirme ===
    for i, (orig, heatmap, pred, gt) in enumerate(visuals):
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1); plt.imshow(orig); plt.title("Orijinal"); plt.axis("off")
        plt.subplot(1, 4, 2); plt.imshow(heatmap, cmap="gray"); plt.title("Anomaly Heatmap"); plt.axis("off")
        plt.subplot(1, 4, 3); plt.imshow(pred, cmap="gray"); plt.title("Tahmin Maskesi"); plt.axis("off")
        plt.subplot(1, 4, 4); plt.imshow(gt, cmap="gray"); plt.title("Gerçek Mask"); plt.axis("off")
        plt.suptitle(f"Görsel {i}", fontsize=14)
        plt.tight_layout()
        plt.show()

    # === Sonuçları yazdır ===
    print(f"\n🔹 Eşik: {best_thresh * 100:.1f}")
    print(f"🔹 F1 Score: {f1_cls:.4f}")
    print(f"🔹 ROC AUC: {auc_cls:.4f}")
    print(f"🔹 Confusion Matrix:\n{cm}")
    print(f"✅ Ortalama IoU: {mean_iou:.4f}")
