# ==== Gerekli Kütüphaneler ====
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm  # işlem sırasında ilerleme çubuğu göstermek için
from sklearn.covariance import LedoitWolf  # kovaryans matrisi için
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from PIL import Image

# ==== VERİ ÖN İŞLEME ====
def load_and_preprocess_images(root_dir):
    image_paths = []
    processed_images = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Görseli 256x256 boyutuna getir
                img = cv2.resize(img, (256, 256))
                img = img.astype(np.float32) / 255.0  # normalize et (0-1 arası)

                # Görseli griye çevir (dark region analizi için)
                gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

                # Otsu threshold ile koyu bölgeleri bul
                _, dark_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                dark_mask = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

                # Orijinal görsel + dark mask ortalaması
                combined = img * 0.5 + dark_mask * 0.5

                processed_images.append(combined)
                image_paths.append(img_path)

    return processed_images, image_paths

# PyTorch dataset sınıfı: görselleri modele uygun formatta döner
class CustomImageDataset(Dataset):
    def __init__(self, images_np):
        self.images = images_np

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = torch.tensor(img).permute(2, 0, 1)  # [H, W, C] → [C, H, W]
        return img.float()

# ==== PaDiM Feature Extractor (ResNet18) ====
class ResNet18_FeatureExtractor(nn.Module):
    def __init__(self, layers_to_extract=["layer1", "layer2", "layer3"]):
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.layers_to_extract = layers_to_extract
        self.layer_outputs = {}

        # Hook fonksiyonu ile feature map'leri yakala
        def hook(module, input, output, name):
            self.layer_outputs[name] = output

        # Belirtilen katmanlara hook ekle
        if "layer1" in layers_to_extract:
            model.layer1.register_forward_hook(lambda m, i, o: hook(m, i, o, "layer1"))
        if "layer2" in layers_to_extract:
            model.layer2.register_forward_hook(lambda m, i, o: hook(m, i, o, "layer2"))
        if "layer3" in layers_to_extract:
            model.layer3.register_forward_hook(lambda m, i, o: hook(m, i, o, "layer3"))

        self.model = model

    def forward(self, x):
        _ = self.model(x)
        # Yakalanan layer çıktılarını sırayla döndür
        outputs = [self.layer_outputs[k] for k in self.layers_to_extract]
        return outputs

# İki feature map’i aynı boyuta getirip birleştir
def embedding_concat(f1, f2, output_size):
    f1 = F.interpolate(f1, size=output_size, mode='bilinear', align_corners=False)
    return torch.cat([f1, f2], dim=1)

# Eğitim verilerinden embedding çıkar
def compute_embeddings(dataloader, model, device):
    embeddings = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding çıkarılıyor..."):
            batch = batch.to(device)
            features = model(batch)

            # Feature map'leri en küçük boyuta indir ve birleştir
            target_size = features[1].shape[2:]
            resized = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in features]
            embedding = torch.cat(resized, dim=1)
            embeddings.append(embedding.cpu().numpy())

    return np.concatenate(embeddings, axis=0)

# Ortalama ve kovaryans matrisi hesapla
def compute_mean_cov(embedding):
    N, C, H, W = embedding.shape
    embedding = embedding.transpose(0, 2, 3, 1).reshape(-1, C)  # [N, H, W, C] → [N*H*W, C]
    mean_per_patch = np.mean(embedding, axis=0)
    cov = LedoitWolf().fit(embedding).covariance_
    return mean_per_patch, cov

# Mahalanobis uzaklığına dayalı anomaly heatmap üret
def mahalanobis_map(embedding, mean, cov_inv):
    N, C, H, W = embedding.shape
    embedding = embedding[0].reshape(C, H * W).T  # [H*W, C]
    dist = [np.sqrt((e - mean).T @ cov_inv @ (e - mean)) for e in embedding]
    return np.array(dist).reshape(H, W)

# Her görsel için anomaly skoru hesapla (max değer)
def compute_mahalanobis_scores(test_loader, model, mean, cov_inv, device):
    scores = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Skor hesaplanıyor..."):
            batch = batch.to(device)
            features = model(batch)

            target_size = features[1].shape[2:]
            resized = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in features]
            embedding = torch.cat(resized, dim=1).cpu().numpy()

            score_map = mahalanobis_map(embedding, mean, cov_inv)
            image_score = np.max(score_map)  # maksimum anomali skoru
            scores.append(image_score)
    return scores

# Segmentasyon için IoU ve F1 score hesapla
def compute_segmentation_metrics(pred_mask, true_mask):
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    iou = intersection / (union + 1e-6)
    f1 = f1_score(true_mask, pred_mask)

    return iou, f1

# Segmentasyon performansını değerlendir
def evaluate_segmentation(test_loader, test_paths, model, mean, cov_inv, device, gt_mask_dir, threshold=50):
    iou_list = []
    f1_list = []

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Segmentasyon Testi")):
            batch = batch.to(device)
            features = model(batch)

            target_size = features[1].shape[2:]
            resized = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in features]
            embedding = torch.cat(resized, dim=1).cpu().numpy()

            heatmap = mahalanobis_map(embedding, mean, cov_inv)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
            heatmap = heatmap ** 2.20

            heatmap_resized = cv2.resize(heatmap, (256, 256))
            bin_mask = (heatmap_resized > (threshold / 255)).astype(np.uint8)

            test_img_name = os.path.basename(test_paths[idx]).replace(".jpg", "_mask.jpg")
            gt_path = os.path.join(gt_mask_dir, test_img_name)
            gt_mask = np.array(Image.open(gt_path).resize((256, 256)))
            gt_mask = (gt_mask > 127).astype(np.uint8)

            iou, f1 = compute_segmentation_metrics(bin_mask, gt_mask)
            iou_list.append(iou)
            f1_list.append(f1)

            if idx < 3:
                original = cv2.imread(test_paths[idx])
                original = cv2.resize(original, (256, 256))
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

                show_comparison(idx, original, heatmap_resized, bin_mask, gt_mask)

    avg_iou = np.mean(iou_list)
    avg_f1 = np.mean(f1_list)
    print(f"\n✅ Ortalama IoU: {avg_iou:.4f}")

# Görselleri karşılaştırmak için çizim fonksiyonu
def show_comparison(index, original_img, heatmap_resized, pred_mask, gt_mask):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ['Orijinal', 'Anomaly Heatmap', 'Tahmin Maskesi', 'Gerçek Mask']
    images = [original_img, heatmap_resized, pred_mask*255, gt_mask*255]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img if len(img.shape) == 3 else img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle(f"Görsel {index}", fontsize=16)
    plt.tight_layout()
    plt.show()

# ==== ANA PROGRAM ====
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Eğitim ve test görsellerini yükle ve ön işle
    train_images, _ = load_and_preprocess_images("Wood_dataset/wood/train/good")
    test_images, _ = load_and_preprocess_images("Wood_dataset/wood/test/defect")

    train_loader = DataLoader(CustomImageDataset(train_images), batch_size=8, shuffle=False)
    test_loader = DataLoader(CustomImageDataset(test_images), batch_size=1, shuffle=False)

    # Test için 'good' görseller de ayrıca alınır (karşılaştırma için)
    test_good_images, _ = load_and_preprocess_images("Wood_dataset/wood/test/good")
    test_good_loader = DataLoader(CustomImageDataset(test_good_images), batch_size=1, shuffle=False)

    model = ResNet18_FeatureExtractor().to(device)

    # Eğitim görsellerinden feature'lar çıkar
    train_embed = compute_embeddings(train_loader, model, device)
    print("Embedding shape:", train_embed.shape)

    # Eğitimden ortalama ve kovaryans hesapla
    mean, cov = compute_mean_cov(train_embed)
    cov_inv = np.linalg.inv(cov)

    # Anomaly heatmap örneği oluştur (tek görsel üzerinden)
    test_embed = compute_embeddings(test_loader, model, device)
    heatmap = mahalanobis_map(test_embed, mean, cov_inv)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

    heatmap_resized = cv2.resize(heatmap, (256, 256))
    plt.imshow(heatmap_resized, cmap='jet')
    plt.title("PaDiM Anomaly Heatmap")
    plt.axis('off')
    plt.colorbar()
    plt.show()

    # Good ve defect görseller için skorlar
    good_scores = compute_mahalanobis_scores(test_good_loader, model, mean, cov_inv, device)
    defect_scores = compute_mahalanobis_scores(test_loader, model, mean, cov_inv, device)

    # Sınıflandırma skorları ve etiketler
    y_true = [0]*len(good_scores) + [1]*len(defect_scores)
    y_scores = good_scores + defect_scores

    # Basit bir eşik değeri ile tahmin et
    threshold = 95.0
    y_pred = [1 if s > threshold else 0 for s in y_scores]

    # Performans metrikleri
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)

    # Sonuçları yazdır
    print(f"🔹 Eşik: {threshold}")
    print(f"🔹 F1 Score: {f1:.4f}")
    print(f"🔹 ROC AUC: {auc:.4f}")
    print("🔹 Confusion Matrix:\n", cm)

    print("İlk 5 good skor:", good_scores[:5])
    print("İlk 5 defect skor:", defect_scores[:5])

    # Segmentasyon testi
    test_defect_images, test_defect_paths = load_and_preprocess_images("Wood_dataset/wood/test/defect")
    test_defect_loader = DataLoader(CustomImageDataset(test_defect_images), batch_size=1, shuffle=False)

    evaluate_segmentation(
        test_loader=test_defect_loader,
        test_paths=test_defect_paths,
        model=model,
        mean=mean,
        cov_inv=cov_inv,
        device=device,
        gt_mask_dir="Wood_dataset/wood/ground_truth/defect",
        threshold=60
    )
