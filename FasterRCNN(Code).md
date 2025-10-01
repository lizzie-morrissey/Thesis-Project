#Faster R-CNN Code

!pip install -q albumentations pandas tqdm scikit-learn

import os, time, random, re
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import cv2
import albumentations as A
import math
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, spearmanr
import torch

import zipfile
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou

from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

from google.colab import drive
drive.mount('/content/drive')

zip_path = "/content/drive/MyDrive/acne04.zip"
extract_path = "/content/ACNE04"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Files extracted to:", extract_path)

for root, dirs, files in os.walk(extract_path):
    print(root, "->", len(files), "files")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DATA_ROOT = "/content/ACNE04"
SPLITS = ["train","valid","test"]
IMG_EXTS = (".jpg")
CLASS_NAMES = ["lesion"]
NUM_CLASSES = 1 + 1

IMG_SIZE = 640
BATCH_SIZE = 4
NUM_WORKERS = 0
EPOCHS = 15
LR = 0.005
IOU_THRESH = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

def load_split_df(split_dir: str):
    csv_path = os.path.join(DATA_ROOT, split_dir, "_annotations.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"filename","width","height","xmin","ymin","xmax","ymax"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{split_dir}/_annotations.csv missing columns: {missing}")
    df["split"] = split_dir
    return df

df_train = load_split_df("train")
df_val   = load_split_df("valid")
df_test  = load_split_df("test")
df_all   = pd.concat([df_train, df_val, df_test], ignore_index=True)

def build_records_for_split(split: str):
    sub = df_all[df_all["split"]==split]
    records = defaultdict(list)
    split_dir = Path(DATA_ROOT)/split
    image_files = [f.name for f in split_dir.iterdir() if f.suffix in IMG_EXTS]
    for fn in image_files:
        records.setdefault(fn, [])
    for _, r in sub.iterrows():
        fn = Path(str(r["filename"])).name
        xmin, ymin, xmax, ymax = float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])
        records[fn].append({"bbox":[xmin,ymin,xmax,ymax], "label": 1})
    return records

train_records = build_records_for_split("train")
val_records = build_records_for_split("valid")
test_records = build_records_for_split("test")

class AcneDataset(Dataset):
    def __init__(self, split, records, train=True):
        self.split = split
        self.split_dir = str(Path(DATA_ROOT)/split)
        self.files = sorted(list(records.keys()))
        self.recs = records
        self.train = train
        self.tf = self._build_tf(train)

    def _build_tf(self, train):
        if train:
            return A.Compose([
                A.LongestMaxSize(max_size=IMG_SIZE),
                A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.04), rotate=(-10, 10), shear=(-4, 4), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=12, val_shift_limit=8, p=0.3),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
                A.MotionBlur(blur_limit=3, p=0.1), ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))
        else:
            return A.Compose([
                A.LongestMaxSize(max_size=IMG_SIZE),
                A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_CONSTANT), ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        path = f"{self.split_dir}/{fn}"
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = [x["bbox"] for x in self.recs[fn]]
        labels = [x["label"] for x in self.recs[fn]]

        t = self.tf(image=img, bboxes=bboxes, labels=labels)
        img = t["image"]; bboxes = t["bboxes"]; labels = t["labels"]

        img_t = torch.from_numpy(img).permute(2,0,1).float()/255.0

        target = {}
        if len(bboxes)==0:
            target["boxes"] = torch.zeros((0,4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        else:
            target["boxes"] = torch.tensor(bboxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)

        target["image_id"] = torch.tensor([idx])
        target["iscrowd"]  = torch.zeros((len(bboxes),), dtype=torch.int64)
        if len(bboxes):
            xyxy = target["boxes"]
            target["area"] = (xyxy[:,2]-xyxy[:,0])*(xyxy[:,3]-xyxy[:,1])
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)
        return img_t, target

def collate_fn(batch): return tuple(zip(*batch))

train_ds = AcneDataset("train", train_records, train=True)
val_ds = AcneDataset("valid", val_records, train=False)
test_ds = AcneDataset("test",  test_records, train=False)

train_counts = np.array([len(train_records[f]) for f in train_ds.files], dtype=np.float32)
oversample_exponent = 0.75
weights = (train_counts + 1.0) ** oversample_exponent
weights = weights / weights.sum()
sampler = WeightedRandomSampler(weights, num_samples=len(train_ds.files), replacement=True)

#Dataloaders
use_pin_memory = torch.cuda.is_available()
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=use_pin_memory)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=use_pin_memory)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=use_pin_memory)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
aspect_ratios = ((0.8, 1.0, 1.25),) * len(anchor_sizes)
anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
model.rpn.anchor_generator = anchor_gen

model.rpn.box_fg_iou_thresh = 0.6
model.rpn.box_bg_iou_thresh = 0.3
model.rpn.nms_thresh = 0.7

model.roi_heads.box_score_thresh = 0.0
model.roi_heads.nms_thresh = 0.45
model.roi_heads.detections_per_img = 300

in_feats = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, NUM_CLASSES)

model.to(DEVICE)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)

warmup_epochs = 1
total_epochs  = EPOCHS

def lr_lambda(current_epoch):
    if current_epoch < warmup_epochs:
        return float(current_epoch + 1) / float(max(1, warmup_epochs))
    progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

lr_sched = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def train_one_epoch(epoch):
    model.train()
    running = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train | Epoch {epoch:02d}", leave=False)
    for i, batch in pbar:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, targets = batch[0], batch[1]
        else:
            raise RuntimeError(f"Unexpected batch structure: {type(batch)} with len={len(batch) if hasattr(batch,'__len__') else 'NA'}")

        images  = [im.to(DEVICE, non_blocking=True) for im in images]
        targets = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += float(loss.item())
        avg_loss = running / (i + 1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")

@torch.no_grad()
def validate(iou_thresh=IOU_THRESH):
    model.eval()
    tp = fp = fn = 0
    aps = []

    pbar = tqdm(val_loader, total=len(val_loader), desc="Valid", leave=False)
    for images, targets in pbar:
        images = [im.to(DEVICE, non_blocking=True) for im in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            pb, ps = out["boxes"].cpu(), out["scores"].cpu()
            gb     = tgt["boxes"]

            if len(ps) > 0:
                order = torch.argsort(ps, descending=True)
                pb_sorted = pb[order]
                hits = []
                used_ap = set()
                for i in range(len(pb_sorted)):
                    if len(gb)==0:
                        hits.append(0); continue
                    ious = box_iou(pb_sorted[i].unsqueeze(0), gb).squeeze(0).numpy()
                    j = ious.argmax()
                    ok = (ious[j] >= iou_thresh) and (j not in used_ap)
                    hits.append(1 if ok else 0)
                    if ok: used_ap.add(j)
                if len(gb)>0 and len(hits)>0:
                    hits = np.array(hits)
                    tp_c = np.cumsum(hits)
                    fp_c = np.cumsum(1-hits)
                    recalls    = tp_c / max(1,len(gb))
                    precisions = tp_c / np.maximum(1, tp_c + fp_c)
                    ap = 0.0
                    for t in np.linspace(0,1,11):
                        p = precisions[recalls>=t].max() if np.any(recalls>=t) else 0
                        ap += p/11.0
                    aps.append(ap)

            # P/R/F1 bookkeeping
            used = set()
            for i in range(len(pb)):
                if len(gb)==0:
                    fp += 1; continue
                ious = box_iou(pb[i].unsqueeze(0), gb).squeeze(0).numpy()
                j = ious.argmax()
                if ious[j] >= iou_thresh and j not in used:
                    tp += 1; used.add(j)
                else:
                    fp += 1
            fn += max(0, len(gb) - len(used))

        prec_sofar = tp / max(1, tp+fp)
        rec_sofar = tp / max(1, tp+fn)
        pbar.set_postfix(P=f"{prec_sofar:.3f}", R=f"{rec_sofar:.3f}")

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision+recall)==0 else 2*precision*recall/(precision+recall)
    mAP50 = float(np.mean(aps)) if len(aps)>0 else 0.0

best_map = -1.0
for epoch in range(1, EPOCHS+1):
    train_one_epoch(epoch)
    lr_sched.step()
    print(f"Epoch {epoch} LR:", lr_sched.get_last_lr()[0])
    mAP50, _ = validate()
    if mAP50 > best_map:
        best_map = mAP50
        torch.save(model.state_dict(), "/content/fasterrcnn_acne04_singleclass_best.pth")
        print("Saved best model")
print(f"Best mAP@0.5: {best_map:.3f}")

@torch.no_grad()
def collect_predictions(model, data_loader, iou_thresh=0.5):
    model.eval()
    all_scores, all_tp = [], []
    num_gt = 0
    img_has_gt, img_max_score = [], []

    pbar = tqdm(data_loader, desc="Collect", leave=False)
    for images, targets in pbar:
        images = [im.to(DEVICE) for im in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            pb = out["boxes"].cpu()
            ps = out["scores"].cpu().numpy()
            gb = tgt["boxes"]

            img_has_gt.append(int(len(gb) > 0))
            img_max_score.append(float(ps.max()) if len(ps) else 0.0)
            num_gt += len(gb)

            if len(ps)==0:
                continue

            order = np.argsort(-ps)
            pb_sorted = pb[order]
            ps_sorted = ps[order]
            matched_gt = set()
            for i in range(len(ps_sorted)):
                if len(gb)==0:
                    all_scores.append(ps_sorted[i]); all_tp.append(0); continue
                ious = box_iou(pb_sorted[i].unsqueeze(0), gb).squeeze(0).numpy()
                j = int(ious.argmax()) if ious.size else -1
                ok = (j >= 0) and (ious[j] >= iou_thresh) and (j not in matched_gt)
                if ok: matched_gt.add(j)
                all_scores.append(ps_sorted[i]); all_tp.append(1 if ok else 0)

    return (np.array(all_scores), np.array(all_tp, dtype=np.int32), int(num_gt),
            np.array(img_has_gt, dtype=np.int32), np.array(img_max_score, dtype=np.float32))

val_scores, val_tp, val_num_gt, _, _ = collect_predictions(model, val_loader, iou_thresh=IOU_THRESH)

platt = LogisticRegression(solver="lbfgs")
if len(val_scores) > 0:
    platt.fit(val_scores.reshape(-1,1), val_tp)
    def platt_cal(scores): return platt.predict_proba(scores.reshape(-1,1))[:,1]
    cal_scores = platt_cal(val_scores)

    order = np.argsort(-cal_scores)
    tp_sorted = val_tp[order]
    fp_sorted = 1 - tp_sorted
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(fp_sorted)
    recalls = tp_cum / max(1, val_num_gt)
    precisions = tp_cum / np.maximum(1, tp_cum + fp_cum)
    f1s = (2 * precisions * recalls) / np.maximum(1e-8, precisions + recalls)
    best_idx = int(np.argmax(f1s))
    CONF_THRESH = float(cal_scores[order][best_idx])
else:
    def platt_cal(scores): return scores
    CONF_THRESH = 0.5

print(f"Calibrated best-F1 threshold (val) = {CONF_THRESH:.3f}")


test_scores, test_tp, test_num_gt, img_has_gt, img_max_score = collect_predictions(model, test_loader, iou_thresh=IOU_THRESH)

if len(test_scores) > 0:
    test_scores_cal = platt_cal(test_scores)
else:
    test_scores_cal = test_scores

if len(test_scores_cal)==0:
    precisions = np.array([0.0]); recalls = np.array([0.0]); f1s = np.array([0.0]); ap11 = 0.0
    best = None
else:
    order = np.argsort(-test_scores_cal)
    tp_sorted = test_tp[order]; fp_sorted = 1 - tp_sorted
    tp_cum = np.cumsum(tp_sorted); fp_cum = np.cumsum(fp_sorted)
    recalls = tp_cum / max(1, test_num_gt)
    precisions = tp_cum / np.maximum(1, tp_cum + fp_cum)

    ap11 = 0.0
    for t in np.linspace(0,1,11):
        p = precisions[recalls>=t].max() if np.any(recalls>=t) else 0.0
        ap11 += p/11.0
    f1s = (2 * precisions * recalls) / np.maximum(1e-8, precisions + recalls)
    best_idx = int(np.argmax(f1s))
    best = {
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(f1s[best_idx]),
        "threshold": float(test_scores_cal[order][best_idx]),
        "tp": int(tp_cum[best_idx]),
        "fp": int(fp_cum[best_idx]),
        "fn": int(max(0, test_num_gt - tp_cum[best_idx])),
    }

print(f"\nTEST Detection (IoU≥0.5)  AP(11-pt): {ap11:.3f}")
if best:
    print(f"Best-F1 (calibrated): thr={best['threshold']:.3f} | P={best['precision']:.3f} R={best['recall']:.3f} F1={best['f1']:.3f} "
          f"| TP={best['tp']} FP={best['fp']} FN={best['fn']} (GT={test_num_gt})")

# Plotting
def plot_detection_pr(precisions, recalls, f1s, ap, best, title="Detection PR (IoU≥0.5)"):
    plt.figure(figsize=(6,5))
    plt.plot(recalls, precisions, label=f"PR curve (AP={ap:.3f})")
    if best:
        plt.scatter([best["recall"]], [best["precision"]], s=60,
                    label=f"Best F1={best['f1']:.3f} @thr={best['threshold']:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.2); plt.show()
    
plot_detection_pr(precisions, recalls, f1s, ap11, best)

@torch.no_grad()
def count_lesions(model, dataset, conf_thresh=0.5, min_box_wh=0):
    "Counts detections per image above conf_thresh; optional tiny-box filter."
    model.eval()
    counts, mean_scores = [], []
    filenames = dataset.files
    for i in tqdm(range(len(dataset)), desc=f"Counting @thr={conf_thresh:.3f}"):
        img_t, _ = dataset[i]
        out = model([img_t.to(DEVICE)])[0]
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()

        if min_box_wh > 0 and len(boxes):
            w = boxes[:,2]-boxes[:,0]; h = boxes[:,3]-boxes[:,1]
            keep_size = (w >= min_box_wh) & (h >= min_box_wh)
            boxes  = boxes[keep_size]; scores = scores[keep_size]

        if len(scores) > 0:
            scores_cal = platt_cal(scores)
        else:
            scores_cal = scores

        keep = scores_cal >= conf_thresh
        counts.append(int(keep.sum()))
        mean_scores.append(float(scores_cal[keep].mean()) if keep.any() else 0.0)

    return pd.DataFrame({"filename": filenames, "lesion_count": counts, "mean_conf": mean_scores})

def derive_percentile_bands(val_counts, p_mild=0.40, p_mod=0.75, p_sev=0.90):
    mild_max = int(np.quantile(val_counts, p_mild))
    mod_max = int(np.quantile(val_counts, p_mod))
    sev_max = int(np.quantile(val_counts, p_sev))
    return mild_max, mod_max, sev_max

def grade_by_percentiles(n, mild_max, mod_max, sev_max):
    if n <= 2: return "clear"
    if n <= mild_max: return "mild"
    if n <= mod_max:  return "moderate"
    if n <= sev_max:  return "severe"
    return "very severe"

thr_for_grading = CONF_THRESH if np.isfinite(CONF_THRESH) else 0.5
print(f"\nSeverity grading using calibrated threshold = {thr_for_grading:.3f}")

val_grade_df = count_lesions(model, val_ds, conf_thresh=thr_for_grading, min_box_wh=0)
mild_max, mod_max, sev_max = derive_percentile_bands(val_grade_df["lesion_count"].values, p_mild=0.40, p_mod=0.75, p_sev=0.90)
val_grade_df["severity"] = val_grade_df["lesion_count"].apply(lambda n: grade_by_percentiles(n, mild_max, mod_max, sev_max))
print("Derived percentile cutoffs:", {"mild_max": mild_max, "mod_max": mod_max, "sev_max": sev_max})

test_grade_df = count_lesions(model, test_ds, conf_thresh=thr_for_grading, min_box_wh=0)
test_grade_df["severity"] = test_grade_df["lesion_count"].apply(lambda n: grade_by_percentiles(n, mild_max, mod_max, sev_max))

def gt_counts_from_records(records, file_list):
    "Count GT boxes per image, aligned to dataset file order."
    return np.array([len(records.get(fn, [])) for fn in file_list], dtype=int)

gt_counts_val = gt_counts_from_records(val_records, val_ds.files)
gt_counts_test = gt_counts_from_records(test_records, test_ds.files)

def derive_gt_bands_from_val(gt_counts_val, p_mild=0.40, p_mod=0.75, p_sev=0.90):
    mild_max = int(np.quantile(gt_counts_val, p_mild))
    mod_max = int(np.quantile(gt_counts_val, p_mod))
    sev_max = int(np.quantile(gt_counts_val, p_sev))
    return mild_max, mod_max, sev_max

mild_max_gt, mod_max_gt, sev_max_gt = derive_gt_bands_from_val(gt_counts_val, 0.40, 0.75, 0.90)

def assign_severity_from_bands(n, mild_max, mod_max, sev_max):
    if n <= 2: return "clear"
    if n <= mild_max: return "mild"
    if n <= mod_max:  return "moderate"
    if n <= sev_max:  return "severe"
    return "very severe"

gt_severity_test = [assign_severity_from_bands(n, mild_max_gt, mod_max_gt, sev_max_gt) for n in gt_counts_test]

@torch.no_grad()
def count_lesions_for_dataset(model, dataset, conf_thresh=0.5):
    model.eval()
    counts = []
    for i in range(len(dataset)):
        img_t, _ = dataset[i]
        out = model([img_t.to(DEVICE)])[0]
        scores = out["scores"].detach().cpu().numpy()
        n = int((scores >= conf_thresh).sum())
        counts.append(n)
    return np.array(counts, dtype=int)

try:
    thr_for_grading = float(CONF_THRESH)
except:
    thr_for_grading = 0.5

pred_counts_test = count_lesions_for_dataset(model, test_ds, conf_thresh=thr_for_grading)

pred_severity_test = [assign_severity_from_bands(n, mild_max_gt, mod_max_gt, sev_max_gt) for n in pred_counts_test]

def safe_corr(x, y):
    if np.std(x)==0 or np.std(y)==0:
        return float('nan'), float('nan')
    pr = pearsonr(x, y)[0]
    sr = spearmanr(x, y).correlation
    return pr, sr

mae = float(np.mean(np.abs(pred_counts_test - gt_counts_test)))
rmse = float(np.sqrt(np.mean((pred_counts_test - gt_counts_test)**2)))
bias = float(np.mean(pred_counts_test - gt_counts_test))
pear, spear = safe_corr(pred_counts_test, gt_counts_test)

print("\nCounting metrics (TEST)")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Bias (pred - GT): {bias:.2f}")
print(f"Pearson r:  {pear:.3f}")
print(f"Spearman ρ: {spear:.3f}")

#Bland-Altman Plot
plt.figure(figsize=(6,5))
m = 0.5*(pred_counts_test + gt_counts_test)
d = pred_counts_test - gt_counts_test
plt.scatter(m, d, alpha=0.6)
mean_diff = np.mean(d); sd = np.std(d)
loa_low, loa_high = mean_diff - 1.96*sd, mean_diff + 1.96*sd
plt.axhline(mean_diff, linestyle='--')
plt.axhline(loa_low, color='r', linestyle='--')
plt.axhline(loa_high, color='r', linestyle='--')
plt.xlabel("Mean of (Pred, GT) lesion counts")
plt.ylabel("Difference (Pred - GT)")
plt.title("Bland–Altman (counts agreement)")
plt.grid(alpha=0.2)
plt.show()

labels_order = ["clear","mild","moderate","severe","very severe"]

def to_index(labels):
    idx_map = {k:i for i,k in enumerate(labels_order)}
    return np.array([idx_map.get(x, -1) for x in labels])

y_true = to_index(gt_severity_test)
y_pred = to_index(pred_severity_test)

valid_mask = (y_true >= 0) & (y_pred >= 0)
y_true_v = y_true[valid_mask]
y_pred_v = y_pred[valid_mask]

acc = accuracy_score(y_true_v, y_pred_v)
macro_f1 = f1_score(y_true_v, y_pred_v, average='macro')
kappa = cohen_kappa_score(y_true_v, y_pred_v)

print("\nSeverity classification (TEST)")
print(f"Accuracy:{acc:.3f}")
print(f"Macro F1:{macro_f1:.3f}")
print(f"Cohen's κ:{kappa:.3f}")

cm = confusion_matrix(y_true_v, y_pred_v, labels=list(range(len(labels_order))))
print("\nConfusion matrix (rows=GT, cols=Pred):")
print(pd.DataFrame(cm, index=labels_order, columns=labels_order))

print("\nClassification report:")
print(classification_report(y_true_v, y_pred_v, target_names=labels_order, zero_division=0))

#Confusion Matrix
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest')
plt.xticks(ticks=range(len(labels_order)), labels=labels_order, rotation=30)
plt.yticks(ticks=range(len(labels_order)), labels=labels_order)
plt.title("Severity confusion matrix (TEST)")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Ground truth")
plt.tight_layout()
plt.show()

@torch.no_grad()
def benchmark_inference(model, dataset, n_samples=50):
    model.eval()
    idxs = np.linspace(0, len(dataset)-1, num=min(n_samples, len(dataset)), dtype=int)
    for i in idxs[:3]:
        _ = model([dataset[i][0].to(DEVICE)])
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    for i in idxs:
        _ = model([dataset[i][0].to(DEVICE)])
    if torch.cuda.is_available(): torch.cuda.synchronize()
    elapsed = time.time() - t0
    per_img = elapsed / len(idxs)
    fps = 1.0 / per_img if per_img > 0 else float('inf')
    return per_img, fps

per_img_s, fps = benchmark_inference(model, test_ds, n_samples=60)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nEfficiency")
print(f"Avg inference time: {per_img_s*1000:.1f} ms / image  (~{fps:.1f} FPS)")
print(f"Trainable parameters: {params/1e6:.2f} M")
