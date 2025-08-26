
"""
PatchCore-style soldering detector for PCB images
=================================================
- Builds a memory bank from *normal solder patches* (from a folder or auto-seeded from the test image)
- Uses mid-layer features of a pre-trained ResNet (layer2+layer3)
- Coreset (k-center greedy) to compress memory
- Scores each patch by NN distance to memory (smaller distance => more solder-like)
- Produces heatmap, binary mask, overlay, and CSV of detections

Install (Python 3.9+):
  pip install torch torchvision opencv-python pandas numpy

Optional (faster search):
  pip install faiss-cpu

Usage:
  # 1) Build memory from normal images
  python patchcore_solder_detect.py train --normal_dir ./normal_pcbs --out memory.npz

  # 2) Inference on one image
  python patchcore_solder_detect.py infer --image pcb.png --memory memory.npz --out_dir ./out

  # (Single-image quick demo without dataset: auto-seed from the same image)
  python patchcore_solder_detect.py infer --image pcb.png --auto_seed --out_dir ./out
"""

import os, sys, math, argparse
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

import torch
import torch.nn.functional as F
from torchvision import models, transforms

try:
    import faiss  # optional
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# ----------------- Feature extractor -----------------
class ResNetMidFeats(torch.nn.Module):
    def __init__(self, name='resnet50'):
        super().__init__()
        if name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif name == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Unsupported backbone")
        self.stem = torch.nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.l1, self.l2, self.l3, self.l4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4

        self.norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    def forward(self, x):
        # x: [B,3,H,W], float32 [0..1]
        x = self.norm(x)
        x = self.stem(x)
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        return f2, f3  # mid-level

def extract_patch_features(img_bgr, model, device='cpu', long_side=896, p=3):
    # resize with aspect ratio
    h, w = img_bgr.shape[:2]
    scale = long_side / max(h, w)
    new_w, new_h = int(round(w*scale)), int(round(h*scale))
    img = cv2.cvtColor(cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).float().permute(2,0,1) / 255.0
    t = t.unsqueeze(0).to(device)

    with torch.no_grad():
        f2, f3 = model(t)
        if f2.shape[-2:] != f3.shape[-2:]:
            f3 = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        Fcat = torch.cat([f2, f3], dim=1)  # [1,C,H,W]

        # locally-aware average
        Fpad = F.pad(Fcat, (p, p, p, p), mode='reflect')
        Favg = torch.nn.AvgPool2d(kernel_size=2*p+1, stride=1)(Fpad)

        # flatten to [N,D] with grid shape
        feat_map = Favg.squeeze(0).permute(1,2,0).contiguous()  # [H,W,C]
        H, W, C = feat_map.shape
        feats = feat_map.view(-1, C).cpu().numpy()
    return feats, (H, W), (new_h, new_w), (h, w)

# ----------------- Seed mask (solder candidates) -----------------
def auto_seed_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31,31))
    tophat = cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, kernel)
    _, seed = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=2)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    return seed

# ----------------- Coreset (k-center greedy) -----------------
def kcenter_greedy(X, budget, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    chosen = []
    idx0 = int(rng.integers(0, n))
    chosen.append(idx0)
    d2 = np.sum((X - X[idx0])**2, axis=1)
    for _ in range(1, budget):
        idx = int(np.argmax(d2))
        chosen.append(idx)
        d2 = np.minimum(d2, np.sum((X - X[idx])**2, axis=1))
    return np.array(chosen, dtype=int)

# ----------------- NN distance -----------------
def nn_distance_numpy(X, Y, batch=8192):
    out = np.empty((X.shape[0],), dtype=np.float32)
    Y2 = np.sum(Y**2, axis=1, keepdims=True).T
    for i in range(0, X.shape[0], batch):
        xb = X[i:i+batch]
        d2 = (np.sum(xb**2, axis=1, keepdims=True) - 2*xb@Y.T) + Y2
        out[i:i+batch] = np.min(d2, axis=1)
    return out

def nn_distance_faiss(X, Y):
    index = faiss.IndexFlatL2(Y.shape[1])
    index.add(Y.astype(np.float32))
    D, I = index.search(X.astype(np.float32), 1)
    return D[:,0]

# ----------------- Memory IO -----------------
def save_memory(path, memory, H, W):
    np.savez(path, memory=memory.astype(np.float32), H=int(H), W=int(W))

def load_memory(path):
    d = np.load(path)
    return d['memory'], int(d['H']), int(d['W'])

# ----------------- Train -----------------
def train_memory(normal_dir: Path, out_path: Path, device='cpu', coreset_ratio=0.1, backbone='resnet50'):
    model = ResNetMidFeats(backbone).to(device).eval()
    all_feats = []
    for img_path in sorted(list(normal_dir.glob("*.png")) + list(normal_dir.glob("*.jpg")) + list(normal_dir.glob("*.jpeg"))):
        img = cv2.imread(str(img_path))
        feats, (H,W), _, _ = extract_patch_features(img, model, device=device)
        # auto seed on each normal image to pick solder-like patches only
        seed = auto_seed_mask(img)
        seed_grid = cv2.resize(seed, (W, H), interpolation=cv2.INTER_NEAREST).reshape(-1)
        idx = np.where(seed_grid>0)[0]
        if idx.size == 0:
            continue
        all_feats.append(feats[idx])
    M = np.concatenate(all_feats, axis=0)
    # compress
    if coreset_ratio < 1.0:
        budget = max(200, int(len(M)*coreset_ratio))
        keep = kcenter_greedy(M, budget, seed=42)
        M = M[keep]
    save_memory(out_path, M, H, W)
    return M, (H, W)

# ----------------- Inference -----------------
def infer(image_path: Path, out_dir: Path, memory=None, memory_path=None, auto_seed=False, device='cpu', backbone='resnet50'):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path))
    model = ResNetMidFeats(backbone).to(device).eval()

    feats, (H,W), (new_h, new_w), (h, w) = extract_patch_features(img, model, device=device)

    if memory is None:
        if memory_path is not None and Path(memory_path).exists():
            memory, _, _ = load_memory(memory_path)
        elif auto_seed:
            seed = auto_seed_mask(img)
            seed_grid = cv2.resize(seed, (W, H), interpolation=cv2.INTER_NEAREST).reshape(-1)
            idx = np.where(seed_grid>0)[0]
            if idx.size == 0:
                # fallback: top-k brightest downsampled patches
                eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                eq_ds = cv2.resize(eq, (W, H), interpolation=cv2.INTER_AREA).reshape(-1)
                idx = np.argsort(eq_ds)[-max(100, int(0.02*W*H)):]
            memory = feats[idx]
        else:
            raise ValueError("Provide --memory or use --auto_seed.")

    # distance
    if HAVE_FAISS:
        d2 = nn_distance_faiss(feats, memory)
    else:
        d2 = nn_distance_numpy(feats, memory)

    sim = np.exp(-d2 / (np.percentile(d2, 90)+1e-6))  # similarity
    sim_map = sim.reshape(H, W)
    sim_map = cv2.GaussianBlur(sim_map, (0,0), 1.0)
    sim_big = cv2.resize(sim_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # threshold
    thr = float(np.quantile(sim_big, 0.92))
    mask = (sim_big >= thr).astype(np.uint8)*255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)

    # blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rows, overlay = [], img.copy()
    count = 0
    for i in range(1, num_labels):
        x,y,bw,bh,area = stats[i]
        if area < 20: continue
        count += 1
        cx, cy = centroids[i]
        rows.append({"id": count, "x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "cx": float(cx), "cy": float(cy), "area": int(area)})
        cv2.rectangle(overlay, (x,y), (x+bw, y+bh), (0,255,0), 2)
        cv2.putText(overlay, str(count), (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # visuals
    denom = float(np.ptp(sim_big))
    denom = denom if denom > 0 else 1e-8
    heat = (255 * (sim_big - float(np.min(sim_big))) / denom).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    # save
    out_heat = out_dir/"heatmap.png"
    out_mask = out_dir/"mask.png"
    out_overlay = out_dir/"overlay.png"
    out_csv = out_dir/"points.csv"

    cv2.imwrite(str(out_heat), heat_color)
    cv2.imwrite(str(out_mask), mask)
    cv2.imwrite(str(out_overlay), overlay)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[OK] Saved: {out_overlay}, {out_heat}, {out_mask}, {out_csv}")
    print(f"Detections: {count}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    ap_train = sub.add_parser("train")
    ap_train.add_argument("--normal_dir", type=str, required=True)
    ap_train.add_argument("--out", type=str, required=True)
    ap_train.add_argument("--coreset", type=float, default=0.1)
    ap_train.add_argument("--backbone", type=str, default="resnet50")

    ap_infer = sub.add_parser("infer")
    ap_infer.add_argument("--image", type=str, required=True)
    ap_infer.add_argument("--memory", type=str, default=None)
    ap_infer.add_argument("--auto_seed", action="store_true")
    ap_infer.add_argument("--out_dir", type=str, required=True)
    ap_infer.add_argument("--backbone", type=str, default="resnet50")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.cmd == "train":
        normal_dir = Path(args.normal_dir)
        out_path = Path(args.out)
        train_memory(normal_dir, out_path, device=device, coreset_ratio=args.coreset, backbone=args.backbone)
    elif args.cmd == "infer":
        infer(Path(args.image), Path(args.out_dir), memory_path=args.memory, auto_seed=args.auto_seed, device=device, backbone=args.backbone)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
