import os, sys, time
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath("RT-DETR/rtdetrv2_pytorch"))
from src.data.dataset import mscoco_category2name, mscoco_label2category
from load_rtdetr import load_rtdetr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {lab: mscoco_category2name[cat] for lab, cat in mscoco_label2category.items()}

def main():
    print(f"Probando RT-DETR en {DEVICE}...")
    model, postprocessor = load_rtdetr(family="v2", size="s", device=DEVICE)
    
    img = cv2.imread("test.jpg")
    h, w = img.shape[:2]
    
    # Warmup
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((640, 640), Image.BILINEAR)
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255)[None].to(DEVICE)
    with torch.no_grad():
        for _ in range(3):
            _ = postprocessor(model(x), torch.tensor([[w, h]], device=DEVICE))
            
    # Benchmark
    t0 = time.time()
    n_iters = 20
    with torch.no_grad():
        for _ in range(n_iters):
            labels, boxes, scores = postprocessor(model(x), torch.tensor([[w, h]], device=DEVICE))
    dt = (time.time() - t0) / n_iters
    fps = 1.0 / dt
    
    labels, boxes, scores = labels[0].cpu().numpy(), boxes[0].cpu().numpy(), scores[0].cpu().numpy()
    keep = scores >= 0.5
    print(f"\nRT-DETR v2-S Inferencia: {dt*1000:.2f} ms por frame ({fps:.1f} FPS)")
    print(f"Detecciones con conf >= 0.5: {np.sum(keep)}")
    for b, s, l in zip(boxes[keep], scores[keep], labels[keep]):
        print(f"  - {CLASS_NAMES.get(int(l), 'Unknown')}: {s:.2f} at {b.astype(int)}")

if __name__ == "__main__":
    main()
