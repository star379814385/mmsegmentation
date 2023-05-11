from pathlib import Path
from mmseg.apis.inference import init_segmentor, inference_segmentor
import cv2
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    cfg_path = r"D:\project\koletorSDD\fcn_unet_s5-d16_4x4_640x256_160k_custom.py"
    checkpoint = r"D:\project\koletorSDD\epoch_200.pth"
    img_root = r"D:\dataset\xx"
    save_root = r"D:\project\koletorSDD\save_vis"
    
    model = init_segmentor(cfg_path, checkpoint, device="cuda:0")
    
    img_paths = list(Path(img_root).rglob("*.jpg"))
    for img_path in tqdm(img_paths):
        img_path = str(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        idmaps = inference_segmentor(model, img)
        detect_mask = (idmaps[0] > 0).astype(np.uint8) * 255
        vis = np.hstack((img[..., 0], detect_mask))
        
        # label_path = img_path.replace(".jpg", "_label.bmp")
        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        # vis = np.hstack((img[..., 0], detect_mask, label))
        save_path = img_path.replace(img_root, save_root)
        if not Path(save_path).parent.exists():
            Path(save_path).parent.mkdir(parents=True)
        cv2.imwrite(save_path, vis)