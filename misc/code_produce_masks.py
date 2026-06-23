import os, cv2, torch
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm

class SwinUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="mit_b0",
            encoder_weights=None,   # load checkpoint rồi, không cần imagenet
            in_channels=3,
            classes=1,
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seg_model = SwinUNet().to(device)
ckpt = torch.load("best_swinunet.pth", map_location=device)
seg_model.load_state_dict(ckpt)
seg_model.eval()

def segment_dermoscopic(img_path, model, device, img_size=256, threshold=0.5):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")

    orig_h, orig_w = img_bgr.shape[:2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))

    x = img_resized.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)              # (3,256,256)
    x = torch.from_numpy(x).unsqueeze(0)  # (1,3,256,256)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)                 # (1,1,256,256)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_256 = (prob > threshold).astype(np.uint8) * 255
    mask_orig = cv2.resize(mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return mask_orig
# from pathlib import Pat


sys.path.insert(0, "/marimo/code")

from milk10k_effb2_metadata.data import load_paired_dataframe
# df = load_paired_dataframe(Path("/path/to/milk10k"))
# df có cột dermoscopic_path theo code của bạn
DATA_DIR = Path("/marimo/milk10k")  # sửa đúng path dataset của bạn

df = load_paired_dataframe(DATA_DIR)

print(df[["lesion_id", "dermoscopic_path"]].head())
out_mask_dir = "milk10k_dermoscopic_masks"
os.makedirs(out_mask_dir, exist_ok=True)

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row["dermoscopic_path"]
    lesion_id = row["lesion_id"]

    mask = segment_dermoscopic(img_path, seg_model, device)

    save_path = os.path.join(out_mask_dir, f"{lesion_id}_dermoscopic_mask.png")
    cv2.imwrite(save_path, mask)
    
    import matplotlib.pyplot as plt



def plot_seg_samples(df, out_mask_dir, n=5, seed=123):
    sample_df = df.sample(n, random_state=seed)

    fig, ax = plt.subplots(n, 3, figsize=(15, 5*n))

    if n == 1:
        ax = ax[None, :]

    for i, (_, r) in enumerate(sample_df.iterrows()):
        lid = r["lesion_id"]

        image = cv2.imread(r["dermoscopic_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_img = cv2.imread(
            os.path.join(out_mask_dir, f"{lid}_dermoscopic_mask.png"),
            cv2.IMREAD_GRAYSCALE
        )

        overlay = image.copy()
        overlay[mask_img > 0] = (
            0.7 * overlay[mask_img > 0]
            + 0.3 * np.array([255, 0, 0])
        ).astype(np.uint8)

        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask_img, cmap="gray")
        ax[i, 2].imshow(overlay)

        ax[i, 0].set_title(f"Original: {lid}")
        ax[i, 1].set_title("Mask")
        ax[i, 2].set_title("Overlay")

        for j in range(3):
            ax[i, j].axis("off")

    plt.tight_layout()
    plt.show()
plot_seg_samples(df, out_mask_dir, n=10)