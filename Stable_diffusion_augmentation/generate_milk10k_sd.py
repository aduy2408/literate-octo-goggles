#!/usr/bin/env python3
"""
Generate Stable Diffusion img2img augmentations for one MILK10k class and one
MILK10k image modality.

MILK10k stores two images per lesion/sample: one clinical close-up image and
one dermoscopic image. This script intentionally requires exactly one modality
so clinical augmentation only uses clinical source images, and dermoscopic
augmentation only uses dermoscopic source images.

Expected data layout:
  MILK10k_Training_Input/<lesion_id>/<isic_id>.jpg
  MILK10k_Training_GroundTruth.csv
  MILK10k_Training_Metadata.csv

Example:
  python Stable_diffusion_augmentation/generate_milk10k_sd.py \
    --class-name MEL \
    --image-type dermoscopic \
    --num-per-image 2 \
    --max-source-images 50 \
    --output-dir Stable_diffusion_augmentation/out_mel
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageFile, ImageOps
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

LABEL_COLUMNS = [
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
]

CLASS_PROMPTS = {
    "AKIEC": "actinic keratosis or intraepithelial carcinoma skin lesion",
    "BCC": "basal cell carcinoma skin lesion",
    "BEN_OTH": "benign other skin lesion",
    "BKL": "benign keratosis-like skin lesion",
    "DF": "dermatofibroma skin lesion",
    "INF": "inflammatory skin lesion",
    "MAL_OTH": "other malignant skin lesion",
    "MEL": "melanoma skin lesion",
    "NV": "melanocytic nevus skin lesion",
    "SCCKA": "squamous cell carcinoma or keratoacanthoma skin lesion",
    "VASC": "vascular skin lesion",
}

IMAGE_TYPE_PROMPTS = {
    "clinical_close_up": "clinical close-up dermatology photograph",
    "dermoscopic": "dermoscopic dermatology image",
}

MODEL_PRESETS = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sd21": "stabilityai/stable-diffusion-2-1",
    "openjourney": "prompthero/openjourney",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable Diffusion img2img augmentation for one MILK10k class.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="MILK10k root folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("Stable_diffusion_augmentation/out"))
    parser.add_argument("--class-name", required=True, choices=LABEL_COLUMNS, help="MILK10k class to augment.")
    parser.add_argument(
        "--image-type",
        choices=["dermoscopic", "clinical_close_up"],
        default="dermoscopic",
        help="MILK10k modality to use as img2img source. Do not mix modalities.",
    )
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS),
        default="sd15",
        help="Named base model preset. Ignored when --model-id is provided.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face model id or local model path. Overrides --model-preset.",
    )
    parser.add_argument(
        "--lora-weights",
        type=Path,
        default=None,
        help="Optional LoRA adapter path/folder to load on top of the base model.",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA strength when --lora-weights is used.")
    parser.add_argument("--num-per-image", type=int, default=1, help="Generated images per selected source image.")
    parser.add_argument("--max-source-images", type=int, default=None, help="Limit number of source images.")
    parser.add_argument("--size", type=int, default=512, help="Square generation size.")
    parser.add_argument("--strength", type=float, default=0.35, help="Img2img noise strength. Try 0.25-0.45 for medical data.")
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle selected source images before limiting.")
    parser.add_argument("--prompt", default=None, help="Override default class prompt.")
    parser.add_argument(
        "--negative-prompt",
        default=(
            "text, watermark, logo, label, ruler, frame, multiple lesions, unrealistic anatomy, "
            "cartoon, painting, low quality, blurry, overexposed, underexposed"
        ),
    )
    parser.add_argument("--fp32", action="store_true", help="Use float32 instead of fp16 on CUDA.")
    parser.add_argument(
        "--allow-black-images",
        action="store_true",
        help="Allow nearly black outputs. By default these fail because they usually mean safety-checker blocking.",
    )
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Disable diffusers safety checker if it incorrectly blocks clinical skin images.",
    )
    return parser.parse_args()


def normalize_image_type(image_type: str) -> str:
    if image_type == "clinical: close-up":
        return "clinical_close_up"
    return image_type.replace(" ", "_").replace(":", "").replace("-", "_")


def load_class_rows(data_dir: Path, class_name: str, image_type: str) -> list[dict[str, str]]:
    input_dir = data_dir / "MILK10k_Training_Input"
    gt_path = data_dir / "MILK10k_Training_GroundTruth.csv"
    meta_path = data_dir / "MILK10k_Training_Metadata.csv"

    if not gt_path.exists() or not meta_path.exists() or not input_dir.exists():
        raise FileNotFoundError(
            "Missing MILK10k files. Expected MILK10k_Training_GroundTruth.csv, "
            "MILK10k_Training_Metadata.csv, and MILK10k_Training_Input under --data-dir."
        )

    selected_lesions = set()
    with gt_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if float(row[class_name]) == 1.0:
                selected_lesions.add(row["lesion_id"])

    rows = []
    with meta_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["lesion_id"] not in selected_lesions:
                continue
            image_type_norm = normalize_image_type(row["image_type"])
            if image_type_norm != image_type:
                continue
            source_path = input_dir / row["lesion_id"] / f"{row['isic_id']}.jpg"
            if not source_path.exists():
                continue
            rows.append(
                {
                    "lesion_id": row["lesion_id"],
                    "isic_id": row["isic_id"],
                    "image_type_norm": image_type_norm,
                    "source_path": str(source_path),
                }
            )

    if not rows:
        raise ValueError(f"No source images found for class={class_name}, image_type={image_type}.")

    return rows


def load_pipeline(args: argparse.Namespace):
    import torch
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except RuntimeError as exc:
        message = str(exc)
        if "flash_attn.flash_attn_interface" in message or "xformers" in message:
            raise RuntimeError(
                "Diffusers failed while importing xformers/flash-attn. xformers is optional for this script; "
                "your install appears broken. Fix with:\n\n"
                "  pip uninstall -y xformers flash-attn\n\n"
                "Then rerun. The script still enables attention slicing on CUDA."
            ) from exc
        raise

    model_id = resolve_model_id(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" or args.fp32 else torch.float16
    kwargs = {"torch_dtype": dtype}
    if args.disable_safety_checker:
        kwargs.update({"safety_checker": None, "feature_extractor": None, "requires_safety_checker": False})

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, **kwargs)
    if args.lora_weights is not None:
        pipe.load_lora_weights(str(args.lora_weights.expanduser().resolve()))
        pipe.fuse_lora(lora_scale=args.lora_scale)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe, device


def resolve_model_id(args: argparse.Namespace) -> str:
    return args.model_id or MODEL_PRESETS[args.model_preset]


def prepare_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS)


def looks_like_blocked_black_image(image: Image.Image) -> bool:
    stat_image = image.convert("L").resize((32, 32), resample=Image.Resampling.BILINEAR)
    pixels = list(stat_image.getdata())
    mean_value = sum(pixels) / max(len(pixels), 1)
    bright_pixels = sum(1 for value in pixels if value > 8)
    return mean_value < 3.0 and bright_pixels / max(len(pixels), 1) < 0.01


def main() -> None:
    args = parse_args()
    import torch

    if args.num_per_image < 1:
        raise ValueError("--num-per-image must be >= 1")
    if not 0.0 <= args.strength <= 1.0:
        raise ValueError("--strength must be in [0, 1]")

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    image_dir = output_dir / args.class_name / args.image_type
    image_dir.mkdir(parents=True, exist_ok=True)
    if not args.disable_safety_checker:
        print(
            "Warning: diffusers safety checker is enabled. Clinical/dermoscopic skin images may be falsely "
            "blocked and returned as black images. If that happens, rerun with --disable-safety-checker."
        )

    rows = load_class_rows(data_dir, args.class_name, args.image_type)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.max_source_images is not None:
        rows = rows[: args.max_source_images]

    prompt = args.prompt or f"{CLASS_PROMPTS[args.class_name]}, {IMAGE_TYPE_PROMPTS[args.image_type]}"
    model_id = resolve_model_id(args)
    pipe, device = load_pipeline(args)

    manifest_path = output_dir / "augmentation_manifest.csv"
    rng = random.Random(args.seed)
    fieldnames = [
        "class_name",
        "generated_path",
        "source_path",
        "lesion_id",
        "source_isic_id",
        "image_type",
        "prompt",
        "negative_prompt",
        "seed",
        "model_id",
        "strength",
        "guidance_scale",
        "steps",
    ]

    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for row in tqdm(rows, total=len(rows), desc=f"Generating {args.class_name}"):
            init_image = prepare_image(Path(row["source_path"]), args.size)
            for aug_idx in range(args.num_per_image):
                seed = rng.randrange(0, 2**31 - 1)
                generator = torch.Generator(device=device).manual_seed(seed)
                result = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    image=init_image,
                    strength=args.strength,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.steps,
                    generator=generator,
                )
                out_name = f"{row['lesion_id']}_{row['isic_id']}_sd_{aug_idx:02d}_seed{seed}.jpg"
                out_path = image_dir / out_name
                image = result.images[0]
                if not args.allow_black_images and looks_like_blocked_black_image(image):
                    raise RuntimeError(
                        "Stable Diffusion returned a nearly black image. This usually means the default safety checker "
                        "blocked a clinical skin image as NSFW. Rerun with:\n\n"
                        "  --disable-safety-checker\n\n"
                        f"Blocked output path would have been: {out_path}"
                    )
                image.save(out_path, quality=95)

                writer.writerow(
                    {
                        "class_name": args.class_name,
                        "generated_path": str(out_path),
                        "source_path": row["source_path"],
                        "lesion_id": row["lesion_id"],
                        "source_isic_id": row["isic_id"],
                        "image_type": row["image_type_norm"],
                        "prompt": prompt,
                        "negative_prompt": args.negative_prompt,
                        "seed": seed,
                        "model_id": model_id,
                        "strength": args.strength,
                        "guidance_scale": args.guidance_scale,
                        "steps": args.steps,
                    }
                )
                f.flush()

    print(f"Saved generated images to: {image_dir}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
