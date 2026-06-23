#!/usr/bin/env python3
"""
Generate paired Stable Diffusion img2img augmentations for minority MILK10k classes.

Each synthetic sample contains both modalities for the same synthetic lesion id:
one clinical close-up image and one dermoscopic image. The output layout is
compatible with predict_milk10k_effb2_dual_metadata.py.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFile, ImageOps
from tqdm.auto import tqdm

from generate_milk10k_sd import CLASS_PROMPTS, IMAGE_TYPE_PROMPTS, LABEL_COLUMNS, MODEL_PRESETS, normalize_image_type

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_MINORITY_CLASSES = ["MAL_OTH", "BEN_OTH", "VASC", "INF", "DF"]
MODALITIES = ("clinical_close_up", "dermoscopic")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
NEUTRAL_METADATA = {
    "age_approx": "",
    "sex": "unknown",
    "skin_tone_class": "",
    "site": "unknown",
}
DEFAULT_NEGATIVE_PROMPT = (
    "text, watermark, logo, label, ruler, frame, multiple lesions, unrealistic anatomy, "
    "cartoon, painting, low quality, blurry, overexposed, underexposed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired Stable Diffusion img2img augmentation for minority MILK10k classes.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="MILK10k root folder.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Image root. Defaults to --data-dir/MILK10k_Training_Input.")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Metadata CSV. Defaults to --data-dir/MILK10k_Training_Metadata.csv.",
    )
    parser.add_argument(
        "--groundtruth-csv",
        type=Path,
        default=None,
        help="Ground-truth CSV. Defaults to --data-dir/MILK10k_Training_GroundTruth.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("Stable_diffusion_augmentation/out_minority_pairs"))
    parser.add_argument(
        "--class-names",
        nargs="+",
        choices=LABEL_COLUMNS,
        default=DEFAULT_MINORITY_CLASSES,
        help="Classes to augment. Defaults to MILK10k classes with fewer than 100 paired lesions.",
    )
    parser.add_argument("--num-per-lesion", type=int, default=1, help="Synthetic paired samples per source lesion.")
    parser.add_argument("--max-source-lesions", type=int, default=None, help="Optional cap per class.")
    parser.add_argument("--size", type=int, default=512, help="Square generation size.")
    parser.add_argument("--strength", type=float, default=0.35, help="Img2img noise strength.")
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle source lesions within each class before limiting.")
    parser.add_argument("--clinical-prompt", default=None, help="Override clinical prompt.")
    parser.add_argument("--dermoscopic-prompt", default=None, help="Override dermoscopic prompt.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="sd15")
    parser.add_argument("--model-id", default=None, help="Hugging Face model id or local model path. Overrides --model-preset.")
    parser.add_argument("--clinical-lora-weights", type=Path, default=None)
    parser.add_argument("--dermoscopic-lora-weights", type=Path, default=None)
    parser.add_argument("--clinical-lora-scale", type=float, default=1.0)
    parser.add_argument("--dermoscopic-lora-scale", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true", help="Do not regenerate images that already exist.")
    parser.add_argument("--diagnose-data", action="store_true", help="Print data matching diagnostics and exit before diffusion.")
    parser.add_argument(
        "--allow-black-images",
        action="store_true",
        help="Allow nearly black outputs. By default these fail because they usually mean the safety checker blocked a medical skin image.",
    )
    parser.add_argument("--fp32", action="store_true", help="Use float32 instead of fp16 on CUDA.")
    parser.add_argument("--disable-safety-checker", action="store_true")
    return parser.parse_args()


def resolve_model_id(args: argparse.Namespace) -> str:
    return args.model_id or MODEL_PRESETS[args.model_preset]


def resolve_data_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    data_dir = args.data_dir.expanduser().resolve()
    input_dir = (args.input_dir or data_dir / "MILK10k_Training_Input").expanduser().resolve()
    gt_path = (args.groundtruth_csv or data_dir / "MILK10k_Training_GroundTruth.csv").expanduser().resolve()
    meta_path = (args.metadata_csv or data_dir / "MILK10k_Training_Metadata.csv").expanduser().resolve()
    if not gt_path.exists() or not meta_path.exists() or not input_dir.exists():
        raise FileNotFoundError(
            "Missing MILK10k files. Expected MILK10k_Training_GroundTruth.csv, "
            "MILK10k_Training_Metadata.csv, and MILK10k_Training_Input. Pass --input-dir, "
            "--metadata-csv, or --groundtruth-csv if they are not under --data-dir."
        )
    return input_dir, gt_path, meta_path


def read_labels(gt_path: Path) -> dict[str, str]:
    labels = {}
    with gt_path.open(newline="") as f:
        for row in csv.DictReader(f):
            for class_name in LABEL_COLUMNS:
                if float(row[class_name]) == 1.0:
                    labels[row["lesion_id"]] = class_name
                    break
    return labels


def resolve_source_image(input_dir: Path, lesion_id: str, isic_id: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        path = input_dir / lesion_id / f"{isic_id}{suffix}"
        if path.exists():
            return path
    return None


def load_paired_rows(input_dir: Path, gt_path: Path, meta_path: Path, class_names: list[str]) -> tuple[list[dict[str, str]], list[str]]:
    paired, metadata_columns, diagnostics = load_paired_rows_with_diagnostics(input_dir, gt_path, meta_path, class_names)
    if not paired:
        raise ValueError(format_diagnostics_error(class_names, input_dir, gt_path, meta_path, diagnostics))
    return paired, metadata_columns


def load_paired_rows_with_diagnostics(
    input_dir: Path,
    gt_path: Path,
    meta_path: Path,
    class_names: list[str],
) -> tuple[list[dict[str, str]], list[str], dict[str, object]]:
    labels = read_labels(gt_path)
    selected = {lesion_id for lesion_id, label in labels.items() if label in class_names}
    by_lesion: dict[str, dict[str, dict[str, str]]] = {}
    metadata_rows_by_class: Counter[str] = Counter()
    metadata_modality_by_class: Counter[tuple[str, str]] = Counter()
    existing_modality_by_class: Counter[tuple[str, str]] = Counter()
    missing_source_examples: list[str] = []

    with meta_path.open(newline="") as f:
        reader = csv.DictReader(f)
        metadata_columns = reader.fieldnames or []
        for row in reader:
            lesion_id = row["lesion_id"]
            if lesion_id not in selected:
                continue
            class_name = labels[lesion_id]
            metadata_rows_by_class[class_name] += 1
            image_type = normalize_image_type(row["image_type"])
            metadata_modality_by_class[(class_name, image_type)] += 1
            if image_type not in MODALITIES:
                continue
            source_path = resolve_source_image(input_dir, lesion_id, row["isic_id"])
            if source_path is None:
                if len(missing_source_examples) < 8:
                    missing_source_examples.append(str(input_dir / lesion_id / f"{row['isic_id']}.jpg"))
                continue
            existing_modality_by_class[(class_name, image_type)] += 1
            copied = dict(row)
            copied["source_path"] = str(source_path)
            copied["image_type_norm"] = image_type
            by_lesion.setdefault(lesion_id, {})[image_type] = copied

    paired = []
    for lesion_id, modalities in by_lesion.items():
        if all(modality in modalities for modality in MODALITIES):
            paired.append(
                {
                    "lesion_id": lesion_id,
                    "class_name": labels[lesion_id],
                    "clinical_source_path": modalities["clinical_close_up"]["source_path"],
                    "clinical_source_isic_id": modalities["clinical_close_up"]["isic_id"],
                    "dermoscopic_source_path": modalities["dermoscopic"]["source_path"],
                    "dermoscopic_source_isic_id": modalities["dermoscopic"]["isic_id"],
                }
            )

    diagnostics = {
        "labels_by_target_class": dict(sorted(Counter(label for label in labels.values() if label in class_names).items())),
        "selected_lesions": len(selected),
        "metadata_rows_by_class": dict(sorted(metadata_rows_by_class.items())),
        "metadata_modality_by_class": {
            f"{class_name}/{modality}": count
            for (class_name, modality), count in sorted(metadata_modality_by_class.items())
        },
        "existing_modality_by_class": {
            f"{class_name}/{modality}": count
            for (class_name, modality), count in sorted(existing_modality_by_class.items())
        },
        "lesions_with_any_existing_modality": len(by_lesion),
        "paired_lesions_by_class": class_counts(paired),
        "missing_source_examples": missing_source_examples,
    }
    return sorted(paired, key=lambda row: (row["class_name"], row["lesion_id"])), metadata_columns, diagnostics


def format_diagnostics_error(
    class_names: list[str],
    input_dir: Path,
    gt_path: Path,
    meta_path: Path,
    diagnostics: dict[str, object],
) -> str:
    lines = [
        f"No paired clinical/dermoscopic lesions found for classes: {', '.join(class_names)}",
        "",
        "Data diagnostics:",
        f"  input_dir={input_dir}",
        f"  groundtruth_csv={gt_path}",
        f"  metadata_csv={meta_path}",
        f"  labels_by_target_class={diagnostics['labels_by_target_class']}",
        f"  selected_lesions={diagnostics['selected_lesions']}",
        f"  metadata_rows_by_class={diagnostics['metadata_rows_by_class']}",
        f"  metadata_modality_by_class={diagnostics['metadata_modality_by_class']}",
        f"  existing_modality_by_class={diagnostics['existing_modality_by_class']}",
        f"  lesions_with_any_existing_modality={diagnostics['lesions_with_any_existing_modality']}",
        f"  paired_lesions_by_class={diagnostics['paired_lesions_by_class']}",
    ]
    missing = diagnostics.get("missing_source_examples") or []
    if missing:
        lines.append("  missing_source_examples=")
        lines.extend(f"    {path}" for path in missing)
    lines.extend(
        [
            "",
            "Most likely fix: pass the real training image root via --input-dir.",
            "Expected image layout: <input-dir>/<lesion_id>/<isic_id>.jpg",
        ]
    )
    return "\n".join(lines)


def print_data_diagnostics(input_dir: Path, gt_path: Path, meta_path: Path, class_names: list[str]) -> None:
    rows, metadata_columns, diagnostics = load_paired_rows_with_diagnostics(input_dir, gt_path, meta_path, class_names)
    print("Data diagnostics")
    print(f"  input_dir={input_dir}")
    print(f"  groundtruth_csv={gt_path}")
    print(f"  metadata_csv={meta_path}")
    print(f"  metadata_columns={len(metadata_columns)}")
    for key, value in diagnostics.items():
        print(f"  {key}={value}")
    print(f"  paired_rows={len(rows)}")


def select_rows(rows: list[dict[str, str]], class_names: list[str], max_source_lesions: int | None, shuffle: bool, seed: int):
    rng = random.Random(seed)
    selected = []
    for class_name in class_names:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        if shuffle:
            rng.shuffle(class_rows)
        if max_source_lesions is not None:
            class_rows = class_rows[:max_source_lesions]
        selected.extend(class_rows)
    return selected


def class_counts(rows: list[dict[str, str]], key: str = "class_name") -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def build_tasks(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str | int]]:
    rng = random.Random(args.seed)
    tasks = []
    for row in rows:
        for aug_idx in range(args.num_per_lesion):
            synthetic_lesion_id = f"{row['lesion_id']}__sdpair_{aug_idx:03d}"
            clinical_isic_id = f"{synthetic_lesion_id}__clinical"
            dermoscopic_isic_id = f"{synthetic_lesion_id}__dermoscopic"
            pair_dir = args.output_dir / "prediction_input" / synthetic_lesion_id
            tasks.append(
                {
                    **row,
                    "synthetic_lesion_id": synthetic_lesion_id,
                    "synthetic_pair_index": aug_idx,
                    "clinical_generated_path": str(pair_dir / f"{clinical_isic_id}.jpg"),
                    "dermoscopic_generated_path": str(pair_dir / f"{dermoscopic_isic_id}.jpg"),
                    "clinical_synthetic_isic_id": clinical_isic_id,
                    "dermoscopic_synthetic_isic_id": dermoscopic_isic_id,
                    "clinical_seed": rng.randrange(0, 2**31 - 1),
                    "dermoscopic_seed": rng.randrange(0, 2**31 - 1),
                }
            )
    return tasks


def log_generation_plan(
    all_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    tasks: list[dict[str, str | int]],
    input_dir: Path,
    gt_path: Path,
    meta_path: Path,
    args: argparse.Namespace,
) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "generation_plan.csv"
    config_path = output_dir / "generation_config.json"

    plan_fields = [
        "class_name",
        "synthetic_lesion_id",
        "source_lesion_id",
        "synthetic_pair_index",
        "clinical_source_path",
        "dermoscopic_source_path",
        "clinical_generated_path",
        "dermoscopic_generated_path",
        "clinical_seed",
        "dermoscopic_seed",
    ]
    with plan_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=plan_fields)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "class_name": task["class_name"],
                    "synthetic_lesion_id": task["synthetic_lesion_id"],
                    "source_lesion_id": task["lesion_id"],
                    "synthetic_pair_index": task["synthetic_pair_index"],
                    "clinical_source_path": task["clinical_source_path"],
                    "dermoscopic_source_path": task["dermoscopic_source_path"],
                    "clinical_generated_path": task["clinical_generated_path"],
                    "dermoscopic_generated_path": task["dermoscopic_generated_path"],
                    "clinical_seed": task["clinical_seed"],
                    "dermoscopic_seed": task["dermoscopic_seed"],
                }
            )

    payload = {
        "input_dir": str(input_dir),
        "groundtruth_csv": str(gt_path),
        "metadata_csv": str(meta_path),
        "output_dir": str(output_dir),
        "class_names": args.class_names,
        "available_source_lesions_by_class": class_counts(all_rows),
        "selected_source_lesions_by_class": class_counts(selected_rows),
        "synthetic_pairs_by_class": class_counts([{"class_name": str(task["class_name"])} for task in tasks]),
        "total_selected_source_lesions": len(selected_rows),
        "total_synthetic_pairs": len(tasks),
        "num_per_lesion": args.num_per_lesion,
        "max_source_lesions": args.max_source_lesions,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "model_id": resolve_model_id(args),
        "model_preset": args.model_preset,
        "size": args.size,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "steps": args.steps,
        "clinical_lora_weights": str(args.clinical_lora_weights) if args.clinical_lora_weights else None,
        "dermoscopic_lora_weights": str(args.dermoscopic_lora_weights) if args.dermoscopic_lora_weights else None,
        "clinical_lora_scale": args.clinical_lora_scale,
        "dermoscopic_lora_scale": args.dermoscopic_lora_scale,
        "skip_existing": args.skip_existing,
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Generation plan")
    print(f"  Input dir: {input_dir}")
    print(f"  Metadata CSV: {meta_path}")
    print(f"  Ground truth CSV: {gt_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Available source lesions by class: {payload['available_source_lesions_by_class']}")
    print(f"  Selected source lesions by class: {payload['selected_source_lesions_by_class']}")
    print(f"  Synthetic pairs by class: {payload['synthetic_pairs_by_class']}")
    print(f"  Total synthetic pairs: {len(tasks)}")
    print(f"  Model: {resolve_model_id(args)}, strength={args.strength}, steps={args.steps}, size={args.size}")
    print(f"  Saved plan: {plan_path}")
    print(f"  Saved config: {config_path}")


def load_pipeline(args: argparse.Namespace, lora_weights: Path | None, lora_scale: float):
    import torch
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except RuntimeError as exc:
        message = str(exc)
        if "flash_attn.flash_attn_interface" in message or "xformers" in message:
            raise RuntimeError(
                "Diffusers failed while importing xformers/flash-attn. Your environment likely has a broken "
                "xformers install. For this SD 1.5 img2img script, xformers is optional; uninstall it and rerun:\n\n"
                "  pip uninstall -y xformers flash-attn\n\n"
                "Then keep using attention slicing, which this script enables automatically on CUDA. "
                "If you really need xformers, use a Python 3.10/3.11 CUDA environment with matching torch, "
                "xformers, and flash-attn wheels."
            ) from exc
        raise

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" or args.fp32 else torch.float16
    kwargs = {"torch_dtype": dtype}
    if args.disable_safety_checker:
        kwargs.update({"safety_checker": None, "feature_extractor": None, "requires_safety_checker": False})

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(resolve_model_id(args), **kwargs)
    if lora_weights is not None:
        pipe.load_lora_weights(str(lora_weights.expanduser().resolve()))
        pipe.fuse_lora(lora_scale=lora_scale)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe, device


def prepare_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.fit(img.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)


def looks_like_blocked_black_image(image: Image.Image) -> bool:
    stat_image = image.convert("L").resize((32, 32), resample=Image.Resampling.BILINEAR)
    pixels = list(stat_image.getdata())
    mean_value = sum(pixels) / max(len(pixels), 1)
    bright_pixels = sum(1 for value in pixels if value > 8)
    return mean_value < 3.0 and bright_pixels / max(len(pixels), 1) < 0.01


def modality_prompt(args: argparse.Namespace, class_name: str, modality: str) -> str:
    if modality == "clinical_close_up" and args.clinical_prompt:
        return args.clinical_prompt
    if modality == "dermoscopic" and args.dermoscopic_prompt:
        return args.dermoscopic_prompt
    return f"{CLASS_PROMPTS[class_name]}, {IMAGE_TYPE_PROMPTS[modality]}"


def generate_modality(tasks: list[dict[str, str | int]], args: argparse.Namespace, modality: str) -> None:
    import torch

    if modality == "clinical_close_up":
        lora_weights = args.clinical_lora_weights
        lora_scale = args.clinical_lora_scale
        source_key = "clinical_source_path"
        output_key = "clinical_generated_path"
        seed_key = "clinical_seed"
    else:
        lora_weights = args.dermoscopic_lora_weights
        lora_scale = args.dermoscopic_lora_scale
        source_key = "dermoscopic_source_path"
        output_key = "dermoscopic_generated_path"
        seed_key = "dermoscopic_seed"

    pipe, device = load_pipeline(args, lora_weights, lora_scale)
    desc = f"Generating {modality}"
    generated = 0
    skipped = 0
    for task in tqdm(tasks, total=len(tasks), desc=desc):
        out_path = Path(str(task[output_key]))
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        init_image = prepare_image(Path(str(task[source_key])), args.size)
        generator = torch.Generator(device=device).manual_seed(int(task[seed_key]))
        result = pipe(
            prompt=modality_prompt(args, str(task["class_name"]), modality),
            negative_prompt=args.negative_prompt,
            image=init_image,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=generator,
        )
        image = result.images[0]
        if not args.allow_black_images and looks_like_blocked_black_image(image):
            raise RuntimeError(
                "Stable Diffusion returned a nearly black image. This usually means the default safety checker "
                "blocked a clinical skin image as NSFW. Rerun with:\n\n"
                "  --disable-safety-checker\n\n"
                f"Blocked output path would have been: {out_path}"
            )
        image.save(out_path, quality=95)
        generated += 1

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{modality}: generated={generated}, skipped_existing={skipped}")


def neutral_metadata_row(columns: list[str], task: dict[str, str | int], modality: str) -> dict[str, str]:
    row = {column: "" for column in columns}
    row["lesion_id"] = str(task["synthetic_lesion_id"])
    row["image_type"] = "clinical: close-up" if modality == "clinical_close_up" else "dermoscopic"
    row["isic_id"] = str(task["clinical_synthetic_isic_id"] if modality == "clinical_close_up" else task["dermoscopic_synthetic_isic_id"])
    row["attribution"] = "Stable Diffusion synthetic augmentation"
    row["copyright_license"] = "synthetic"
    row["image_manipulation"] = "synthetic"
    for key, value in NEUTRAL_METADATA.items():
        if key in row:
            row[key] = value
    for column in columns:
        if column.startswith("MONET_"):
            row[column] = "0"
    return row


def write_outputs(tasks: list[dict[str, str | int]], metadata_columns: list[str], args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = resolve_model_id(args)

    manifest_path = output_dir / "paired_augmentation_manifest.csv"
    manifest_fields = [
        "class_name",
        "synthetic_lesion_id",
        "source_lesion_id",
        "synthetic_pair_index",
        "clinical_generated_path",
        "dermoscopic_generated_path",
        "clinical_source_path",
        "dermoscopic_source_path",
        "clinical_source_isic_id",
        "dermoscopic_source_isic_id",
        "clinical_synthetic_isic_id",
        "dermoscopic_synthetic_isic_id",
        "clinical_seed",
        "dermoscopic_seed",
        "clinical_prompt",
        "dermoscopic_prompt",
        "negative_prompt",
        "model_id",
        "strength",
        "guidance_scale",
        "steps",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "class_name": task["class_name"],
                    "synthetic_lesion_id": task["synthetic_lesion_id"],
                    "source_lesion_id": task["lesion_id"],
                    "synthetic_pair_index": task["synthetic_pair_index"],
                    "clinical_generated_path": task["clinical_generated_path"],
                    "dermoscopic_generated_path": task["dermoscopic_generated_path"],
                    "clinical_source_path": task["clinical_source_path"],
                    "dermoscopic_source_path": task["dermoscopic_source_path"],
                    "clinical_source_isic_id": task["clinical_source_isic_id"],
                    "dermoscopic_source_isic_id": task["dermoscopic_source_isic_id"],
                    "clinical_synthetic_isic_id": task["clinical_synthetic_isic_id"],
                    "dermoscopic_synthetic_isic_id": task["dermoscopic_synthetic_isic_id"],
                    "clinical_seed": task["clinical_seed"],
                    "dermoscopic_seed": task["dermoscopic_seed"],
                    "clinical_prompt": modality_prompt(args, str(task["class_name"]), "clinical_close_up"),
                    "dermoscopic_prompt": modality_prompt(args, str(task["class_name"]), "dermoscopic"),
                    "negative_prompt": args.negative_prompt,
                    "model_id": model_id,
                    "strength": args.strength,
                    "guidance_scale": args.guidance_scale,
                    "steps": args.steps,
                }
            )

    metadata_path = output_dir / "metadata_for_prediction.csv"
    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_columns)
        writer.writeheader()
        for task in tasks:
            writer.writerow(neutral_metadata_row(metadata_columns, task, "clinical_close_up"))
            writer.writerow(neutral_metadata_row(metadata_columns, task, "dermoscopic"))

    gt_path = output_dir / "groundtruth_for_prediction.csv"
    with gt_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lesion_id", *LABEL_COLUMNS])
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "lesion_id": task["synthetic_lesion_id"],
                    **{class_name: "1.0" if class_name == task["class_name"] else "0.0" for class_name in LABEL_COLUMNS},
                }
            )

    print(f"Saved paired manifest: {manifest_path}")
    print(f"Saved neutral prediction metadata: {metadata_path}")
    print(f"Saved synthetic ground truth: {gt_path}")


def main() -> None:
    args = parse_args()
    if args.num_per_lesion < 1:
        raise ValueError("--num-per-lesion must be >= 1")
    if not 0.0 <= args.strength <= 1.0:
        raise ValueError("--strength must be in [0, 1]")

    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.disable_safety_checker:
        print(
            "Warning: diffusers safety checker is enabled. Clinical/dermoscopic skin images may be falsely "
            "blocked and returned as black images. If that happens, rerun with --disable-safety-checker."
        )

    input_dir, gt_path, meta_path = resolve_data_paths(args)
    if args.diagnose_data:
        print_data_diagnostics(input_dir, gt_path, meta_path, args.class_names)
        return

    all_rows, metadata_columns = load_paired_rows(input_dir, gt_path, meta_path, args.class_names)
    rows = select_rows(all_rows, args.class_names, args.max_source_lesions, args.shuffle, args.seed)
    tasks = build_tasks(rows, args)
    if not tasks:
        raise ValueError("No source lesions selected for generation.")

    log_generation_plan(all_rows, rows, tasks, input_dir, gt_path, meta_path, args)
    generate_modality(tasks, args, "clinical_close_up")
    generate_modality(tasks, args, "dermoscopic")
    write_outputs(tasks, metadata_columns, args)

    print(f"Generated {len(tasks)} paired synthetic lesions under: {args.output_dir / 'prediction_input'}")


if __name__ == "__main__":
    main()
