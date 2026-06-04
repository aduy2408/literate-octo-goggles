# Stable Diffusion Augmentation cho MILK10k

Thư mục này có 3 script:

- `prepare_milk10k_sd_training_set.py`: tách data MILK10k thành folder train Stable Diffusion/LoRA cho **1 class** và **1 loại ảnh**.
- `generate_milk10k_sd.py`: dùng Stable Diffusion `img2img` để tạo ảnh augmentation cho **1 class** và **1 loại ảnh**.
- `plot_generated_images.py`: gom ảnh trong 1 folder thành grid/contact sheet để kiểm tra nhanh.

## Điểm quan trọng của MILK10k

Mỗi `lesion_id` trong MILK10k thường có 2 ảnh:

- `clinical: close-up`
- `dermoscopic`

Khi train/generate Stable Diffusion, không nên trộn 2 modality này. Nếu muốn train model tạo ảnh clinical thì chỉ dùng ảnh `clinical: close-up`. Nếu muốn train model tạo ảnh dermoscopic thì chỉ dùng ảnh `dermoscopic`.

Vì vậy các script chỉ cho chọn:

```bash
--image-type clinical_close_up
```

hoặc:

```bash
--image-type dermoscopic
```

Không có chế độ `all`.

## Class hợp lệ

```text
AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC
```

## 1. Chuẩn bị folder train Stable Diffusion/LoRA

Ví dụ tách class `MEL`, chỉ lấy ảnh clinical:

```bash
python Stable_diffusion_augmentation/prepare_milk10k_sd_training_set.py \
  --class-name MEL \
  --image-type clinical_close_up \
  --output-dir Stable_diffusion_augmentation/train_mel_clinical
```

Ví dụ tách class `MEL`, chỉ lấy ảnh dermoscopic:

```bash
python Stable_diffusion_augmentation/prepare_milk10k_sd_training_set.py \
  --class-name MEL \
  --image-type dermoscopic \
  --output-dir Stable_diffusion_augmentation/train_mel_dermoscopic
```

Output:

```text
Stable_diffusion_augmentation/train_mel_clinical/
  images/
    *.jpg
    *.txt
  training_manifest.csv
```

Mặc định script tạo symlink tới ảnh gốc để tiết kiệm dung lượng. Nếu muốn copy file thật:

```bash
python Stable_diffusion_augmentation/prepare_milk10k_sd_training_set.py \
  --class-name MEL \
  --image-type clinical_close_up \
  --output-dir Stable_diffusion_augmentation/train_mel_clinical \
  --copy
```

Mỗi file `.txt` là caption đi kèm ảnh, dùng được với nhiều trainer LoRA/DreamBooth.

## 2. Sinh ảnh augmentation bằng Stable Diffusion img2img

Cài dependency:

```bash
pip install torch diffusers transformers accelerate pillow tqdm
```

Sinh ảnh clinical cho class `MEL`:

```bash
python Stable_diffusion_augmentation/generate_milk10k_sd.py \
  --class-name MEL \
  --image-type clinical_close_up \
  --num-per-image 2 \
  --max-source-images 50 \
  --output-dir Stable_diffusion_augmentation/out_mel_clinical
```

Sinh ảnh dermoscopic cho class `MEL`:

```bash
python Stable_diffusion_augmentation/generate_milk10k_sd.py \
  --class-name MEL \
  --image-type dermoscopic \
  --num-per-image 2 \
  --max-source-images 50 \
  --output-dir Stable_diffusion_augmentation/out_mel_dermoscopic
```

Output:

```text
out_mel_clinical/
  MEL/
    clinical_close_up/
      *.jpg
  augmentation_manifest.csv
```

`augmentation_manifest.csv` lưu ảnh sinh ra, ảnh nguồn, `lesion_id`, `isic_id`, seed, prompt, model, strength.

## 3. Plot tất cả ảnh sinh ra trong 1 folder

Ví dụ sau khi generate ra `Stable_diffusion_augmentation/out_mel_dermoscopic`:

```bash
python Stable_diffusion_augmentation/plot_generated_images.py \
  Stable_diffusion_augmentation/out_mel_dermoscopic \
  --output-file Stable_diffusion_augmentation/out_mel_dermoscopic_contact_sheet.png \
  --columns 5 \
  --thumb-size 224
```

Script mặc định quét đệ quy, nên nó tự tìm ảnh trong các folder con như:

```text
out_mel_dermoscopic/
  MEL/
    dermoscopic/
      *.jpg
```

Nếu folder có quá nhiều ảnh và chỉ muốn xem nhanh 100 ảnh đầu:

```bash
python Stable_diffusion_augmentation/plot_generated_images.py \
  Stable_diffusion_augmentation/out_mel_dermoscopic \
  --max-images 100
```

## Chọn model Stable Diffusion

`generate_milk10k_sd.py` hỗ trợ 2 cách chọn model.

Cách 1: dùng preset:

```bash
python Stable_diffusion_augmentation/generate_milk10k_sd.py \
  --class-name MEL \
  --image-type dermoscopic \
  --model-preset sd15
```

Preset hiện có:

```text
sd15        -> runwayml/stable-diffusion-v1-5
sd21        -> stabilityai/stable-diffusion-2-1
openjourney -> prompthero/openjourney
```

Cách 2: truyền trực tiếp Hugging Face model id hoặc local path:

```bash
python Stable_diffusion_augmentation/generate_milk10k_sd.py \
  --class-name MEL \
  --image-type dermoscopic \
  --model-id runwayml/stable-diffusion-v1-5
```

Nếu bạn đã fine-tune LoRA/DreamBooth và có model local:

```bash
python Stable_diffusion_augmentation/generate_milk10k_sd.py \
  --class-name MEL \
  --image-type dermoscopic \
  --model-id /path/to/your/fine_tuned_model
```

`--model-id` sẽ override `--model-preset`.

Nếu train ra LoRA adapter riêng, giữ base model bằng `--model-preset` hoặc `--model-id`, rồi load LoRA:

```bash
python Stable_diffusion_augmentation/generate_milk10k_sd.py \
  --class-name MEL \
  --image-type dermoscopic \
  --model-preset sd15 \
  --lora-weights /path/to/mel_dermoscopic_lora \
  --lora-scale 0.8
```

## Tham số nên chỉnh

- `--strength`: mức thay đổi so với ảnh gốc. Với ảnh y khoa nên bắt đầu `0.25` đến `0.45`.
- `--guidance-scale`: mức bám prompt. Có thể thử `5.0` đến `8.0`.
- `--steps`: số bước denoise. Thường `25` đến `40` là đủ.
- `--num-per-image`: số ảnh sinh từ mỗi ảnh nguồn.
- `--max-source-images`: giới hạn số ảnh nguồn để test nhanh.
- `--prompt`: tự viết prompt thay cho prompt mặc định.
- `--negative-prompt`: loại bỏ watermark, text, ảnh mờ, v.v.

Ví dụ test nhanh trước:

```bash
python Stable_diffusion_augmentation/generate_milk10k_sd.py \
  --class-name MEL \
  --image-type dermoscopic \
  --max-source-images 5 \
  --num-per-image 1 \
  --strength 0.35 \
  --steps 25 \
  --output-dir Stable_diffusion_augmentation/test_mel_dermoscopic
```

## Gợi ý workflow

1. Dùng `prepare_milk10k_sd_training_set.py` để tách đúng class và đúng modality.
2. Train/fine-tune Stable Diffusion hoặc LoRA bằng folder `images/`.
3. Dùng `generate_milk10k_sd.py --model-id /path/to/model_da_finetune` để sinh augmentation.
4. Kiểm tra chất lượng ảnh sinh trước khi trộn vào pipeline train classifier.

Không nên dùng augmentation sinh ra làm dữ liệu đánh giá hoặc validation/test.

## Workflow khuyến nghị với 8GB VRAM local và RTX A6000

### Trên máy local 8GB VRAM

Chỉ nên dùng để test pipeline, không nên train thật.

Việc nên làm:

- Test tách đúng class và đúng modality.
- Test sinh 3-5 ảnh bằng `img2img`.
- Test xem output folder và manifest có đúng không.
- Test prompt/strength cơ bản.

Lệnh test tách data:

```bash
python Stable_diffusion_augmentation/prepare_milk10k_sd_training_set.py \
  --class-name MEL \
  --image-type dermoscopic \
  --max-images 20 \
  --output-dir Stable_diffusion_augmentation/test_train_mel_dermoscopic
```

Lệnh test generate nhẹ:

```bash
python Stable_diffusion_augmentation/generate_milk10k_sd.py \
  --class-name MEL \
  --image-type dermoscopic \
  --model-preset sd15 \
  --max-source-images 3 \
  --num-per-image 1 \
  --size 512 \
  --strength 0.30 \
  --steps 20 \
  --output-dir Stable_diffusion_augmentation/test_gen_mel_dermoscopic
```

Với 8GB VRAM, nên dùng:

- SD 1.5 ở `512x512`.
- `--steps 20` đến `30`.
- `--num-per-image 1` khi test.
- `--strength 0.25` đến `0.40`.
- Không train full model.

### Trên RTX A6000

Nên train LoRA, không nên full fine-tune ngay từ đầu. Với MILK10k, LoRA đủ để học style/modality/class mà ít overfit hơn full fine-tune.

Nên train riêng từng cặp:

```text
MEL + clinical_close_up
MEL + dermoscopic
BCC + clinical_close_up
BCC + dermoscopic
...
```

Không nên train chung clinical và dermoscopic trong cùng một LoRA nếu mục tiêu là generate riêng từng modality.

Khuyến nghị ban đầu:

- Base model: `sd15`.
- Resolution: `512`.
- LoRA rank: `8` hoặc `16`.
- Batch size: tăng theo VRAM, nhưng bắt đầu từ `4` hoặc `8`.
- Epoch/steps: bắt đầu thấp rồi xem overfit.
- Caption giữ rõ modality, ví dụ `melanoma skin lesion, dermoscopic dermatology image`.

Sau khi train LoRA xong, generate bằng:

```bash
python Stable_diffusion_augmentation/generate_milk10k_sd.py \
  --class-name MEL \
  --image-type dermoscopic \
  --model-preset sd15 \
  --lora-weights /path/to/mel_dermoscopic_lora \
  --lora-scale 0.8 \
  --num-per-image 3 \
  --strength 0.35 \
  --steps 30 \
  --output-dir Stable_diffusion_augmentation/out_mel_dermoscopic_lora
```

Với class cực ít ảnh như `MAL_OTH`, nên cẩn thận vì LoRA rất dễ học thuộc. Bắt đầu bằng `img2img` strength thấp, kiểm tra thủ công, và không đưa ảnh sinh vào validation/test.
