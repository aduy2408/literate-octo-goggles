---
task_categories:
- image-classification
pretty_name: MILK10k
---

# MILK10k

This repository contains a staged copy of the MILK10k training data prepared from the local files in this workspace.

## Files

- `metadata.csv`: Hugging Face imagefolder metadata with `file_name`, diagnosis label, lesion id, ISIC id, modality, MILK metadata, and supplement columns.
- `images/`: JPEG images organized by lesion id.
- `original_csvs/`: Original MILK10k CSV files used to build this upload.

## Counts

Total images: 10480

Labels:

- AKIEC: 606
- BCC: 5044
- BEN_OTH: 88
- BKL: 1088
- DF: 104
- INF: 100
- MAL_OTH: 18
- MEL: 900
- NV: 1492
- SCCKA: 946
- VASC: 94

Image types:

- clinical_close_up: 5240
- dermoscopic: 5240

## License

The source CSV metadata lists `CC-BY-NC` in `copyright_license`. Confirm the exact upstream license and usage constraints before publishing publicly or using commercially.
