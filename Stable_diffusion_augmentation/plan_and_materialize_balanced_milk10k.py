#!/usr/bin/env python3
"""Audit MILK10k imbalance, plan paired SD augmentation, and optionally materialize a capped dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ModuleNotFoundError:  # Matplotlib fallback keeps audit usable in minimal environments.
    sns = None

LABEL_COLUMNS = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
MODALITIES = {"clinical: close-up", "dermoscopic"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Audit and materialize a safely balanced paired MILK10k dataset.")
    p.add_argument("--base-data-dir", type=Path, required=True)
    p.add_argument("--base-input-dir", type=Path, default=None)
    p.add_argument("--augmented-groundtruth", type=Path, default=None)
    p.add_argument("--augmented-metadata", type=Path, default=None)
    p.add_argument("--synthetic-input-dir", type=Path, default=None)
    p.add_argument("--qc-summary", type=Path, default=None)
    p.add_argument("--fresh-start", action="store_true", help="Ignore every existing synthetic CSV and plan from base real data only.")
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--scripts-dir", type=Path, default=None, help="Command output folder; defaults to Stable_diffusion_augmentation/.")
    p.add_argument("--materialize-dir", type=Path, default=None)
    p.add_argument("--bcc-cap-ratio", type=float, default=1.5)
    p.add_argument("--tail-floor", type=int, default=150)
    p.add_argument("--max-synthetic-real-ratio", type=float, default=2.0)
    p.add_argument("--max-synthetic-per-source", type=int, default=3)
    p.add_argument("--min-target-prob", type=float, default=0.4)
    p.add_argument("--require-target-pred", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-variants", type=int, default=1)
    p.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def resolve_paths(args):
    base = args.base_data_dir.expanduser().resolve()
    gt = base / "MILK10k_Training_GroundTruth.csv"
    meta = base / "MILK10k_Training_Metadata.csv"
    input_dir = (args.base_input_dir or base / "MILK10k_Training_Input").expanduser().resolve()
    if not input_dir.exists() and args.base_input_dir is None:
        input_dir = (base.parent / "MILK10k_Training_Input").resolve()
    info = base / "augmented_info"
    aug_gt = args.augmented_groundtruth or info / "MILK10k_Training_GroundTruth(2).csv"
    aug_meta = args.augmented_metadata or info / "MILK10k_Training_Metadata(3).csv"
    required = [gt, meta, input_dir] + ([] if args.fresh_start else [aug_gt, aug_meta])
    missing = [str(path) for path in required if not path.exists()]
    if missing: raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    return gt, meta, input_dir, aug_gt.expanduser().resolve(), aug_meta.expanduser().resolve()


def attach_labels(gt):
    missing = set(LABEL_COLUMNS) - set(gt.columns)
    if missing: raise ValueError(f"Ground truth missing labels: {sorted(missing)}")
    result = gt.copy(); result["label"] = result[LABEL_COLUMNS].idxmax(axis=1)
    if result.lesion_id.duplicated().any(): raise ValueError("Duplicate lesion_id in ground truth.")
    return result


def source_id(lesion_id): return str(lesion_id).split("__sdpair_", 1)[0]


def load_inventory(args):
    gt_path, meta_path, input_dir, aug_gt_path, aug_meta_path = resolve_paths(args)
    base_gt_raw = pd.read_csv(gt_path); base_meta = pd.read_csv(meta_path); base_gt = attach_labels(base_gt_raw)
    base_ids = set(base_gt.lesion_id.astype(str))
    if args.fresh_start:
        synth_gt = pd.DataFrame(columns=[*base_gt.columns, "source_lesion_id", "pair_metadata_complete"])
        synth_meta = pd.DataFrame(columns=base_meta.columns)
        return base_gt_raw, base_gt, base_meta, synth_gt, synth_meta, input_dir
    aug_gt_raw = pd.read_csv(aug_gt_path); aug_meta = pd.read_csv(aug_meta_path); aug_gt = attach_labels(aug_gt_raw)
    synth_gt = aug_gt[~aug_gt.lesion_id.astype(str).isin(base_ids)].copy()
    synth_meta = aug_meta[aug_meta.lesion_id.astype(str).isin(set(synth_gt.lesion_id.astype(str)))].copy()
    if base_ids - set(aug_gt.lesion_id.astype(str)): raise ValueError("Augmented ground truth omits base lesions.")
    if synth_gt.lesion_id.duplicated().any(): raise ValueError("Duplicate synthetic lesion IDs.")
    synth_gt["source_lesion_id"] = synth_gt.lesion_id.map(source_id)
    modality_counts = synth_meta.groupby("lesion_id").image_type.agg(lambda values: set(map(str, values)))
    synth_gt["pair_metadata_complete"] = synth_gt.lesion_id.map(lambda x: modality_counts.get(x, set()) == MODALITIES)
    return base_gt_raw, base_gt, base_meta, synth_gt, synth_meta, input_dir


def add_file_and_qc_status(args, synth, synth_meta):
    result = synth.copy(); qc = None
    if args.qc_summary:
        qc = pd.read_csv(args.qc_summary.expanduser().resolve()).drop_duplicates("synthetic_lesion_id").set_index("synthetic_lesion_id")
    if args.synthetic_input_dir:
        image_root = args.synthetic_input_dir.expanduser().resolve()
        paths = synth_meta.assign(path=synth_meta.apply(lambda row: image_root / str(row.lesion_id) / f"{row.isic_id}.jpg", axis=1))
        file_counts = paths.groupby("lesion_id").path.agg(lambda values: sum(Path(x).exists() for x in values))
        result["existing_image_files"] = result.lesion_id.map(lambda x: int(file_counts.get(x, 0)))
        result["pair_files_complete"] = result.lesion_id.map(lambda x: file_counts.get(x, 0) == 2)
    else:
        result["existing_image_files"] = 0
        result["pair_files_complete"] = False
    result["qc_available"] = result.lesion_id.isin(set(qc.index)) if qc is not None else False
    result["target_probability"] = result.lesion_id.map(qc.target_class_probability) if qc is not None and "target_class_probability" in qc else np.nan
    result["is_target_predicted"] = result.lesion_id.map(qc.is_target_predicted).astype(str).eq("True") if qc is not None and "is_target_predicted" in qc else False
    result["qc_pass"] = result.qc_available & (pd.to_numeric(result.target_probability, errors="coerce").fillna(0) >= args.min_target_prob)
    if args.require_target_pred: result["qc_pass"] &= result.is_target_predicted
    result["usable"] = result.pair_metadata_complete & result.pair_files_complete & result.qc_pass
    return result


def capped_synthetic(synth, real_counts, args, usable_only):
    candidates = synth[synth.usable].copy() if usable_only else synth.copy()
    candidates = candidates.sort_values(["label", "is_target_predicted", "target_probability", "lesion_id"], ascending=[True, False, False, True])
    candidates["source_rank"] = candidates.groupby(["label", "source_lesion_id"]).cumcount() + 1
    candidates = candidates[candidates.source_rank <= args.max_synthetic_per_source]
    selected = []
    for label, group in candidates.groupby("label", sort=True):
        cap = int(math.floor(real_counts.get(label, 0) * args.max_synthetic_real_ratio))
        selected.append(group.head(cap))
    return pd.concat(selected, ignore_index=True) if selected else candidates.iloc[:0]


def plan_counts(base, synth, inventory_selected, usable_selected, args):
    real = base.label.value_counts().reindex(LABEL_COLUMNS, fill_value=0).astype(int)
    raw = synth.label.value_counts().reindex(LABEL_COLUMNS, fill_value=0).astype(int)
    inventory = inventory_selected.label.value_counts().reindex(LABEL_COLUMNS, fill_value=0).astype(int)
    usable = usable_selected.label.value_counts().reindex(LABEL_COLUMNS, fill_value=0).astype(int)
    second = int(real.drop("BCC").max()); bcc_cap = min(int(real.BCC), int(math.floor(second * args.bcc_cap_ratio)))
    rows = []
    for label in LABEL_COLUMNS:
        target = int(real[label])
        if label in {"BEN_OTH", "DF", "INF", "MAL_OTH", "VASC"}:
            target = min(args.tail_floor, int(math.floor(real[label] * (1 + args.max_synthetic_real_ratio))))
        kept_real = bcc_cap if label == "BCC" else int(real[label])
        rows.append({"class": label, "real_count": int(real[label]), "raw_synthetic_inventory": int(raw[label]),
                     "capped_inventory": int(inventory[label]), "verified_usable": int(usable[label]), "target_total": target,
                     "kept_real": kept_real, "final_inventory_total": kept_real + int(inventory[label]),
                     "final_verified_total": kept_real + int(usable[label]),
                     "additional_needed_inventory": max(0, target - int(real[label]) - int(inventory[label])),
                     "additional_needed_verified": max(0, target - int(real[label]) - int(usable[label]))})
    return pd.DataFrame(rows), bcc_cap


def bcc_strata(base_meta, base, cap, seed):
    bcc = base[base.label.eq("BCC")][["lesion_id"]].merge(base_meta.drop_duplicates("lesion_id"), on="lesion_id", how="left")
    age = pd.to_numeric(bcc.age_approx, errors="coerce"); bcc["age_bin"] = pd.cut(age, [-np.inf,29,49,69,np.inf], labels=["<30","30-49","50-69","70+"]).astype(str)
    for column in ("sex", "site", "skin_tone_class", "age_bin"): bcc[column] = bcc[column].fillna("unknown").astype(str)
    bcc["stratum"] = bcc[["sex","site","skin_tone_class","age_bin"]].agg("|".join, axis=1)
    counts = bcc.stratum.value_counts().sort_index(); exact = counts / len(bcc) * cap; quota = np.floor(exact).astype(int)
    for key in (exact-quota).sort_values(ascending=False).index:
        if quota.sum() >= cap: break
        if quota[key] < counts[key]: quota[key] += 1
    chosen=[]; rng=np.random.default_rng(seed)
    for key in counts.index:
        ids=bcc.loc[bcc.stratum.eq(key),"lesion_id"].to_numpy();rng.shuffle(ids);chosen.extend(ids[:quota[key]])
    if len(chosen)<cap:
        remaining=np.array(sorted(set(bcc.lesion_id)-set(chosen)));rng.shuffle(remaining);chosen.extend(remaining[:cap-len(chosen)])
    return set(map(str, chosen[:cap])), bcc


def drift_table(bcc, selected_ids):
    rows=[]
    for column in ("sex","site","skin_tone_class","age_bin"):
        full=bcc[column].value_counts(normalize=True); kept=bcc[bcc.lesion_id.astype(str).isin(selected_ids)][column].value_counts(normalize=True)
        for value in sorted(set(full.index)|set(kept.index)): rows.append({"field":column,"value":value,"full_ratio":float(full.get(value,0)),"kept_ratio":float(kept.get(value,0)),"absolute_drift":abs(float(full.get(value,0)-kept.get(value,0)))})
    return pd.DataFrame(rows)


def selection_manifest(synth, inventory_ids, usable_ids):
    result=synth.copy();result["selected_inventory"]=result.lesion_id.isin(inventory_ids);result["selected_usable"]=result.lesion_id.isin(usable_ids)
    def reason(row):
        if not row.pair_metadata_complete:return "incomplete_metadata_pair"
        if not row.pair_files_complete:return "missing_image_pair"
        if not row.qc_available:return "qc_missing"
        if not row.qc_pass:return "qc_failed"
        if not row.selected_usable:return "class_or_source_cap"
        return "selected"
    result["selection_reason"]=result.apply(reason,axis=1)
    return result


def source_diversity(synth, selected):
    rows=[]
    for label in LABEL_COLUMNS:
        group=synth[synth.label.eq(label)]; kept=selected[selected.label.eq(label)]
        rows.append({"class":label,"inventory":len(group),"inventory_sources":group.source_lesion_id.nunique(),
                     "selected":len(kept),"selected_sources":kept.source_lesion_id.nunique(),
                     "max_selected_per_source":int(kept.groupby("source_lesion_id").size().max()) if len(kept) else 0})
    return pd.DataFrame(rows)


def plot_reports(report_dir, distribution, sources):
    if sns is not None:sns.set_theme(style="whitegrid")
    def grouped(frame,x,y,hue,path,log=False):
        plt.figure(figsize=(14,6))
        if sns is not None:sns.barplot(data=frame,x=x,y=y,hue=hue)
        else:
            pivot=frame.pivot(index=x,columns=hue,values=y);pivot.plot(kind="bar",ax=plt.gca())
        if log:plt.yscale("log")
        plt.xticks(rotation=35);plt.tight_layout();plt.savefig(path,dpi=170);plt.close()
    long=distribution.melt(id_vars="class",value_vars=["real_count","raw_synthetic_inventory","capped_inventory","verified_usable"],var_name="series",value_name="count")
    grouped(long,"class","count","series",report_dir/"class_distribution.png")
    final=distribution.melt(id_vars="class",value_vars=["real_count","final_inventory_total","final_verified_total","target_total"],var_name="series",value_name="count")
    grouped(final,"class","count","series",report_dir/"balance_before_after.png",True)
    ratio=distribution.assign(synthetic_real_ratio=lambda d:d.capped_inventory/d.real_count.clip(lower=1))
    plt.figure(figsize=(12,5));
    if sns is not None:sns.barplot(data=ratio,x="class",y="synthetic_real_ratio")
    else:plt.bar(ratio["class"],ratio.synthetic_real_ratio)
    plt.axhline(2,color="red",linestyle="--");plt.xticks(rotation=35);plt.tight_layout();plt.savefig(report_dir/"synthetic_real_ratio.png",dpi=170);plt.close()
    plt.figure(figsize=(12,5));
    if sns is not None:sns.barplot(data=sources,x="class",y="selected_sources")
    else:plt.bar(sources["class"],sources.selected_sources)
    plt.xticks(rotation=35);plt.tight_layout();plt.savefig(report_dir/"source_diversity.png",dpi=170);plt.close()


def transfer(src, dst, mode, overwrite):
    if not src.exists(): raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:return
        dst.unlink()
    if mode=="copy":shutil.copy2(src,dst)
    elif mode=="symlink":dst.symlink_to(src.resolve())
    else:
        try:os.link(src,dst)
        except OSError:shutil.copy2(src,dst)


def materialize_variant(destination, base_gt_raw, base_meta, selected_base_ids, synth_gt, synth_meta, selected_synth_ids, base_input, synth_input, args):
    if synth_input is None: raise ValueError("--synthetic-input-dir is required for materialization.")
    if destination.exists() and not args.overwrite and any(destination.iterdir()):raise FileExistsError(f"Output is not empty: {destination}")
    destination.mkdir(parents=True,exist_ok=True);output_images=destination/"MILK10k_Training_Input"
    kept_gt=base_gt_raw[base_gt_raw.lesion_id.astype(str).isin(selected_base_ids)].copy()
    kept_meta=base_meta[base_meta.lesion_id.astype(str).isin(selected_base_ids)].copy()
    add_gt=synth_gt[synth_gt.lesion_id.astype(str).isin(selected_synth_ids)][["lesion_id",*LABEL_COLUMNS]].copy()
    add_meta=synth_meta[synth_meta.lesion_id.astype(str).isin(selected_synth_ids)].copy()
    counts=add_meta.groupby("lesion_id").image_type.agg(lambda x:set(map(str,x)))
    bad=[x for x in selected_synth_ids if counts.get(x,set())!=MODALITIES]
    if bad:raise ValueError(f"Incomplete synthetic pairs: {bad[:5]}")
    combined_meta=pd.concat([kept_meta.assign(source_root=str(base_input)),add_meta.assign(source_root=str(synth_input))])
    for _,row in combined_meta.iterrows():
        src=Path(row.source_root)/str(row.lesion_id)/f"{row.isic_id}.jpg";dst=output_images/str(row.lesion_id)/f"{row.isic_id}.jpg";transfer(src,dst,args.link_mode,args.overwrite)
    pd.concat([kept_gt,add_gt],ignore_index=True).to_csv(destination/"MILK10k_Training_GroundTruth.csv",index=False)
    pd.concat([kept_meta,add_meta],ignore_index=True).to_csv(destination/"MILK10k_Training_Metadata.csv",index=False)


def command_script(args, plan, report_dir):
    needed_rows=plan[(plan.additional_needed_verified>0)&~plan["class"].eq("MAL_OTH")]
    classes=" ".join(needed_rows["class"].astype(str));per_source=args.max_synthetic_per_source
    max_sources=max([math.ceil(int(value)/per_source) for value in needed_rows.additional_needed_verified] or [1])
    base_data=args.base_data_dir.expanduser().resolve();base_input=(args.base_input_dir.expanduser().resolve() if args.base_input_dir else base_data/"MILK10k_Training_Input")
    if not base_input.exists():base_input=base_data.parent/"MILK10k_Training_Input"
    header=["#!/usr/bin/env bash","set -euo pipefail","","# Generated commands. Review paths before running.",
            f'BASE_DATA="{base_data}"',f'BASE_INPUT="{base_input}"',f'REPORT_DIR="{report_dir}"','CHECKPOINT_DIR="/path/to/convnext_5fold_run"',
            'GEN_DIR="$REPORT_DIR/generated_balance_pairs"','CANDIDATE_DIR="$REPORT_DIR/candidate_augmented"','FINAL_DIR="/path/to/milk10k_balanced"',""]
    generate=[]
    if classes:
        generate += [f"# Planned classes: {classes}; planner applies exact caps after QC.",
                     f"python Stable_diffusion_augmentation/generate_milk10k_sd_pairs.py --data-dir \"$BASE_DATA\" --input-dir \"$BASE_INPUT\" --output-dir \"$GEN_DIR\" --class-names {classes} --num-per-lesion {per_source} --max-source-lesions {max_sources} --shuffle --skip-existing",""]
    qc_materialize=["# QC outputs: $GEN_DIR/effb2_qc_predictions.csv and effb2_qc_summary.csv.",
              "python Stable_diffusion_augmentation/run_effb2_qc.py --checkpoint-dir \"$CHECKPOINT_DIR\" --output-dir \"$GEN_DIR\"", "",
              "python Stable_diffusion_augmentation/filter_paired_augmentation_by_qc.py --manifest \"$GEN_DIR/paired_augmentation_manifest.csv\" --qc-summary \"$GEN_DIR/effb2_qc_summary.csv\" --output \"$GEN_DIR/filtered_manifest.csv\" --min-target-prob 0.4 --require-target-pred", "",
              "# Build a temporary complete MILK10k dataset from QC-passed pairs.",
              "python Stable_diffusion_augmentation/materialize_augmented_milk10k_dataset.py --input-dir \"$BASE_INPUT\" --metadata-csv \"$BASE_DATA/MILK10k_Training_Metadata.csv\" --groundtruth-csv \"$BASE_DATA/MILK10k_Training_GroundTruth.csv\" --augmentation-manifest \"$GEN_DIR/filtered_manifest.csv\" --output-dir \"$CANDIDATE_DIR\" --symlink --synthetic-metadata neutral --overwrite", "",
              "# Apply BCC stratified cap and final source/class caps into a separate dataset.",
              "python Stable_diffusion_augmentation/plan_and_materialize_balanced_milk10k.py --base-data-dir \"$BASE_DATA\" --augmented-groundtruth \"$CANDIDATE_DIR/MILK10k_Training_GroundTruth.csv\" --augmented-metadata \"$CANDIDATE_DIR/MILK10k_Training_Metadata.csv\" --synthetic-input-dir \"$CANDIDATE_DIR/MILK10k_Training_Input\" --qc-summary \"$GEN_DIR/effb2_qc_summary.csv\" --report-dir \"$REPORT_DIR/final_audit\" --materialize-dir \"$FINAL_DIR\" --require-target-pred --overwrite", "",
              "# Train safely: synthetic IDs stay train-only.",
              "# python milk10k_effb2_dermoscopic_metadata/train_milk10k_effb2_dermoscopic_metadata.py --data-dir \"$FINAL_DIR\" --output-dir /path/to/run --split-manifest /path/to/run/split_v2.json --synthetic-train-only --metadata-mode none --loss ldam", ""]
    code_dir=(args.scripts_dir.expanduser().resolve() if args.scripts_dir else Path(__file__).resolve().parent)
    code_dir.mkdir(parents=True,exist_ok=True)
    exported={"run_fresh_balance_01_generate.sh":header+generate,
              "run_fresh_balance_02_qc_materialize.sh":header+qc_materialize}
    for name,lines in exported.items():
        path=code_dir/name;path.write_text("\n".join(lines),encoding="utf-8");path.chmod(0o755)


def report_markdown(args, plan, sources, manifest, bcc_cap, usable_count, missing_pairs, missing_image_files):
    warnings=[]
    if args.qc_summary is None:warnings.append("QC summary was not provided; no synthetic row is considered verified usable.")
    if args.synthetic_input_dir is None:warnings.append("Synthetic image root was not provided; materialization is disabled.")
    if missing_pairs:warnings.append(f"Synthetic images missing: {missing_image_files} files across {missing_pairs} incomplete inventory pairs.")
    warnings.append("MAL_OTH should use external/manual-reviewed data; do not scale SD from only a few source lesions.")
    lines=["# MILK10k Balance Report","",f"- BCC static cap: {bcc_cap}",f"- verified usable synthetic lesions: {usable_count}",f"- source cap: {args.max_synthetic_per_source}",f"- synthetic/real cap: {args.max_synthetic_real_ratio}x","","## Warnings",""]+[f"- {x}" for x in warnings]
    lines += ["","## Augmentation plan","","```",plan.to_string(index=False),"```","","## Source diversity","","```",sources.to_string(index=False),"```","","## Selection reasons","",manifest.selection_reason.value_counts().to_string(),""]
    return "\n".join(lines)


def run(args):
    if args.num_variants<1 or args.max_synthetic_per_source<1:raise ValueError("num variants and source cap must be positive.")
    report_dir=args.report_dir.expanduser().resolve();report_dir.mkdir(parents=True,exist_ok=True)
    base_gt_raw,base,base_meta,synth_gt,synth_meta,base_input=load_inventory(args)
    synth=add_file_and_qc_status(args,synth_gt,synth_meta);real_counts=base.label.value_counts().to_dict()
    inventory_selected=capped_synthetic(synth,real_counts,args,False);usable_selected=capped_synthetic(synth,real_counts,args,True)
    plan,bcc_cap=plan_counts(base,synth,inventory_selected,usable_selected,args)
    manifest=selection_manifest(synth,set(inventory_selected.lesion_id),set(usable_selected.lesion_id));sources=source_diversity(synth,inventory_selected)
    variants=[];drifts=[]
    materialize_error = None
    if args.materialize_dir and len(inventory_selected) and not len(usable_selected):
        materialize_error = ValueError(
            "Materialization refused: synthetic inventory exists but no pair passed image/QC validation. "
            "Check --synthetic-input-dir and --qc-summary; audit reports are still written."
        )
    for index in range(args.num_variants):
        ids,bcc=bcc_strata(base_meta,base,bcc_cap,args.seed+index);selected_base=set(base.loc[~base.label.eq("BCC"),"lesion_id"].astype(str))|ids
        variants.append(selected_base);drift=drift_table(bcc,ids);drift["variant"]=index;drifts.append(drift)
        base[["lesion_id","label"]].assign(selected=lambda frame:frame.lesion_id.astype(str).isin(selected_base),variant=index).to_csv(report_dir/f"selected_base_manifest_variant_{index:02d}.csv",index=False)
        if args.materialize_dir and materialize_error is None:
            root=args.materialize_dir.expanduser().resolve();destination=root if args.num_variants==1 else root/f"variant_{index:02d}"
            materialize_variant(destination,base_gt_raw,base_meta,selected_base,synth_gt,synth_meta,set(usable_selected.lesion_id),base_input,args.synthetic_input_dir.expanduser().resolve() if args.synthetic_input_dir else None,args)
    plan.to_csv(report_dir/"augmentation_plan.csv",index=False);sources.to_csv(report_dir/"source_diversity.csv",index=False);manifest.to_csv(report_dir/"selected_synthetic_manifest.csv",index=False)
    pd.concat(drifts,ignore_index=True).to_csv(report_dir/"bcc_distribution_drift.csv",index=False)
    distribution=plan.copy();distribution.to_csv(report_dir/"class_distribution.csv",index=False);plot_reports(report_dir,distribution,sources)
    missing_pairs=int((~synth.pair_files_complete).sum());missing_image_files=int((2-synth.existing_image_files.clip(upper=2)).sum())
    payload={"policy":vars(args),"bcc_cap":bcc_cap,"inventory_lesions":len(synth),"usable_synthetic_lesions":len(usable_selected),"missing_image_pairs":missing_pairs,"missing_image_files":missing_image_files,"plan":plan.to_dict("records")}
    payload["policy"]={k:str(v) if isinstance(v,Path) else v for k,v in payload["policy"].items()};(report_dir/"balance_plan.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    (report_dir/"balance_report.md").write_text(report_markdown(args,plan,sources,manifest,bcc_cap,len(usable_selected),missing_pairs,missing_image_files),encoding="utf-8")
    command_script(args,plan,report_dir)
    print(f"Base lesions: {len(base)}; synthetic inventory: {len(synth)}; verified usable: {len(usable_selected)}")
    print(f"BCC cap: {bcc_cap}; reports: {report_dir}")
    if materialize_error is not None:raise materialize_error
    if args.materialize_dir:print(f"Materialized dataset: {args.materialize_dir.expanduser().resolve()}")
    return {"plan":plan,"manifest":manifest,"bcc_cap":bcc_cap,"usable":usable_selected}


def main():run(parse_args())
if __name__=="__main__":main()
