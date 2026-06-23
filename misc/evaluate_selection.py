#!/usr/bin/env python3

import json
from pathlib import Path
import pandas as pd
import numpy as np

def calculate_mean_confidence(preds_df, true_label_col='true_label', prob_cols_prefix='prob_'):
    # Assumes true_label is the string name of the class
    confidences = {}
    for class_name in preds_df[true_label_col].unique():
        class_df = preds_df[preds_df[true_label_col] == class_name]
        correct_df = class_df[class_df['pred_label'] == class_name]
        if len(correct_df) > 0:
            confidences[class_name] = correct_df[f'{prob_cols_prefix}{class_name}'].mean()
        else:
            confidences[class_name] = 0.0
    return confidences

def main():
    root_dir = Path("dual_encoder_ablation_runs")
    runs = [
        "run_baseline_ema",
        "run_tau025_sampler_ema",
        "run_tau050_sampler_ema",
        "run_tau025_nosampler_ema"
    ]
    
    results = {}
    baseline_stats = None
    
    for run in runs:
        run_dir = root_dir / run
        metrics_path = run_dir / "metrics.json"
        preds_path = run_dir / "predictions.csv"
        
        if not metrics_path.exists():
            print(f"Skipping {run}: metrics.json not found")
            continue
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        stats = {
            "f1_macro": metrics.get("f1_macro", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
            "global_temperature": metrics.get("global_temperature", 1.0)
        }
        
        # Load per-class metrics if available, or extract from main metrics
        # We look for recall_INF, recall_DF, recall_MEL, recall_SCCKA
        for cls in ["INF", "DF", "MEL", "SCCKA"]:
            stats[f"recall_{cls}"] = metrics.get(f"recall_{cls}", 0.0)
            
        # Load predictions to compute mean confidence of head classes on correct predictions
        if preds_path.exists():
            try:
                preds_df = pd.read_csv(preds_path)
                confidences = calculate_mean_confidence(preds_df)
                # Head classes are usually NV, MEL, BCC, etc. Assuming NV and MEL are heads.
                stats["conf_NV"] = confidences.get("NV", 0.0)
                stats["conf_BCC"] = confidences.get("BCC", 0.0)
            except Exception as e:
                print(f"Warning: could not process predictions for {run}: {e}")
                
        results[run] = stats
        if run == "run_baseline_ema":
            baseline_stats = stats
            
    if not baseline_stats:
        print("Baseline run not found. Cannot perform relative comparisons.")
        return
        
    print(f"{'Run':<30} | {'F1':<6} | {'Acc':<6} | {'INF Rec':<7} | {'DF Rec':<6} | {'MEL Rec':<7} | {'Temp':<6}")
    print("-" * 80)
    for run, stats in results.items():
        print(f"{run:<30} | {stats['f1_macro']:.4f} | {stats['accuracy']:.4f} | {stats['recall_INF']:.4f}  | {stats['recall_DF']:.4f} | {stats['recall_MEL']:.4f}  | {stats['global_temperature']:.2f}")

    print("\n--- Criteria Check ---")
    for run, stats in results.items():
        if run == "run_baseline_ema":
            continue
            
        print(f"\nEvaluating {run}:")
        
        f1_diff = stats['f1_macro'] - baseline_stats['f1_macro']
        f1_pass = f1_diff >= -0.005
        print(f" - Macro-F1 >= baseline - 0.005: {'PASS' if f1_pass else 'FAIL'} (Diff: {f1_diff:+.4f})")
        
        inf_rec = stats['recall_INF']
        inf_pass = inf_rec >= 0.30
        print(f" - INF recall >= 0.30: {'PASS' if inf_pass else 'FAIL'} ({inf_rec:.4f})")
        
        df_rec = stats['recall_DF']
        df_pass = df_rec >= 0.60
        print(f" - DF recall >= 0.60: {'PASS' if df_pass else 'FAIL'} ({df_rec:.4f})")
        
        mel_diff = stats['recall_MEL'] - baseline_stats['recall_MEL']
        sccka_diff = stats['recall_SCCKA'] - baseline_stats['recall_SCCKA']
        mel_pass = mel_diff >= -0.02
        sccka_pass = sccka_diff >= -0.02
        print(f" - MEL recall giảm <= 0.02: {'PASS' if mel_pass else 'FAIL'} (Diff: {mel_diff:+.4f})")
        print(f" - SCCKA recall giảm <= 0.02: {'PASS' if sccka_pass else 'FAIL'} (Diff: {sccka_diff:+.4f})")
        
        acc_diff = stats['accuracy'] - baseline_stats['accuracy']
        acc_pass = acc_diff >= -0.015
        print(f" - Accuracy giảm <= 0.015: {'PASS' if acc_pass else 'FAIL'} (Diff: {acc_diff:+.4f})")
        
        if "conf_NV" in stats and "conf_NV" in baseline_stats:
            conf_nv_diff = stats["conf_NV"] - baseline_stats["conf_NV"]
            conf_pass = conf_nv_diff >= -0.05
            print(f" - Mean confidence NV giảm <= 0.05: {'PASS' if conf_pass else 'FAIL'} (Diff: {conf_nv_diff:+.4f})")

if __name__ == "__main__":
    main()
