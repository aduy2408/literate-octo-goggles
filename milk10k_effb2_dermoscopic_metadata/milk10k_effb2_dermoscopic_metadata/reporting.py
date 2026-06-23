"""Structured data, prediction, confusion, environment, and k-fold reports."""

from __future__ import annotations

import json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .core import json_safe
from .data import synthetic_mask


def environment_info():
    import torch
    def git(*args):
        try: return subprocess.run(["git",*args],capture_output=True,text=True,timeout=5).stdout.strip()
        except Exception: return None
    return {"timestamp_utc":datetime.now(timezone.utc).isoformat(),"cwd":str(Path.cwd()),"command":sys.argv,
            "python":sys.version.replace("\n"," "),"platform":platform.platform(),"executable":sys.executable,
            "torch":{"version":torch.__version__,"cuda_available":torch.cuda.is_available(),
                     "device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "git":{"commit":git("rev-parse","HEAD"),"branch":git("rev-parse","--abbrev-ref","HEAD"),
                   "status_short":(git("status","--short") or "").splitlines()}}


def distribution(df, class_names):
    synth = synthetic_mask(df); ignored = df.get("ignore_metadata",pd.Series(False,index=df.index)).fillna(False).astype(bool).to_numpy()
    return {"rows":len(df),"real_rows":int((~synth).sum()),"synthetic_rows":int(synth.sum()),
            "ignore_metadata_rows":int(ignored.sum()),
            "class_counts":df.label.value_counts().reindex(class_names,fill_value=0).astype(int).to_dict(),
            "synthetic_class_counts":df.loc[synth,"label"].value_counts().reindex(class_names,fill_value=0).astype(int).to_dict()}


def save_data_summary(output_dir, full_df, train_df, val_df, class_names):
    payload={"full":distribution(full_df,class_names),"train":distribution(train_df,class_names),"val":distribution(val_df,class_names),
             "synthetic_train_only":bool(synthetic_mask(train_df).sum() and not synthetic_mask(val_df).sum())}
    (output_dir/"data_summary.json").write_text(json.dumps(json_safe(payload),indent=2),encoding="utf-8")
    lines=["# Split Summary",""]
    for split in ("full","train","val"):
        item=payload[split]; lines += [f"## {split.title()}","",f"- rows: {item['rows']}",f"- real: {item['real_rows']}",f"- synthetic: {item['synthetic_rows']}","","| class | count | synthetic |","|---|---:|---:|"]
        lines += [f"| {name} | {count} | {item['synthetic_class_counts'][name]} |" for name,count in item["class_counts"].items()]; lines.append("")
    (output_dir/"split_summary.md").write_text("\n".join(lines),encoding="utf-8")
    return payload


def prediction_summary(prob, class_names):
    pred=prob.argmax(1); sorted_prob=np.sort(prob,axis=1); confidence=sorted_prob[:,-1]; second=sorted_prob[:,-2]
    entropy=-np.sum(prob*np.log(np.clip(prob,1e-12,1)),axis=1); counts=np.bincount(pred,minlength=len(class_names))
    return {"rows":len(prob),"predicted_class_counts":{n:int(counts[i]) for i,n in enumerate(class_names)},
            "mean_probability":{n:float(prob[:,i].mean()) for i,n in enumerate(class_names)},
            "mean_confidence":float(confidence.mean()),"median_confidence":float(np.median(confidence)),
            "mean_top1_top2_gap":float((confidence-second).mean()),"mean_entropy":float(entropy.mean()),
            "low_confidence_rows":int((confidence<.5).sum())}


def confusion_analysis(cm,class_names):
    pairs=[]
    for i,true in enumerate(class_names):
        total=int(cm[i].sum())
        for j,pred in enumerate(class_names):
            if i!=j and cm[i,j]: pairs.append({"true":true,"predicted":pred,"count":int(cm[i,j]),"rate_of_true":float(cm[i,j]/total) if total else 0})
    return {"top_confusion_pairs":sorted(pairs,key=lambda x:x["count"],reverse=True)[:20]}


def save_diagnostics(output_dir,args,data_summary,metrics,per_class,cm,prob,class_names,fold=None):
    pred=prediction_summary(prob,class_names); conf=confusion_analysis(cm,class_names); warnings=[]
    for row in per_class.to_dict("records"):
        if row["support"]<=5: warnings.append({"severity":"medium","code":"tiny_validation_support","class":row["class"]})
        if row["recall_sensitivity"]==0: warnings.append({"severity":"high","code":"zero_recall","class":row["class"]})
    diag={"fold":fold,"warnings":warnings,"prediction_summary":pred,"confusion_analysis":conf}
    for name,value in (("prediction_summary",pred),("confusion_analysis",conf),("run_diagnostics",diag)):
        (output_dir/f"{name}.json").write_text(json.dumps(json_safe(value),indent=2),encoding="utf-8")
    lines=["# Dermoscopic Run Report","",f"- fold: {fold}",f"- backbone: {args.backbone}",f"- metadata_mode: {args.metadata_mode}",f"- loss: {args.loss}","","## Metrics",""]
    lines += [f"- {key}: {metrics.get(key)}" for key in ("accuracy","balanced_accuracy","f1_macro","dice_macro","roc_auc_macro_ovr","top3_accuracy")]
    lines += ["","## Per class","","```",per_class.to_string(index=False),"```","","## Top confusions",""]
    lines += [f"- {x['true']} -> {x['predicted']}: {x['count']} ({x['rate_of_true']:.1%})" for x in conf["top_confusion_pairs"][:12]]
    lines += ["","## Warnings",""]+[f"- [{x['severity']}] {x['code']}: {x.get('class','')}" for x in warnings]
    (output_dir/"run_report.md").write_text("\n".join(lines),encoding="utf-8")
    return diag


def save_kfold_summary(metrics_list,output_dir):
    keys=("best_selection_metric","best_val_tail_recall_macro","accuracy","balanced_accuracy","f1_macro","f1_weighted","dice_macro","roc_auc_macro_ovr","top3_accuracy")
    rows=[{"fold":i,**{k:m.get(k) for k in keys}} for i,m in enumerate(metrics_list)]
    frame=pd.DataFrame(rows); frame.to_csv(output_dir/"kfold_summary.csv",index=False)
    payload={"folds":rows,"mean":{},"std":{}}
    for key in keys:
        values=pd.to_numeric(frame[key],errors="coerce").dropna(); payload["mean"][key]=None if values.empty else float(values.mean()); payload["std"][key]=None if values.empty else float(values.std(ddof=0))
    (output_dir/"kfold_summary.json").write_text(json.dumps(json_safe(payload),indent=2),encoding="utf-8")
    (output_dir/"kfold_report.md").write_text("# K-fold Summary\n\n```\n"+frame.to_string(index=False)+"\n```\n",encoding="utf-8")
    return payload
