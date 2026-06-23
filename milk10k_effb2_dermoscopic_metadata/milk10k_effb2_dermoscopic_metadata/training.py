"""Training orchestration for single-dermoscopic-image metadata models."""

from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from .core import apply_class_bias, compute_metrics, json_safe, optimize_class_bias, save_evaluation, set_seed
from .data import (append_augmented_rows, create_or_load_split, fit_metadata_spec, load_dermoscopic_dataframe,
                   make_loaders, metadata_vector, synthetic_mask)
from .losses import build_loss
from .model import DermoscopicMetadataClassifier, METADATA_MODES, set_encoder_trainable, set_metadata_trainable
from .reporting import environment_info, save_data_summary, save_diagnostics, save_kfold_summary

CHECKPOINT_SCHEMA_VERSION = 2


def parse_args(argv=None):
    p=argparse.ArgumentParser(description="Train a dermoscopic-only classifier with optional metadata.")
    p.add_argument("--data-dir",type=Path,required=True); p.add_argument("--input-dir",type=Path,default=None)
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--split-manifest",type=Path,required=True)
    p.add_argument("--metadata-mode",choices=METADATA_MODES,default="none"); p.add_argument("--backbone",default="efficientnet_b2")
    p.add_argument("--backbone-backend",choices=["auto","timm","torchvision"],default="auto")
    p.add_argument("--encoder-checkpoint",type=Path,default=None); p.add_argument("--resume-checkpoint",type=Path,default=None)
    p.add_argument("--imagenet-pretrained",action="store_true")
    p.add_argument("--image-size",type=int,default=260); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--freeze-epochs",type=int,default=5); p.add_argument("--finetune-epochs",type=int,default=20)
    p.add_argument("--head-lr",type=float,default=1e-4); p.add_argument("--encoder-lr",type=float,default=1e-5); p.add_argument("--metadata-lr",type=float,default=None)
    p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--branch-dim",type=int,default=512); p.add_argument("--metadata-dim",type=int,default=64)
    p.add_argument("--metadata-gate-hidden-dim",type=int,default=None); p.add_argument("--freeze-metadata-head",action="store_true")
    p.add_argument("--classifier-hidden-dim",type=int,default=512); p.add_argument("--dropout",type=float,default=.3)
    p.add_argument("--loss",choices=["ce","focal","ldam","ce_dice","ce_f1"],default="ce")
    p.add_argument("--focal-gamma",type=float,default=2.0); p.add_argument("--dice-weight",type=float,default=.3); p.add_argument("--f1-weight",type=float,default=.3)
    p.add_argument("--f1-ignore-classes",nargs="*",default=[]); p.add_argument("--f1-class-weight",action="append",default=[])
    p.add_argument("--ldam-beta",type=float,default=.9999); p.add_argument("--ldam-max-margin",type=float,default=.5)
    p.add_argument("--ldam-drw-start-epoch",type=int,default=0); p.add_argument("--ldam-alpha-max",type=float,default=10.0); p.add_argument("--tail-num-classes",type=int,default=4)
    p.add_argument("--class-weight",action="store_true"); p.add_argument("--weighted-sampler",action="store_true"); p.add_argument("--sampler-power",type=float,default=1.0)
    p.add_argument("--synthetic-train-only",action="store_true"); p.add_argument("--augmented-data-dir",type=Path,default=None)
    p.add_argument("--augmented-max-per-class",type=int,default=0); p.add_argument("--augmented-classes",nargs="*",default=[]); p.add_argument("--zero-augmented-metadata",action="store_true")
    p.add_argument("--k-folds",type=int,default=1); p.add_argument("--val-size",type=float,default=.2); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--amp",action="store_true"); p.add_argument("--patience",type=int,default=6)
    p.add_argument("--selection-metric",choices=["f1_macro","dice_macro"],default="f1_macro")
    p.add_argument("--calibrate-bias",action="store_true"); p.add_argument("--calibration-metric",choices=["f1_macro","dice_macro"],default="dice_macro")
    p.add_argument("--calibration-max-bias",type=float,default=1.5); p.add_argument("--calibration-step",type=float,default=.25); p.add_argument("--calibration-passes",type=int,default=3)
    return p.parse_args(argv)


def extract_state(payload): return payload.get("model_state",payload.get("model_state_dict",payload.get("state_dict",payload)))


def infer_checkpoint_backend(path,device):
    keys=list(extract_state(torch.load(path.expanduser().resolve(),map_location=device,weights_only=False)))
    normalized=[]
    for key in keys:
        changed=True
        while changed:
            changed=False
            for prefix in ("module.","model.","_orig_mod.","encoder.","dermoscopic_encoder."):
                if key.startswith(prefix): key=key.removeprefix(prefix); changed=True; break
        normalized.append(key)
    timm=sum(k.startswith(("conv_stem.","blocks.","conv_head.","stages.","stem.")) for k in normalized)
    tv=sum(k.startswith(("features.","avgpool.","classifier.")) for k in normalized)
    if timm>tv:return "timm"
    if tv>timm:return "torchvision"
    raise RuntimeError(f"Cannot infer backend from {path}; pass --backbone-backend.")


def load_encoder_checkpoint(path,encoder,device):
    state=extract_state(torch.load(path.expanduser().resolve(),map_location=device,weights_only=False)); target=encoder.state_dict(); matched={}
    for raw,value in state.items():
        key=raw; changed=True
        while changed:
            changed=False
            for prefix in ("module.","_orig_mod.","model.","encoder.","dermoscopic_encoder.","backbone."):
                if key.startswith(prefix): key=key.removeprefix(prefix); changed=True; break
        if key in target and tuple(target[key].shape)==tuple(value.shape): matched[key]=value
    if not matched: raise RuntimeError(f"No compatible encoder weights in {path}")
    target.update(matched); encoder.load_state_dict(target); print(f"Loaded encoder keys={len(matched)}, skipped={len(state)-len(matched)}")


def make_optimizer(model,args,encoder_trainable):
    encoder=[]; metadata=[]; head=[]
    for name,param in model.named_parameters():
        if not param.requires_grad: continue
        if name.startswith("encoder."): encoder.append(param)
        elif name.startswith(("metadata_head.","metadata_gate.")): metadata.append(param)
        else: head.append(param)
    groups=[{"params":head,"lr":args.head_lr}]
    if metadata: groups.append({"params":metadata,"lr":args.metadata_lr or args.head_lr})
    if encoder_trainable and encoder: groups.append({"params":encoder,"lr":args.encoder_lr})
    return torch.optim.AdamW(groups,weight_decay=args.weight_decay)


def metric_key(name): return "".join(x if x.isalnum() else "_" for x in name)


def run_epoch(model,loader,criterion,device,optimizer=None,scaler=None,use_amp=False,class_names=None,tail_indices=None):
    training=optimizer is not None; model.train(training); total_loss=total=correct=top3=0; truths=[]; preds=[]
    for batch in tqdm(loader,leave=False):
        image=batch["image"].to(device,non_blocking=True); metadata=batch["metadata"].to(device,non_blocking=True); labels=batch["label"].to(device,non_blocking=True)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast(device.type,enabled=use_amp): logits=model(image,metadata); loss=criterion(logits,labels)
            if training:
                assert scaler is not None; scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update()
        size=labels.size(0); total_loss+=float(loss.detach())*size; total+=size; predicted=logits.argmax(1); correct+=int((predicted==labels).sum())
        top3+=int(logits.topk(min(3,logits.size(1)),1).indices.eq(labels[:,None]).any(1).sum()); truths.append(labels.cpu().numpy()); preds.append(predicted.cpu().numpy())
    y=np.concatenate(truths); pred=np.concatenate(preds); labels=list(range(len(class_names or [])))
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=labels,zero_division=0); cm=confusion_matrix(y,pred,labels=labels)
    stats={"loss":total_loss/max(total,1),"accuracy":correct/max(total,1),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
           "f1_macro":float(f1.mean()),"dice_macro":float(f1.mean()),"top3_accuracy":top3/max(total,1)}
    for i,name in enumerate(class_names or []):
        key=metric_key(name); stats.update({f"support_{key}":float(support[i]),f"precision_{key}":float(precision[i]),f"recall_{key}":float(recall[i]),f"f1_{key}":float(f1[i]),f"correct_{key}":float(cm[i,i])})
        row_total=int(cm[i].sum())
        for j,pred_name in enumerate(class_names or []):
            if i!=j and cm[i,j]: stats[f"conf_{key}_to_{metric_key(pred_name)}_count"]=float(cm[i,j]); stats[f"conf_{key}_to_{metric_key(pred_name)}_rate"]=float(cm[i,j]/row_total)
    if tail_indices: stats["tail_recall_macro"]=float(recall[tail_indices].mean())
    return stats


@torch.no_grad()
def predict(model,loader,device):
    model.eval(); truth=[]; probs=[]
    for batch in tqdm(loader,leave=False):
        logits=model(batch["image"].to(device),batch["metadata"].to(device)); truth.append(batch["label"].numpy()); probs.append(torch.softmax(logits,1).cpu().numpy())
    return np.concatenate(truth),np.concatenate(probs)


def checkpoint_payload(model,optimizer,scheduler,scaler,epoch,phase,best,best_tail,patience,class_names,label_to_idx,spec,args):
    return {"schema_version":CHECKPOINT_SCHEMA_VERSION,"epoch":epoch,"phase":phase,"best_val_f1_macro":best if args.selection_metric=="f1_macro" else None,
            "best_selection_metric":best,"selection_metric_name":args.selection_metric,"best_val_tail_recall_macro":best_tail,"patience_count":patience,
            "model_type":model.__class__.__name__,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),
            "scaler_state":scaler.state_dict(),"class_names":class_names,"label_to_idx":label_to_idx,"metadata_spec":spec,"args":json_safe(vars(args))}


def tail_config(train_df,class_names,label_to_idx,args):
    if args.loss!="ldam" or args.tail_num_classes<=0:return [],[]
    counts=train_df.label.value_counts().reindex(class_names,fill_value=0); names=sorted(class_names,key=lambda x:(counts[x],x))[:args.tail_num_classes]
    return names,[label_to_idx[x] for x in names]


def train_split(base_df,train_df,val_df,class_names,label_to_idx,args,device,backend,output_dir,fold=None):
    output_dir.mkdir(parents=True,exist_ok=True); split_dir=output_dir/"splits"; split_dir.mkdir(exist_ok=True)
    train_df=append_augmented_rows(base_df,train_df,args); train_df["is_augmented"]=synthetic_mask(train_df); train_df["ignore_metadata"]=train_df.get("ignore_metadata",False)
    val_df["is_augmented"]=synthetic_mask(val_df); val_df["ignore_metadata"]=False
    train_df.to_csv(split_dir/"train.csv",index=False); val_df.to_csv(split_dir/"val.csv",index=False)
    full_summary=pd.concat([train_df,val_df],ignore_index=True,sort=False); data_summary=save_data_summary(output_dir,full_summary,train_df,val_df,class_names)
    spec=fit_metadata_spec(train_df); metadata_input_dim=len(metadata_vector(train_df.iloc[0],spec)); train_loader,val_loader=make_loaders(train_df,val_df,label_to_idx,spec,args)
    model=DermoscopicMetadataClassifier(len(class_names),metadata_input_dim,args.metadata_mode,args.backbone,
        args.imagenet_pretrained or (args.encoder_checkpoint is None and args.resume_checkpoint is None),args.branch_dim,args.metadata_dim,
        args.classifier_hidden_dim,args.dropout,backend,args.metadata_gate_hidden_dim).to(device)
    if args.freeze_metadata_head:set_metadata_trainable(model,False)
    resume=None
    if args.resume_checkpoint:
        resume=torch.load(args.resume_checkpoint.expanduser().resolve(),map_location=device,weights_only=False); model.load_state_dict(resume["model_state"])
    elif args.encoder_checkpoint:load_encoder_checkpoint(args.encoder_checkpoint,model.encoder,device)
    config={"args":json_safe(vars(args)),"environment":environment_info(),"class_names":class_names,"label_to_idx":label_to_idx,"metadata_spec":spec,
            "metadata_input_dim":metadata_input_dim,"model_type":model.__class__.__name__,"backbone_backend_resolved":backend,"train_size":len(train_df),"val_size":len(val_df),"fold":fold}
    (output_dir/"run_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    criterion=build_loss(train_df,label_to_idx,args,device); tail_names,tail_indices=tail_config(train_df,class_names,label_to_idx,args)
    history=[]; history_path=output_dir/"history.csv"
    if resume is not None and history_path.exists():history=pd.read_csv(history_path).to_dict("records")
    best=float(resume.get("best_selection_metric",resume.get("best_val_f1_macro",float("-inf")))) if resume else float("-inf")
    best_tail=float(resume.get("best_val_tail_recall_macro",float("-inf"))) if resume else float("-inf")
    resume_epoch=int(resume.get("epoch",0)) if resume else 0; resume_phase=resume.get("phase") if resume else None
    phases=(("freeze",args.freeze_epochs,False,1),("finetune",args.finetune_epochs,True,args.freeze_epochs+1))
    for phase,count,encoder_trainable,start in phases:
        if count<=0:continue
        end=start+count-1
        if resume and ((resume_phase=="finetune" and phase=="freeze") or resume_epoch>=end):continue
        set_encoder_trainable(model,encoder_trainable); optimizer=make_optimizer(model,args,encoder_trainable)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.2,patience=2); scaler=GradScaler("cuda",enabled=args.amp and device.type=="cuda")
        patience=int(resume.get("patience_count",0)) if resume and resume_phase==phase else 0
        if resume and resume_phase==phase:
            try:
                optimizer.load_state_dict(resume["optimizer_state"])
                if "scheduler_state" in resume:scheduler.load_state_dict(resume["scheduler_state"])
                if "scaler_state" in resume:scaler.load_state_dict(resume["scaler_state"])
            except (ValueError,KeyError) as exc:print(f"Resume state fallback to recreated optimizer/scaler: {exc}")
        for epoch in range(max(start,resume_epoch+1),end+1):
            if hasattr(criterion,"set_epoch"):criterion.set_epoch(epoch)
            train_stats=run_epoch(model,train_loader,criterion,device,optimizer,scaler,args.amp and device.type=="cuda",class_names,tail_indices)
            val_stats=run_epoch(model,val_loader,criterion,device,None,None,args.amp and device.type=="cuda",class_names,tail_indices); scheduler.step(val_stats[args.selection_metric])
            history.append({"phase":phase,"epoch":epoch,**{f"train_{k}":v for k,v in train_stats.items()},**{f"val_{k}":v for k,v in val_stats.items()}}); pd.DataFrame(history).to_csv(history_path,index=False)
            improved=val_stats[args.selection_metric]>best
            if improved:best=val_stats[args.selection_metric];patience=0
            else:patience+=1
            tail_improved=bool(tail_indices and val_stats["tail_recall_macro"]>best_tail)
            if tail_improved:best_tail=val_stats["tail_recall_macro"]
            payload=checkpoint_payload(model,optimizer,scheduler,scaler,epoch,phase,best,best_tail,patience,class_names,label_to_idx,spec,args)
            if improved:torch.save(payload,output_dir/"best.pt")
            if tail_improved:payload.update({"tail_class_names":tail_names,"tail_class_indices":tail_indices});torch.save(payload,output_dir/"tail_best.pt")
            torch.save(payload,output_dir/"last.pt")
            print(f"{phase} epoch={epoch:03d} train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f} val_bal={val_stats['balanced_accuracy']:.4f} val_f1={val_stats['f1_macro']:.4f} val_top3={val_stats['top3_accuracy']:.4f}")
            if tail_indices:print(f"tail={tail_names} val_tail_recall={val_stats['tail_recall_macro']:.4f}")
            if args.patience>0 and patience>=args.patience:print(f"Early stopping {phase} at epoch {epoch}");break
    best_path=output_dir/"best.pt"
    if not best_path.exists():raise RuntimeError("No best checkpoint was produced.")
    best_checkpoint=torch.load(best_path,map_location=device,weights_only=False);model.load_state_dict(best_checkpoint["model_state"])
    y_true,y_prob=predict(model,val_loader,device);metrics,per_class,cm=compute_metrics(y_true,y_prob,class_names)
    metrics={"best_selection_metric":best,"selection_metric_name":args.selection_metric,"best_val_f1_macro":best if args.selection_metric=="f1_macro" else None,
             "best_val_tail_recall_macro":None if best_tail==float("-inf") else best_tail,"fold":fold,**metrics}
    save_evaluation(output_dir,y_true,y_prob,val_df,class_names);(output_dir/"metrics.json").write_text(json.dumps(json_safe(metrics),indent=2),encoding="utf-8")
    if args.calibrate_bias:
        bias,score=optimize_class_bias(y_true,y_prob,class_names,args.calibration_max_bias,args.calibration_step,args.calibration_passes,args.calibration_metric)
        calibrated=apply_class_bias(y_prob,bias); calibrated_metrics=save_evaluation(output_dir,y_true,calibrated,val_df,class_names,"calibrated")
        (output_dir/"calibration.json").write_text(json.dumps({"metric":args.calibration_metric,"optimized_score":score,"class_names":class_names,"class_bias":bias.tolist(),"metrics":calibrated_metrics},indent=2),encoding="utf-8")
        metrics["calibrated"]=calibrated_metrics;(output_dir/"metrics.json").write_text(json.dumps(json_safe(metrics),indent=2),encoding="utf-8")
    save_diagnostics(output_dir,args,data_summary,metrics,per_class,cm,y_prob,class_names,fold)
    return metrics


def run(args):
    if args.freeze_epochs+args.finetune_epochs<=0:raise ValueError("At least one epoch is required.")
    if args.k_folds<1:raise ValueError("--k-folds must be >=1.")
    if args.k_folds>1 and args.resume_checkpoint:raise ValueError("Resume a specific fold directly; root k-fold resume is unsupported.")
    set_seed(args.seed);args.output_dir=args.output_dir.expanduser().resolve();args.output_dir.mkdir(parents=True,exist_ok=True)
    df=load_dermoscopic_dataframe(args.data_dir,args.input_dir);df["is_augmented"]=synthetic_mask(df);df["ignore_metadata"]=False
    class_names=sorted(df.label.unique());label_to_idx={x:i for i,x in enumerate(class_names)};device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backbone_backend=="auto":
        if args.resume_checkpoint:
            saved=torch.load(args.resume_checkpoint,map_location="cpu",weights_only=False).get("args",{});backend=saved.get("backbone_backend","timm");backend="timm" if backend=="auto" else backend
        elif args.encoder_checkpoint:backend=infer_checkpoint_backend(args.encoder_checkpoint,device)
        else:backend="timm"
    else:backend=args.backbone_backend
    args.backbone_backend=backend
    results=[]
    for fold in range(args.k_folds):
        train_df,val_df=create_or_load_split(df,args.split_manifest,args.val_size,args.seed,args.synthetic_train_only,fold,args.k_folds)
        if args.synthetic_train_only and synthetic_mask(val_df).any():raise RuntimeError("Synthetic leakage detected in validation split.")
        output=args.output_dir if args.k_folds==1 else args.output_dir/f"fold_{fold:02d}"
        results.append(train_split(df,train_df,val_df,class_names,label_to_idx,args,device,backend,output,None if args.k_folds==1 else fold))
    if args.k_folds>1:save_kfold_summary(results,args.output_dir)
    return results[0] if args.k_folds==1 else results


def main():run(parse_args())
if __name__=="__main__":main()
