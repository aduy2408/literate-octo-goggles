from __future__ import annotations
import argparse,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd,torch
from PIL import Image
from torch import nn

from milk10k_effb2_dermoscopic_metadata.data import (DermoscopicMetadataDataset,append_augmented_rows,
    create_or_load_split,fit_metadata_spec,metadata_vector,source_lesion_id,synthetic_mask)
from milk10k_effb2_dermoscopic_metadata.losses import LDAMLoss,build_loss
from milk10k_effb2_dermoscopic_metadata.model import DermoscopicMetadataClassifier
from milk10k_effb2_dermoscopic_metadata.training import parse_args,train_split
from milk10k_effb2_dermoscopic_metadata.inference import parse_args as parse_inference_args,run as run_inference


class FakeEncoder(nn.Module):
    num_features=8
    def forward(self,x):
        pooled=x.mean((2,3));return torch.cat([pooled,pooled,pooled[:,:2]],1)


def rows(root,count=12):
    result=[]
    for i in range(count):
        path=root/f"{i}.jpg";Image.fromarray(np.full((24,24,3),100+i,dtype=np.uint8)).save(path)
        result.append({"lesion_id":f"L{i}","isic_id":f"I{i}","image_path":str(path),"label":"A" if i%2==0 else "B",
                       "age_approx":50,"sex":"x","skin_tone_class":2,"site":"arm","MONET_hair":.1,
                       "is_augmented":False,"ignore_metadata":False})
    return pd.DataFrame(result)


class ParityTests(unittest.TestCase):
    def test_manifest_v2_synthetic_never_in_validation_and_kfold_reuses(self):
        df=pd.DataFrame([{"lesion_id":f"R{i}","label":"A" if i%2==0 else "B"} for i in range(20)]+[
            {"lesion_id":f"R{i}__sdpair_{i}","label":"A" if i%2==0 else "B"} for i in range(4)])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"split.json"
            for fold in range(3):
                tr,va=create_or_load_split(df,path,.2,42,True,fold,3)
                self.assertFalse(synthetic_mask(va).any())
                train_sources=set(tr.loc[~synthetic_mask(tr),"lesion_id"].astype(str))
                val_sources=set(va["lesion_id"].astype(str))
                synthetic_sources={source_lesion_id(x) for x in tr.loc[synthetic_mask(tr),"lesion_id"]}
                self.assertTrue(synthetic_sources <= train_sources)
                self.assertFalse(synthetic_sources & val_sources)
            payload=json.loads(path.read_text());self.assertEqual(payload["schema_version"],2);self.assertEqual(len(payload["folds"]),3)

    def test_legacy_manifest_with_synthetic_validation_is_rejected(self):
        df=pd.DataFrame([{"lesion_id":"A","label":"X"},{"lesion_id":"B__sdpair_1","label":"X"}])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"split.json";path.write_text(json.dumps({"train_lesion_ids":["A"],"val_lesion_ids":["B__sdpair_1"]}))
            with self.assertRaisesRegex(ValueError,"synthetic validation"):create_or_load_split(df,path,.2,1,True)

    def test_appended_augmentation_cap_and_zero_metadata(self):
        base=pd.DataFrame([{"lesion_id":"base","label":"A"}]);aug=pd.DataFrame([
            {"lesion_id":f"base__sdpair_{i}","label":"A","age_approx":1} for i in range(4)])
        args=argparse.Namespace(augmented_data_dir=Path("x"),augmented_classes=["A"],augmented_max_per_class=2,seed=1,zero_augmented_metadata=True)
        with patch("milk10k_effb2_dermoscopic_metadata.data.load_dermoscopic_dataframe",return_value=aug):
            result=append_augmented_rows(base,base.copy(),args)
        self.assertEqual(len(result),3);self.assertEqual(sum(bool(x) for x in result.ignore_metadata if pd.notna(x)),2)

    def test_sampler_power_and_zero_metadata_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            df=rows(Path(tmp),4);df.loc[0,"ignore_metadata"]=True;spec=fit_metadata_spec(df)
            ds=DermoscopicMetadataDataset(df,{"A":0,"B":1},spec,lambda _:torch.zeros(3,8,8))
            self.assertTrue(torch.all(ds[0]["metadata"]==0))

    def test_all_losses_and_ldam_epoch_switch(self):
        df=pd.DataFrame({"label":["A","A","B","B"]});mapping={"A":0,"B":1};device=torch.device("cpu")
        base=dict(class_weight=True,focal_gamma=2.,dice_weight=.3,f1_weight=.3,f1_ignore_classes=[],f1_class_weight=[],
                  ldam_beta=.9,ldam_max_margin=.5,ldam_drw_start_epoch=2,ldam_alpha_max=10.)
        logits=torch.randn(4,2,requires_grad=True);labels=torch.tensor([0,0,1,1])
        for name in ("ce","focal","ldam","ce_dice","ce_f1"):
            loss=build_loss(df,mapping,argparse.Namespace(loss=name,**base),device);value=loss(logits,labels);self.assertTrue(torch.isfinite(value))
        ldam=build_loss(df,mapping,argparse.Namespace(loss="ldam",**base),device);self.assertIsInstance(ldam,LDAMLoss);ldam.set_epoch(3);self.assertEqual(ldam.current_epoch,3)

    def test_model_modes_and_old_checkpoint_shape(self):
        with patch("milk10k_effb2_dermoscopic_metadata.model.timm.create_model",return_value=FakeEncoder()):
            for mode in ("none","concat","gated_concat","gated_only"):
                model=DermoscopicMetadataClassifier(2,7,mode,imagenet_pretrained=False,branch_dim=8,metadata_dim=4,classifier_hidden_dim=6,metadata_gate_hidden_dim=3)
                self.assertEqual(tuple(model(torch.rand(2,3,8,8),torch.rand(2,7)).shape),(2,2))

    def test_training_outputs_checkpoint_v2_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);df=rows(root);train=df.iloc[:8].copy();val=df.iloc[8:].copy();out=root/"run"
            args=parse_args(["--data-dir",str(root),"--output-dir",str(out),"--split-manifest",str(root/"split.json"),
                "--metadata-mode","concat","--image-size","16","--batch-size","4","--freeze-epochs","1","--finetune-epochs","0","--patience","0"])
            with patch("milk10k_effb2_dermoscopic_metadata.model.timm.create_model",return_value=FakeEncoder()):
                train_split(df,train,val,["A","B"],{"A":0,"B":1},args,torch.device("cpu"),"timm",out)
            checkpoint=torch.load(out/"last.pt",map_location="cpu",weights_only=False)
            self.assertEqual(checkpoint["schema_version"],2);self.assertIn("scheduler_state",checkpoint);self.assertIn("scaler_state",checkpoint)
            args.resume_checkpoint=out/"last.pt";args.freeze_epochs=2
            with patch("milk10k_effb2_dermoscopic_metadata.model.timm.create_model",return_value=FakeEncoder()):
                train_split(df,train,val,["A","B"],{"A":0,"B":1},args,torch.device("cpu"),"timm",out)
            self.assertEqual(int(pd.read_csv(out/"history.csv").epoch.max()),2)
            for name in ("best.pt","history.csv","metrics.json","data_summary.json","split_summary.md","run_report.md","prediction_summary.json","confusion_analysis.json"):
                self.assertTrue((out/name).exists(),name)

    def test_inference_old_checkpoint_tta_calibration_and_debug_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/"input";lesion=inputs/"L1";lesion.mkdir(parents=True)
            Image.fromarray(np.full((20,20,3),120,dtype=np.uint8)).save(lesion/"I1.jpg")
            metadata_csv=root/"metadata.csv";pd.DataFrame([{"lesion_id":"L1","isic_id":"I1","image_type":"dermoscopic",
                "age_approx":50,"sex":"x","skin_tone_class":2,"site":"arm","MONET_hair":.1}]).to_csv(metadata_csv,index=False)
            spec={"sex_values":["unknown","x"],"site_values":["arm","unknown"],"monet_columns":["MONET_hair"]}
            with patch("milk10k_effb2_dermoscopic_metadata.model.timm.create_model",return_value=FakeEncoder()):
                model=DermoscopicMetadataClassifier(2,7,"concat",imagenet_pretrained=False,branch_dim=8,metadata_dim=4,classifier_hidden_dim=6)
            checkpoint=root/"old.pt";torch.save({"model_state":model.state_dict(),"class_names":["A","B"],"metadata_spec":spec,
                "args":{"metadata_mode":"concat","backbone":"efficientnet_b2","backbone_backend":"timm","branch_dim":8,"metadata_dim":4,"classifier_hidden_dim":6,"dropout":.3,"image_size":16}},checkpoint)
            calibration=root/"bias.json";calibration.write_text(json.dumps({"class_names":["A","B"],"class_bias":[0,0]}))
            output=root/"pred.csv";args=parse_inference_args(["--checkpoint",str(checkpoint),"--input-dir",str(inputs),"--metadata-csv",str(metadata_csv),
                "--output",str(output),"--tta-flips","--calibration-file",str(calibration),"--include-debug-columns"])
            with patch("milk10k_effb2_dermoscopic_metadata.model.timm.create_model",return_value=FakeEncoder()):run_inference(args)
            columns=pd.read_csv(output).columns.tolist();self.assertIn("predicted_label",columns);self.assertIn("confidence",columns);self.assertIn("isic_id",columns)


if __name__=="__main__":unittest.main()
