from __future__ import annotations
import sys,tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plan_and_materialize_balanced_milk10k import LABEL_COLUMNS,parse_args,run


def gt_row(lesion,label):
    return {"lesion_id":lesion,**{name:float(name==label) for name in LABEL_COLUMNS}}


def image(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.full((8,8,3),value,dtype=np.uint8)).save(path)


def fixture(root,missing_synthetic=False):
    base=root/"base";images=base/"MILK10k_Training_Input";base.mkdir()
    labels=["BCC"]*10+["NV"]*4+["BEN_OTH"]*2
    gt=[];meta=[]
    for i,label in enumerate(labels):
        lesion=f"L{i}";gt.append(gt_row(lesion,label))
        for modality,suffix in (("clinical: close-up","c"),("dermoscopic","d")):
            isic=f"{lesion}_{suffix}";meta.append({"lesion_id":lesion,"image_type":modality,"isic_id":isic,"age_approx":40+i,"sex":"x","skin_tone_class":2,"site":"arm"});image(images/lesion/f"{isic}.jpg",80+i)
    pd.DataFrame(gt).to_csv(base/"MILK10k_Training_GroundTruth.csv",index=False);pd.DataFrame(meta).to_csv(base/"MILK10k_Training_Metadata.csv",index=False)
    synth_root=root/"synthetic";aug_gt=list(gt);aug_meta=list(meta);qc=[]
    for i in range(4):
        lesion=f"L{10+i%2}__sdpair_{i:03d}";aug_gt.append(gt_row(lesion,"BEN_OTH"))
        for modality,suffix in (("clinical: close-up","clinical"),("dermoscopic","dermoscopic")):
            isic=f"{lesion}__{suffix}";aug_meta.append({"lesion_id":lesion,"image_type":modality,"isic_id":isic,"age_approx":"","sex":"unknown","skin_tone_class":"","site":"unknown"})
            if not missing_synthetic:image(synth_root/lesion/f"{isic}.jpg",120+i)
        qc.append({"synthetic_lesion_id":lesion,"target_class_probability":.9-i*.1,"is_target_predicted":"True"})
    info=base/"augmented_info";info.mkdir();pd.DataFrame(aug_gt).to_csv(info/"MILK10k_Training_GroundTruth(2).csv",index=False);pd.DataFrame(aug_meta).to_csv(info/"MILK10k_Training_Metadata(3).csv",index=False)
    qc_path=root/"qc.csv";pd.DataFrame(qc).to_csv(qc_path,index=False)
    return base,synth_root,qc_path


class BalancePlannerTests(unittest.TestCase):
    def test_caps_qc_reproducibility_and_hardlink_materialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,synth,qc=fixture(root)
            outputs=[]
            for index in range(2):
                report=root/f"report{index}";materialized=root/f"out{index}"
                args=parse_args(["--base-data-dir",str(base),"--synthetic-input-dir",str(synth),"--qc-summary",str(qc),"--report-dir",str(report),"--materialize-dir",str(materialized),"--max-synthetic-per-source","1","--seed","7"])
                result=run(args);outputs.append(pd.read_csv(materialized/"MILK10k_Training_GroundTruth.csv"))
                self.assertEqual(result["bcc_cap"],6);self.assertLessEqual(len(result["usable"]),2)
                self.assertTrue((report/"class_distribution.png").exists());self.assertTrue((report/"run_balance_pipeline.sh").exists())
                first=materialized/"MILK10k_Training_Input/L0/L0_c.jpg";self.assertEqual(first.stat().st_ino,(base/"MILK10k_Training_Input/L0/L0_c.jpg").stat().st_ino)
            self.assertEqual(set(outputs[0].lesion_id),set(outputs[1].lesion_id))

    def test_materialization_refuses_missing_synthetic_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,synth,qc=fixture(root,True)
            args=parse_args(["--base-data-dir",str(base),"--synthetic-input-dir",str(synth),"--qc-summary",str(qc),"--report-dir",str(root/"report"),"--materialize-dir",str(root/"out")])
            with self.assertRaises((ValueError,FileNotFoundError)):run(args)

    def test_audit_without_images_reports_inventory_but_no_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,_,_=fixture(root)
            result=run(parse_args(["--base-data-dir",str(base),"--report-dir",str(root/"report")]))
            self.assertEqual(len(result["manifest"]),4);self.assertEqual(len(result["usable"]),0)

    def test_fresh_start_ignores_existing_synthetic_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,_,_=fixture(root)
            result=run(parse_args(["--base-data-dir",str(base),"--fresh-start","--report-dir",str(root/"fresh")]))
            self.assertEqual(len(result["manifest"]),0)
            self.assertEqual(int(result["plan"].raw_synthetic_inventory.sum()),0)


if __name__=="__main__":unittest.main()
