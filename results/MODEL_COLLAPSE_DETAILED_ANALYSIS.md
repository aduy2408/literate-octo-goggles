# Detailed Model Collapse, Loss Failure, And Class Overlap Analysis

## Executive conclusion

The current models are not failing because the global metrics are uniformly bad. They are failing because the decision boundary collapses around a few dominant or visually similar classes. Accuracy and ROC-AUC can look acceptable while macro-F1 stays mediocre because several rare classes are either never predicted or are absorbed into common classes.

The recurring failure modes are:

| Failure mode | Evidence | Practical meaning |
|---|---|---|
| Tail-class collapse | `MAL_OTH` F1 is `0.0000` across focal, CE, CE-F1, CE-Dice, LDAM, augmentation, and ConvNeXt | The model effectively does not learn a usable decision region for `MAL_OTH` |
| Near-collapse of `BEN_OTH` | Best checked F1 only reaches `0.3077` with `INF-only +15`; most losses sit around `0.1250-0.1538`, ConvNeXt is `0.0000` | Loss changes barely recover this class |
| `INF` absorbed by `BCC` and neighbors | ConvNeXt predicts only 2/10 `INF`; EffV2 variants usually predict 4-5/10 | The model can rank `INF` somewhat, but the hard class decision is unstable |
| `MEL` / `NV` overlap | `MEL -> NV` and `NV -> MEL` are large across runs | These classes share enough visual feature space that simple loss swaps do not separate them |
| Keratinocyte-like cluster overlap | `AKIEC`, `BCC`, `BKL`, `SCCKA` repeatedly confuse with each other | The model is learning a broad lesion family, not clean class-specific boundaries |

The core problem is not just "bad backbone" or "bad loss". It is a combined tail-data + class-overlap + calibration/decision-boundary problem.

## Collapse by class

### Tail classes are not getting real decision regions

| Class | Support val | What happens | Why it matters |
|---|---:|---|---|
| `MAL_OTH` | 2 | F1 is `0.0000` in every inspected run | The model has no reliable positive region for this class |
| `BEN_OTH` | 9 | Usually 1/9 or 0/9 correct; best inspected run only 2/9 correct | The model treats it as residual noise and pushes it into `MEL`, `BCC`, `NV`, or `BKL` |
| `INF` | 10 | Often 4-5/10 correct in EffV2, 2/10 in ConvNeXt | It is partially learnable but frequently gets absorbed by `BCC` |
| `VASC` | 9 | Usually decent F1 despite tiny support | Not all tail classes are impossible; the collapse is class-specific |
| `DF` | 10 | Usually strong F1 | This suggests the pipeline can learn tail classes when features are separable |

Important point: the model is not simply ignoring every rare class. `DF` and `VASC` can work. The failure is concentrated where the class is both rare and visually/semantically close to another class.

## Per-loss failure table

EfficientNetV2-B2 metadata concat clean runs:

| Loss / approach | Macro-F1 | `BEN_OTH` F1 | `INF` F1 | `MAL_OTH` F1 | `MEL` F1 | `NV` F1 | `SCCKA` F1 | What this says |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Focal | 0.5430 | 0.1429 | 0.5263 | 0.0000 | 0.6087 | 0.8013 | 0.5060 | Focal improves some minority recall but causes broad over-prediction into hard classes |
| CE | 0.5602 | 0.1538 | 0.4444 | 0.0000 | 0.6467 | 0.8224 | 0.6200 | CE is more stable overall, but does not solve tail collapse |
| CE-F1 | 0.5637 | 0.1250 | 0.4444 | 0.0000 | 0.6460 | 0.8307 | 0.6286 | Best macro-F1, but gains come from medium/head classes, not the worst tail |
| CE-Dice | 0.5571 | 0.1250 | 0.4545 | 0.0000 | 0.6111 | 0.7987 | 0.6019 | Dice does not rescue `BEN_OTH` or `MAL_OTH` |
| LDAM | 0.5368 | 0.1429 | 0.4000 | 0.0000 | 0.6190 | 0.8235 | 0.6220 | Margin-based loss still leaves the rarest class dead |
| Focal + `INF-only +15` | 0.5828 | 0.3077 | 0.4211 | 0.0000 | 0.6310 | 0.8190 | 0.6000 | Augmentation helps global score and `BEN_OTH` slightly, but not `MAL_OTH` |

The loss functions are not fixing the real failure. They shuffle mistakes:

- CE/CE-F1 improve macro-F1 mostly by stabilizing `BCC`, `NV`, `SCCKA`, and some medium-support classes.
- Focal increases minority pressure but creates more false positives in confusing regions.
- CE-Dice and LDAM do not create a usable class island for `MAL_OTH`.
- Augmentation helps best overall, but the gain is not a full tail recovery.

## How the classes overlap into each other

### `MAL_OTH` is completely absorbed

Across inspected EffV2 runs, both validation `MAL_OTH` samples are always predicted as other classes:

| Run | `MAL_OTH` confusion |
|---|---|
| Focal clean | `MAL_OTH -> BCC` 1, `MAL_OTH -> NV` 1 |
| CE clean | `MAL_OTH -> BCC` 2 |
| CE-F1 clean | `MAL_OTH -> MEL` 1, `MAL_OTH -> NV` 1 |
| CE-Dice clean | `MAL_OTH -> BCC` 1, `MAL_OTH -> NV` 1 |
| LDAM clean | `MAL_OTH -> BCC` 1, `MAL_OTH -> NV` 1 |
| Focal + `INF-only +15` | `MAL_OTH -> MEL` 1, `MAL_OTH -> NV` 1 |
| ConvNeXt focal clean | `MAL_OTH -> BCC` 2 |

This is total collapse. No current recipe gives `MAL_OTH` even one correct validation prediction. With support `2`, metric volatility is high, but the consistency of the failure across losses is still meaningful.

### `BEN_OTH` gets treated like ambiguous leftovers

`BEN_OTH` has 9 validation samples. It rarely gets a stable prediction:

| Run | Correct `BEN_OTH` | Main wrong destinations |
|---|---:|---|
| ConvNeXt focal clean | 0/9 | `MEL` 4, `BCC` 2, scattered into `AKIEC/BKL/NV` |
| EffV2 focal clean | 1/9 | `BCC`, `BKL`, `INF`, `MEL`, `NV`, `SCCKA` |
| EffV2 CE clean | 1/9 | `BCC`, `BKL`, `INF`, `MEL`, `NV` |
| EffV2 CE-F1 clean | 1/9 | `BCC`, `INF`, `MEL`, `NV` |
| EffV2 CE-Dice clean | 1/9 | `BCC`, `INF`, `MEL`, `NV` |
| EffV2 focal + `INF-only +15` | 2/9 | `MEL` 3, `NV` 2, `BCC/BKL` |

The class does not overlap with one single neighbor. It leaks into many classes, which means the learned representation is not carving out a compact `BEN_OTH` region. It is being interpreted as "whatever benign-ish or ambiguous class looks closest".

### `INF` mostly collapses toward `BCC`

`INF` has 10 validation samples. It is not as dead as `MAL_OTH`, but it is unstable:

| Run | Correct `INF` | Main wrong destinations |
|---|---:|---|
| ConvNeXt focal clean | 2/10 | `BCC` 6 |
| EffV2 focal clean | 5/10 | `BCC` 2, `BKL/MEL/SCCKA` |
| EffV2 CE clean | 4/10 | `BCC` 4 |
| EffV2 CE-F1 clean | 4/10 | `BCC` 2, scattered |
| EffV2 CE-Dice clean | 5/10 | `BCC` 2, `BEN_OTH/MEL/SCCKA` |
| EffV2 focal + `INF-only +15` | 4/10 | `AKIEC` 2, `BCC` 2 |

The surprising bit: `INF-only +15` is the best overall run, but it does not make `INF` itself perfect. The macro-F1 gain likely comes from broader boundary changes, not simply "INF augmentation fixed INF".

### `MEL` and `NV` form a persistent two-way overlap

This pair keeps bleeding both directions:

| Run | `MEL -> NV` | `NV -> MEL` |
|---|---:|---:|
| ConvNeXt focal clean | 17/90 | 16/149 |
| EffV2 focal clean | 26/90 | 10/149 |
| EffV2 CE clean | 22/90 | 12/149 |
| EffV2 CE-F1 clean | 23/90 | 6/149 |
| EffV2 CE-Dice clean | 20/90 | 18/149 |
| EffV2 focal + `INF-only +15` | 26/90 | 8/149 |

CE-F1 reduces `NV -> MEL` compared to focal/CE-Dice, but `MEL -> NV` remains large. This means loss tuning is not enough; the representation still shares visual space for these two classes.

### `AKIEC` / `BCC` / `BKL` / `SCCKA` are a confused lesion cluster

This cluster is where many "reasonable" but wrong predictions happen.

Examples:

| Run | Confusion pattern |
|---|---|
| ConvNeXt focal clean | `AKIEC -> SCCKA` 12, `BKL -> BCC` 19, `BKL -> SCCKA` 17, `SCCKA -> BCC` 13, `SCCKA -> BKL` 12 |
| EffV2 focal clean | `BCC -> AKIEC` 55, `BCC -> SCCKA` 60, `BKL -> SCCKA` 15, `SCCKA -> AKIEC` 21 |
| EffV2 CE clean | `BCC -> AKIEC` 22, `BCC -> BKL` 18, `BCC -> SCCKA` 17, `BKL -> AKIEC` 15, `SCCKA -> AKIEC` 16 |
| EffV2 CE-F1 clean | `BCC -> BKL` 23, `BCC -> SCCKA` 21, `BKL -> SCCKA` 14, `SCCKA -> AKIEC` 11 |
| EffV2 focal + `INF-only +15` | `BCC -> AKIEC` 24, `BCC -> SCCKA` 26, `BKL -> BCC` 16, `SCCKA -> AKIEC` 13 |

Focal clean is especially ugly here: it gets high recall for `AKIEC` and `SCCKA`, but does that by pulling many `BCC` samples into those classes. That explains the low accuracy `0.6813` despite balanced accuracy `0.5614`.

## Why the losses are not helping enough

### Focal loss

Focal is supposed to focus on hard examples, but here "hard" includes ambiguous inter-class boundaries. It increases pressure on difficult samples but does not know whether a hard sample is genuinely underrepresented or visually overlapping.

Observed behavior:

- `AKIEC` recall becomes high in EffV2 focal clean: `0.7705`.
- But `BCC -> AKIEC` explodes to 55 samples.
- `SCCKA` recall is decent: `0.6632`.
- But `BCC -> SCCKA` also explodes to 60 samples.
- Accuracy drops to `0.6813`.

So focal is not cleanly fixing tail classes. It is moving decision boundaries aggressively and paying for it with head-class false positives.

### CE and CE-F1

CE and CE-F1 are more stable. They recover accuracy:

| Loss | Accuracy | Macro-F1 |
|---|---:|---:|
| Focal | 0.6813 | 0.5430 |
| CE | 0.7548 | 0.5602 |
| CE-F1 | 0.7634 | 0.5637 |

But they still fail the worst classes:

- `MAL_OTH`: `0.0000` F1.
- `BEN_OTH`: only `0.1538` for CE and `0.1250` for CE-F1.
- `INF`: `0.4444` for both CE and CE-F1.

CE-F1 improves the average, but not by solving the deepest collapse. It improves enough medium/head classes to lift macro-F1 while the rarest class remains dead.

### CE-Dice

CE-Dice improves balanced accuracy to `0.5784`, the best among clean losses in the table, but it still leaves:

- `MAL_OTH` F1 `0.0000`.
- `BEN_OTH` F1 `0.1250`.
- `MEL -> NV` 20 and `NV -> MEL` 18.

Dice-like pressure helps distribution-level overlap, but it still does not force a reliable class-specific region for extremely rare ambiguous classes.

### LDAM

LDAM should help long-tailed margins, but in this setup it underperforms:

- Macro-F1 `0.5368`, lowest among the clean loss matrix.
- `MAL_OTH` still `0.0000`.
- `BEN_OTH` only `0.1429`.
- `BKL` recall drops to `0.4495`.

This suggests the margin adjustment is not aligned with the actual overlap structure or needs a different schedule/config. As-is, it is not a fix.

### Augmentation

`INF-only +15` is the best current approach by macro-F1 `0.5828`, but the detailed view matters:

- `BEN_OTH` improves to F1 `0.3077`, but only 2/9 correct.
- `MAL_OTH` remains F1 `0.0000`.
- `INF` is only 4/10 correct.
- `MEL -> NV` remains 26/90.

So augmentation helps the overall boundary, but it is not a complete tail solution. It is the best current baseline, not a solved model.

## Model-specific collapse

### ConvNeXt-Base

ConvNeXt clean no-metadata looks deceptively decent:

- Macro-F1 `0.5518`.
- Accuracy `0.7739`.
- ROC-AUC macro `0.9374`.

But it collapses hard:

| Class | F1 | Collapse |
|---|---:|---|
| `BEN_OTH` | 0.0000 | 0/9 correct; mostly `MEL/BCC` |
| `MAL_OTH` | 0.0000 | 0/2 correct; both become `BCC` |
| `INF` | 0.3333 | 2/10 correct; 6/10 become `BCC` |

ConvNeXt is likely learning strong general features for common/separable classes, but it does not preserve minority boundaries. The high ROC-AUC suggests ranking signal exists, but the final hard predictions are bad for tails.

### EfficientNetV2-B2

EffV2-B2 is currently the best because it has the strongest full recipe, not because the raw backbone is naturally best:

- No-metadata backbone clean is weak: macro-F1 `0.5223`.
- Metadata helps.
- CE-F1 helps.
- `INF-only +15` helps most.

But even in the best current result, the model still has unresolved collapse:

- `MAL_OTH` F1 `0.0000`.
- `BEN_OTH` only `0.3077`.
- `INF` only `0.4211`.
- `MEL/NV` overlap remains large.

EffV2-B2 is the best baseline because its recipe is most developed, not because the class separation problem is solved.

### EfficientNet-B2

EffB2 clean backbone is not impressive, but CE + EMA/LWS gets macro-F1 `0.5679`. That suggests post-training/logit adjustment can help hard decisions.

However, this does not prove tail collapse is solved. The EMA/LWS result needs the same per-class audit before being trusted as a final direction. If it lifts macro-F1 by shifting medium classes while leaving `MAL_OTH` and `BEN_OTH` dead, it is the same problem in another form.

## What this implies

### Do not trust accuracy alone

Accuracy rewards correct `BCC`, `NV`, and other common classes. A model can have good accuracy while failing several clinically important or rare categories.

### Do not trust ROC-AUC alone

ROC-AUC can stay high when the model ranks positives somewhat well but chooses poor final class boundaries. ConvNeXt is the clearest case: ROC-AUC macro `0.9374`, but `BEN_OTH` and `MAL_OTH` are dead.

### Macro-F1 is closer, but still not enough

Macro-F1 surfaces the collapse, but it compresses the story. The real acceptance criteria must include per-class F1/recall for:

- `BEN_OTH`
- `INF`
- `MAL_OTH`
- `MEL`
- `NV`
- `AKIEC/BKL/SCCKA`

## Concrete next steps

### 1. Stop testing losses in isolation

The clean loss matrix already shows loss alone does not solve tail collapse. Next runs should combine the best components:

| Run | Reason |
|---|---|
| `effv2b2_meta_concat_ce_f1_s05_aug_inf15_safe` | Best clean loss + best augmentation |
| `effv2b2_meta_concat_ce_dice_s05_aug_inf15_safe` | CE-Dice has best balanced accuracy clean; test with augmentation |
| `effv2b2_meta_concat_ce_tau050_sampler_ema_lws` | Bring EMA/LWS to the main EffV2 recipe |

### 2. Audit EMA/LWS per-class before trusting it

EMA/LWS macro-F1 `0.5679` is promising, but acceptance should require:

- `BEN_OTH` F1 improves beyond `0.3077`, or at least recall above 2/9.
- `INF` recall improves beyond 5/10.
- `MAL_OTH` gets at least one correct prediction, if possible.
- `MEL/NV` two-way confusion decreases.

### 3. Give ConvNeXt a fair recipe, but demand tail recovery

Run:

| Run | Must prove |
|---|---|
| `convnext_meta_concat_ce_f1_s05_clean` | Metadata + CE-F1 improves tail and not only accuracy |
| `convnext_meta_concat_focal_s05_aug_inf15_safe` | Best augmentation can rescue ConvNeXt tail |
| `convnext_meta_concat_ce_f1_s05_aug_inf15_safe` | Strongest ConvNeXt candidate |

If ConvNeXt still has `BEN_OTH=0`, `MAL_OTH=0`, and low `INF`, drop it from final candidate despite good ROC-AUC.

### 4. Treat `MAL_OTH` specially

Current evidence says normal multiclass training does not learn `MAL_OTH`. Options:

- Merge or redefine if the class is too small/noisy for 11-way classification.
- Use binary or hierarchical auxiliary head for `MAL_OTH` / rare malignant other.
- Use source-safe targeted augmentation if valid samples exist.
- Report it explicitly as unresolved if support remains too low.

Do not pretend a loss function fixed `MAL_OTH`; none has.

### 5. Add class-cluster diagnostics to every future report

Every future run should include a small block:

| Cluster | Required checks |
|---|---|
| Tail rare | `BEN_OTH`, `INF`, `MAL_OTH`, `DF`, `VASC` per-class F1/recall |
| Melanocytic | `MEL -> NV`, `NV -> MEL` |
| Keratinocyte-like | `AKIEC/BCC/BKL/SCCKA` cross-confusions |
| Head-class leakage | `BCC` false positives into minority classes |

Without this, a run can look like an improvement while merely moving errors around.

## Bottom line

The models are collapsing in predictable places:

- `MAL_OTH` has no learned decision region.
- `BEN_OTH` is scattered across visually adjacent classes.
- `INF` is often swallowed by `BCC`.
- `MEL` and `NV` remain entangled.
- `AKIEC`, `BCC`, `BKL`, and `SCCKA` form a broad overlapping cluster.

The losses are not enough because they adjust pressure, not representation quality or class separability. Focal over-pushes hard examples and causes leakage. CE/CE-F1 stabilize the model but leave the rarest classes dead. CE-Dice and LDAM do not solve the same collapse. Augmentation is the best current lever, but even the best augmentation run still leaves unresolved class overlap.

The next useful work is not another isolated loss sweep. It is combined recipes plus per-class collapse gates: CE-F1 or EMA/LWS with source-safe augmentation, metadata on ConvNeXt, and explicit acceptance criteria for the classes that are currently being swallowed.
