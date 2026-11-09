# TraceML: Training Data Attribution and Model Behavior Forensics

**Research question:** Which training examples are responsible for a model's
particular predictions, mistakes, or undesirable behavior?

TraceML implements training data attribution (TracIn and a nearest-neighbor
baseline) on a small CIFAR-10 CNN, and tests it against retraining ground truth
and two forensics tasks: finding corrupted labels and finding a backdoor.

## Setup

- Model: small CNN (no BatchNorm), plain SGD, trained on a 5,000-image CIFAR-10
  subset with a learning-rate schedule. Final test accuracy is about 56%.
- Checkpoints and per-epoch learning rates are saved for TracIn.
- Everything runs on CPU.

```
common.py        shared model and data code
models/train.py  trains the model and saves checkpoints
attribution/     knn.py (baseline), tracin.py
eval/            retrain.py, analyze.py            (ground truth)
                 label_noise.py, label_noise_val.py (label-noise forensics)
                 backdoor.py, backdoor_fix.py, backdoor_repair.py (backdoor forensics)
```

## Reproduce

```
pip install -r requirements.txt
python models/train.py
python -m attribution.knn
python -m attribution.tracin
python -m eval.retrain
python -m eval.analyze
python -m eval.label_noise
python -m eval.label_noise_val
python -m eval.backdoor
python -m eval.backdoor_fix
python -m eval.backdoor_repair
```

## Results

### 1. Do attribution scores point to examples of the right class?

Fraction of the top-10 training examples that share the test image's true class
(100 test images):

| Method | Same-class fraction |
|---|---|
| Random | 0.09 |
| kNN (embedding similarity) | 0.50 |
| TracIn, most helpful | 0.99 |
| TracIn, most harmful | 0.00 |

This is a sanity check, not causal evidence. TracIn also uses the test label,
which kNN does not, so the comparison is not fully fair.

### 2. Ground truth: does removing the top examples change the prediction?

For each of 5 test images, the top-100 training examples chosen by each method
were removed, the model was retrained from scratch (3 seeds), and the increase
in that image's test loss was measured.

| Method | Mean increase in test loss |
|---|---|
| Random removal | -0.112 |
| kNN | +0.175 |
| TracIn | +1.008 |

TracIn had the highest mean on all 5 images, but retraining noise is large
(per-image standard deviations up to about 1.1), so exact values should not be
over-interpreted.

### 3. Forensics: finding corrupted labels (30% label noise)

1,500 of 5,000 training labels were flipped to a random other class.

| Detector | AUROC | Precision@1500 |
|---|---|---|
| Random | 0.501 | 0.307 |
| TracIn validation-influence | 0.584 | 0.405 |
| TracIn self-influence | 0.696 | 0.479 |
| Loss + validation-influence | 0.773 | 0.569 |
| **Training loss** | **0.847** | **0.663** |

Repair test: remove the flagged examples, retrain, and measure test accuracy on
the full 10,000-image test set (3 seeds).

| Setting | Test accuracy |
|---|---|
| Clean labels (ceiling) | 0.489 |
| Oracle (remove exactly the corrupted) | 0.480 |
| Training-loss cleaning | 0.460 |
| No cleaning | 0.445 |
| Loss + validation-influence | 0.437 |
| Self-influence cleaning | 0.430 |
| Random removal | 0.426 |
| Validation-influence cleaning | 0.390 |

### 4. Forensics: finding a backdoor

200 of 5,000 training images (4%) were stamped with a 5x5 checkerboard trigger
and relabeled "airplane". The model keeps its clean accuracy (0.507) but sends
100% of triggered non-airplane test images to "airplane". TracIn was run on 20
triggered test images that the model misclassified.

Fraction of each image's top-50 training examples that are poisoned
(base rate 0.04):

| Method | Top-50 poisoned | Detection AUROC |
|---|---|---|
| Random | 0.046 | 0.530 |
| TracIn, raw dot product | 0.105 | 0.643 |
| TracIn, raw, true label | 0.070 | 0.730 |
| TracIn, raw, epoch-3 checkpoint only | 0.083 | 0.712 |
| kNN (embedding similarity) | 0.971 | 1.000 |
| **TracIn, cosine-normalized** | **0.992** | **1.000** |

Repair: remove the 200 flagged examples, retrain (2 seeds), and re-measure.

| Setting | Clean accuracy | Attack success rate |
|---|---|---|
| No cleaning | 0.495 | 1.000 |
| Random removal | 0.511 | 1.000 |
| TracIn, raw | 0.509 | 0.993 |
| kNN | 0.516 | 0.070 |
| **TracIn, cosine** | **0.520** | **0.057** |
| Oracle | 0.520 | 0.057 |

Cosine TracIn flagged exactly the poisoned set, so its result equals the oracle.
Clean-accuracy differences of 1-2 points are within noise with 2 seeds; the drop
in attack success is the real effect.

## Findings

- TracIn finds training examples that causally affect individual predictions,
  clearly more than random or nearest-neighbor selection in this setup.
- Raw TracIn failed at backdoor detection, and normalizing the gradients (cosine
  similarity) fixed it completely. Plausibly large-gradient training examples
  dominate the raw ranking; I have not verified this.
- Cosine TracIn matches, but does not beat, a simple kNN baseline on backdoor
  detection.
- For cleaning label noise, plain training loss beat every attribution-based
  detector I tried. Those were raw (not normalized), so the backdoor result
  suggests they may be worth retesting with normalization.

## Limitations

- Small model, small dataset (5,000 images), one dataset.
- Ground-truth test uses only 5 test images and 3 seeds.
- One label-corruption draw and one noise rate; one trigger pattern and one
  target class for the backdoor, which is an easy case.
- The backdoor analysis assumes some triggered, misclassified test images are
  already known.
- The kNN baseline in section 1 ignores the label, which makes that comparison
  unfair to it.

## Roadmap

- [x] CIFAR-10 CNN with checkpoints
- [x] kNN baseline and TracIn attribution
- [x] Retraining ground-truth test
- [x] Label-noise detection and repair
- [x] Backdoor detection and repair
- [ ] Normalized (cosine) validation-influence for label noise
- [ ] Label-aware kNN baseline
- [ ] Removal-size sweep and more test images
- [ ] Dashboard