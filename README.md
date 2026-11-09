# TraceML: Training Data Attribution and Model Behavior Forensics

**Research question:** Which training examples are responsible for a model's
particular predictions, mistakes, or undesirable behavior?

TraceML implements training data attribution (TracIn and a nearest-neighbor
baseline) on a small CIFAR-10 CNN. It tests the attribution scores against
retraining ground truth and on two forensics tasks: finding corrupted labels
and finding a backdoor.

## Setup

- Model: small CNN (no BatchNorm), plain SGD, trained on a 5,000-image CIFAR-10
  subset with a learning-rate schedule. Final test accuracy is about 56%.
- Checkpoints and per-epoch learning rates are saved, because TracIn needs them.
- Everything runs on CPU.

```
common.py          shared model and data code
models/train.py   trains the model and saves checkpoints
attribution/      knn.py (baseline), tracin.py
eval/             retrain.py, analyze.py                  ground-truth test
                  label_noise.py, label_noise_val.py,
                  label_noise_cos.py                      label-noise forensics
                  backdoor.py, backdoor_fix.py,
                  backdoor_repair.py                      backdoor forensics
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
python -m eval.label_noise_cos
python -m eval.backdoor
python -m eval.backdoor_fix
python -m eval.backdoor_repair
```

Run them in this order, since later scripts reuse earlier outputs (checkpoints
and saved results). Generated files go to `ckpts/`, `data/` and `results/`,
which are git-ignored.

## Methods

- **kNN baseline:** rank training images by cosine similarity between the
  penultimate-layer embeddings of the test and training images.
- **TracIn:** for a test example, sum over saved checkpoints
  `learning_rate * <gradient of the test loss, gradient of the training loss>`
  for every training example. Positive scores mean helpful, negative mean
  harmful.
- **Cosine TracIn:** the same, but both gradients are normalized to unit length
  first, so only their direction matters.
- **Self-influence:** a training example's own TracIn score (squared gradient
  norm summed over checkpoints), used to find unusual examples.
- **Validation-influence:** score each training example against the gradients
  of a small set of trusted, clean validation images.

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
which kNN does not, so the comparison is not fully fair to kNN.

### 2. Ground truth: does removing the top examples change the prediction?

For each of 5 test images, the top-100 training examples chosen by each method
were removed, the model was retrained from scratch (3 seeds), and the increase
in that image's test loss was measured.

| Method | Mean increase in test loss |
|---|---|
| Random removal | -0.112 |
| kNN | +0.175 |
| TracIn | +1.008 |

Per image (mean over seeds, with standard deviation across seeds):

| Test image | Random | kNN | TracIn |
|---|---|---|---|
| 0 | +0.07 (0.18) | +0.05 (0.25) | +0.56 (0.09) |
| 1 | -0.50 (0.58) | -0.31 (0.75) | +1.64 (1.15) |
| 2 | -0.07 (0.28) | +0.08 (0.50) | +0.42 (0.66) |
| 3 | -0.02 (0.71) | +0.73 (0.63) | +1.66 (0.53) |
| 4 | -0.04 (0.09) | +0.33 (0.21) | +0.75 (0.38) |

TracIn has the highest mean on all 5 images. Retraining noise is large (random
removal alone moves a single image's loss by up to about 0.7), so exact values
should not be over-interpreted.

### 3. Forensics: finding corrupted labels (30% label noise)

1,500 of 5,000 training labels were flipped to a random other class. The model
trained on the noisy labels reaches 0.447 test accuracy. Scores are used to
flag the 1,500 most suspicious examples.

| Detector | AUROC | Precision@1500 |
|---|---|---|
| Random | 0.501 | 0.307 |
| TracIn raw, class-conditional validation-influence | 0.469 | 0.283 |
| TracIn validation-influence | 0.584 | 0.405 |
| TracIn cosine, all validation images | 0.663 | 0.463 |
| TracIn self-influence | 0.696 | 0.479 |
| TracIn cosine, class-conditional validation-influence | 0.744 | 0.561 |
| Loss + validation-influence | 0.773 | 0.569 |
| Loss + cosine class-conditional | 0.830 | 0.651 |
| **Training loss** | **0.847** | **0.663** |

The plain validation-influence row used 500 validation images; the three
variants with "class-conditional" or "all validation images" in their names used
1,000. Compare the variants with each other rather than with the older row.

Repair test: remove the flagged examples, retrain, and measure test accuracy on
the full 10,000-image test set (3 seeds).

| Setting | Test accuracy (sd) |
|---|---|
| Clean labels (ceiling) | 0.489 (0.016) |
| Oracle (remove exactly the corrupted) | 0.480 (0.002) |
| Training-loss cleaning | 0.460 (0.004) |
| Loss + cosine class-conditional | 0.454 (0.003) |
| No cleaning | 0.445 (0.005) |
| Loss + validation-influence | 0.437 (0.005) |
| Self-influence cleaning | 0.430 (0.002) |
| Cosine class-conditional cleaning | 0.426 (0.002) |
| Random removal | 0.426 (0.002) |
| Validation-influence cleaning | 0.390 (0.008) |

### 4. Forensics: finding a backdoor

200 of 5,000 training images (4%) were stamped with a 5x5 checkerboard trigger
and relabeled "airplane". The model keeps its clean accuracy (0.507) but sends
100% of triggered non-airplane test images to "airplane". Attribution was run on
20 triggered test images that the model sends to "airplane".

Fraction of each image's top-50 training examples that are poisoned
(base rate 0.04), and detection of all 200 poisoned images from the summed
scores:

| Method | Top-50 poisoned | Detection AUROC |
|---|---|---|
| Random | 0.046 | 0.530 |
| TracIn, raw dot product | 0.105 | 0.643 |
| TracIn, raw, true label | 0.070 | 0.730 |
| TracIn, raw, epoch-3 checkpoint only | 0.083 | 0.712 |
| kNN (embedding similarity) | 0.971 | 1.000 |
| **TracIn, cosine-normalized** | **0.992** | **1.000** |

As a control, raw TracIn applied to the same images without the trigger finds
no poisoned examples (top-50 fraction 0.000).

Repair: remove the 200 flagged examples, retrain (2 seeds), and re-measure.

| Setting | Clean accuracy | Attack success rate |
|---|---|---|
| No cleaning | 0.495 | 1.000 |
| Random removal | 0.511 | 1.000 |
| TracIn, raw | 0.509 | 0.993 |
| kNN | 0.516 | 0.070 |
| **TracIn, cosine** | **0.520** | **0.057** |
| Oracle (remove exactly the poisoned) | 0.520 | 0.057 |

Cosine TracIn flagged exactly the poisoned set, so its result equals the oracle.
Clean-accuracy differences of 1-2 points are within noise with 2 seeds; the drop
in attack success rate is the real effect.

## Findings

- TracIn finds training examples that causally affect individual predictions,
  and does so clearly better than random or nearest-neighbor selection in this
  setup (section 2).
- Raw TracIn failed at backdoor detection, and normalizing the gradients
  (cosine similarity) fixed it completely, with a full repair. A plausible
  explanation is that training examples with large gradients dominate the raw
  ranking, but I have not verified this.
- Normalization also helped for label noise (raw class-conditional AUROC 0.469,
  cosine 0.744), so the lesson held across both forensics tasks.
- Cosine TracIn matches, but does not beat, a simple kNN baseline on backdoor
  detection.
- For label noise, plain training loss beat every TracIn-based detector, and it
  was the only practical method that improved accuracy.
- Better detection did not guarantee better repair. Cosine class-conditional
  cleaning detected corrupted labels well above chance but gave no better
  accuracy than random removal, probably because about 44% of what it flagged
  was clean data. I have not tested this.

## Limitations

- Small model, small dataset (5,000 images), one dataset.
- The ground-truth test uses only 5 test images and 3 seeds, and per-image
  results are noisy.
- One label-corruption draw and one noise rate; one trigger pattern and one
  target class for the backdoor, which is an easy case.
- The backdoor analysis assumes some triggered, misclassified test images are
  already known.
- The kNN baseline in sections 1 and 2 ignores the label, which makes the
  comparison with TracIn unfair to it.
- Standard deviations reported in the repair tables reflect training-seed
  variation only, not the choice of corrupted examples.

## Roadmap

- [x] CIFAR-10 CNN with checkpoints
- [x] kNN baseline and TracIn attribution
- [x] Retraining ground-truth test
- [x] Label-noise detection and repair
- [x] Backdoor detection and repair
- [x] Cosine-normalized validation-influence for label noise
- [ ] Label-aware kNN and cosine TracIn in the ground-truth test
- [ ] Removal-size sweep and more test images
- [ ] Dashboard