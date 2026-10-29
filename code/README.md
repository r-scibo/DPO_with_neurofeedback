# Code: fine-tuning a language model on brain signals

This folder reproduces every number and figure in the [blog post](https://r-scibo.github.io/DPO_with_neurofeedback/).
It runs from the EEG of each answer to the fine-tuned models, their answers and the statistics.

## Data

The EEG and ratings come from a public dataset (CC0): Choi et al. (2024), *EEG dataset for the recognition of different
emotions naturally induced in voice-user interaction*, Scientific Data 11:1084,
[figshare](https://doi.org/10.6084/m9.figshare.c.7207839). 44 people each heard 80 answers from a voice assistant, half
naming their source, and rated each answer on nine judgments.

Shipped in `data/`:

| File | What it is |
|---|---|
| `ratings.csv` | each person's ratings of each answer on the nine judgments (-3 to 3), with source (1/0) and voice (1 = announcer, 0 = child) |
| `folds.csv` | the 10 repeats of 5-fold cross-validation over answers |
| `people_groups.csv` | groups of people used by the zero-shot decoder's random draws |
| `predictions.npz` | the decoder's predictions used for the published models (output of step 2) |
| `study_pairs.csv` | every study question with its answer in both versions, with and without a source |
| `trial_text_map.csv` | which question each answer belongs to (reconstructed; always right about voice and source, about 55 % right about the exact question, see below) |
| `question_bank.csv` | the 236 test questions (and 60 calibration questions) with accepted answers |
| `datasets.csv` | every DPO dataset with its share of sourced answers chosen (output of step 3) |

Shipped in `results/`: the scores of all 1,720 models (`dpo_jobs.csv`, output of step 5), the statistics (step 6) and
the decoder's accuracy with its shuffled-ratings test (`decoder_accuracy.csv`, step 2).

Steps 1-2 start from the EEG of every answer (`work/eeg_epochs.npz`: 44 people x 80 answers x 63 channels x 4 s at
200 Hz, from 1 s before the end of the answer to 3 s after it). It is not redistributed here; it is built from the raw
recordings on figshare: notch filter at 60 Hz, band-pass 1-80 Hz, resampling to 200 Hz, ICA removing eye and heart
components, bad channels interpolated.

## Steps

```
pip install -r requirements.txt
python 1_features.py                   # EEG -> Riemannian features                          (minutes)
python 2_decode.py                     # features -> decoded judgments                        (25 min CPU)
python 2_decode.py --perm 200          #   ... plus the shuffled-ratings test                 (GPU)
python 3_build_dpo_data.py             # decoded judgments -> 1,683 DPO datasets              (minutes)
python 4_train_dpo.py --jobs-file work/train/jobs.txt   # train and answer, 1,720 models  (GPU, ~2 min each)
python 5_score.py                      # answers -> citing rate and accuracy per model        (seconds)
python 6_stats.py                      # per-person and pooled tests, prints the blog numbers (1 min)
python 7_figures.py                    # the blog figures                                     (seconds)
```

Any step can start from the shipped outputs of the step before. The quickest check takes a minute on a laptop:
`python 6_stats.py && python 7_figures.py` prints every number in the blog post and redraws its figures. Step 3 uses
the shipped `data/predictions.npz` unless given `--predictions work/predictions.npz`.

## What each step does

1. **Features.** Each answer's EEG becomes a covariance matrix, re-centred on the person's average and mapped to a flat
   (tangent) space: 2,016 numbers per answer.
2. **Decoder.** Ridge regression predicts each judgment from the features. Every prediction comes from a model that
   never saw that answer. *Calibrated*: trained on the person's other 64 answers. *Zero-shot*: trained on the other 43
   people. *Fine-tuned*: zero-shot, then adjusted on the person. Pairwise accuracy: how often the decoder orders two of
   the person's answers the way they rated them (chance 50 %). The test re-runs everything with shuffled ratings 200 times.
3. **Preference data.** For each person, every answer with a source is compared with every answer without one, in the
   same voice; the higher decoded score wins. Each comparison becomes two DPO examples, each the same question answered
   with and without the source. The sourced version is chosen in the same share of examples as it won comparisons, so
   the only thing a model can learn is how often to cite. 1,000 examples per dataset.
4. **Fine-tuning.** One LoRA adapter per dataset on Qwen2.5-1.5B-Instruct (DPO, 1 epoch, beta 0.1, learning rate 5e-6),
   then 4 answers to each of 236 new questions.
5. **Scoring.** Citing rate (a pattern for "according to", "source", "Wikipedia" and similar, checked by hand on 200
   answers: 97.5 % agreement) and accuracy.
6. **Statistics.** Per person: does the change in each model's citing rank people like their real preference? Spearman
   rho against 10,000 shuffles; Holm correction across the nine judgments. Registered in advance: Interesting,
   calibrated decoder. Pooled: one model from everyone's comparisons, against 39 models trained on coin flips.

## Reproduction checks

Run on the shipped data, the code gives:

- step 1: features identical to the published ones;
- step 2 (GPU, `--perm 200`): Interesting, calibrated decoder 55.3 % pairwise accuracy, p = 0.005, as published;
- step 2 (laptop CPU): predictions equal to `data/predictions.npz` within 7e-6 (the published run used float32 on a GPU);
- step 3: all 1,683 datasets byte-identical to the published training data, also when built from the step-2 predictions
  above;
- step 4: retrained models (the base model and three adapters, on an A40) gave answers identical to the published ones;
- steps 5-6: every model's scores and every statistic identical to the published ones (Interesting, calibrated:
  rho = 0.33, p = 0.013);
- step 7: the blog figures.

## Limits worth knowing

- The dataset does not record which question was played in which trial. `trial_text_map.csv` was reconstructed from
  answer durations, voice and source; it is about 55 % right about the exact question. Because each training pair
  differs only in the source, the question text matters little: the `calibrated-shufmap` models, trained with a random
  map, give a similar result (Interesting, calibrated: rho = 0.28 with the random map, 0.33 with the reconstructed one).
- Each person heard each question in only one version and came to one session.
