# PAIEC 2026 entry: wide-prior label sharing for cold-start predictive evaluation

This repository holds the code of our final entry (archive `L3_wide_prior.zip`, Codabench submission 1033014) to the NeurIPS 2026 Predictive AI Evaluation Competition (Codabench competition 17828). It also holds the code that builds the training corpus, trains the text prior, runs the simulator of official draws and produces the technical report in `reports/paiec_v5_report/`.

## Method in short

`predict([subject, item], labeled)` returns a probability of success. The logit of a prediction is

```
logit = gamma * z + c + d_s + item shifts
```

- `z` is the logit of a frozen two-tower prior on E5 embeddings (`intfloat/multilingual-e5-small`, revision `614241f6`) of the subject description and the item text, with `gamma = 0.3`.
- `c ~ N(0, 2.5^2)` is a level shared by all subjects of one benchmark in the draw, and `d_s ~ N(0, 1.0^2)` is the deviation of one subject. Their posterior given all labels of the benchmark is a Laplace approximation, and predictions use the probit correction.
- Item shifts add the residuals of labeled copies of the same item (identity kernel) and of similar items (E5 kernel `exp((cos - 1) / 0.03)`, variance 3.0).

The submission runs on CPU and bundles the encoder, so it declares no Hugging Face repository. The report explains why the prior is wide. Test groups are much more extreme than training groups.

The code also supports a release-date offset and a model-name ability offset (archives L4 and L5). They helped in simulation, but official runs showed that they raise the Brier score before any label, because most test groups lie below 50% accuracy. Section 7.4 of the report gives the evidence, so the final entry does not use them.

## Layout

| Path | Contents |
| --- | --- |
| `v5/` | Submission runtime (`model.py`, `adapt.py`, `encoder.py`, `spec.py`, `textproj.py`, `name_ability.json`) and the archive builder `build_zip.py` |
| `colab_jobs/` | Corpus build, embeddings, prior training and export, subject offsets, the official-draw simulator, runtime replay, the difficulty-judge experiment and the Colab job control |
| `colab_jobs/results/` | Fitted subject offsets and the final prior `prior_B.npz` used in the archive |
| `scripts/` | Codabench upload, resubmission and score tracking |
| `reports/paiec_v5_report/` | Report source, PDF, number and figure scripts, and the simulation rows behind every table |

## Reproduction

Appendix A of the report lists every command. In short:

1. Request access to `aims-foundations/measurement-db` on Hugging Face and download revision `54074453da7b9911be0f48e4c560be547bc84d21` into `data/mdb_full`. Then run `colab_jobs/build_corpus.py` to build `data/corpus_full`.
2. On one GPU, run `colab_jobs/embed_texts.py`, `colab_jobs/train_eval.py` and `colab_jobs/export_z.py --tag B`. They write the item embeddings, the held-out logits `pairs_z.parquet` from five benchmark folds and the final prior `prior_B.npz`. We used Google Colab L4 machines through `colab_jobs/ctl.py`.
3. Optionally run `colab_jobs/fit_subject_priors.py` for the release-date and name offsets of L4 and L5. It reproduces our files exactly.
4. Run `colab_jobs/sim_official.py` and the scripts in `reports/paiec_v5_report/scripts/` for the simulations, tables and figures.
5. Rebuild the archive and check it with the organizers' validator from the starting kit.

```bash
P=colab_jobs/results/exportB/work/results
python v5/build_zip.py $P/prior_B.npz $P/prior_B.json 2.5 1.0 0.3 dist/v5/L3_wide_prior.zip \
  '{"use_prior": true, "item_exact": 1.0, "item_kernel": [0.03, 3.0]}'
python starting_kit/tools/check_submission_zip.py dist/v5/L3_wide_prior.zip
```

The submitted archive has SHA-256 `a47150effc0761d29b206ccb548d3c7da6a063e7a22072a352e60dc5b15f7ee7`. It was built before the offset code entered `v5/model.py`. A rebuild from this repository adds that code and the unused name table and returns identical predictions, because both offsets are zero without their settings. Appendix A of the report also gives the command for L5.

Scripts find the repository root from their own location. Set `PAIEC_ROOT` to use data stored elsewhere.

## Environment

CPU steps ran with Python 3.12.12, torch 2.14.1, transformers 5.19.0, numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1 and huggingface_hub 1.33.0 (`requirements.txt`). GPU steps used the preinstalled packages of Colab L4 runtimes.

## Data and licenses

The code is released under the MIT license. Training data come only from measurement-db (CC BY-SA 4.0), which is not redistributed here. The E5 encoder is `intfloat/multilingual-e5-small` (MIT). The archive builder downloads it at the pinned revision and checks the SHA-256 of every file.
