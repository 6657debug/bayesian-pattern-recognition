# Variational Bayesian Last Layers for Digit Classification

This repository contains an AI-generated, reproducible pattern-recognition experiment and its report. It applies the Variational Bayesian Last Layers (VBLL) method to a small neural classifier for handwritten digits, then compares the Bayesian predictive distribution with a deterministic baseline.

## Algorithm and source

The experiment is based on Harrison, Willes, and Snoek, [Variational Bayesian Last Layers](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ee56aa4fe26a189782f507d843fd5272-Abstract-Conference.html), ICLR 2024. It trains a deterministic MLP, freezes its features, then fits independent diagonal Gaussian class-weight posteriors with the paper's deterministic classification bound. The authors' package is [VectorInstitute/vbll](https://github.com/VectorInstitute/vbll); this repository contains an AI-generated small-dataset reproduction of the discriminative classification objective.

## Reproduce the experiment

Use Python 3.13 and install the packages listed in `requirements.txt`. The digits data is included with scikit-learn, so the run does not download data.

```powershell
python -m pip install -r requirements.txt
python vbll_digits_classification.py
```

The script uses a fixed seed, stratified train/validation/test splits, selects checkpoints using validation predictive NLL, and writes `experiment_results.json` and `test_metrics.png` beside the script. It reports test accuracy, NLL, multiclass Brier score, 15-bin ECE, and entropy-based error-detection AUROC.

## Main result

On the held-out 450-image test set, the deterministic MLP reached 96.89% accuracy and VBLL reached 96.67%. VBLL's ECE was lower (0.0198 versus 0.0267), while its NLL (0.0998 versus 0.0962) and Brier score (0.0488 versus 0.0475) were slightly higher. This single-split experiment is consistent with a small calibration trade-off, not a general claim of superiority.

## Assignment report

The Word and PDF reports describe the AI-assisted algorithm search, method, implementation, experiment, results, and learning notes:

- `AI_Assignment_2_Report.docx`
- `AI_Assignment_2_Report.pdf`

The representative prompts used during the AI workflow are in `AI_PROMPTS.md`.
