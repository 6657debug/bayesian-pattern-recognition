# Variational Bayesian Last Layers for Digit Classification

An AI-generated small-scale study of the discriminative classification method from Harrison, Willes, and Snoek, [Variational Bayesian Last Layers](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ee56aa4fe26a189782f507d843fd5272-Abstract-Conference.html), ICLR 2024. The authors' package is available at [VectorInstitute/vbll](https://github.com/VectorInstitute/vbll). This repository contains an independent reproduction on the scikit-learn digits dataset; it does not claim to reproduce the paper's full benchmark suite.

## Project files

- `vbll_digits_classification.py`: AI-generated MLP baseline and post-training VBLL implementation.
- `run_multiseed_experiments.py`: ten-seed repeated stratified holdout analysis and paired summary.
- `AI_Assignment_2_Report.docx` and `AI_Assignment_2_Report.pdf`: detailed report with mathematical description, AI workflow, implementation, experiments, results, reflection, and citations.
- `AI_PROMPTS.md`: representative prompts and workflow record.
- `experiment_results.json`: seed-42 test results and ten-seed records and summaries.
- `test_metrics.png` and `multiseed_metrics.png`: single-seed and paired repeated-seed plots.

## Reproduce

Use Python 3.13 and install the packages listed in `requirements.txt`. The dataset ships with scikit-learn, so no data download is needed.

```powershell
python -m pip install -r requirements.txt
python vbll_digits_classification.py
python run_multiseed_experiments.py
```

The single-run command uses seed 42 and writes the individual JSON result and chart. The multi-seed command uses the prespecified seeds `11, 23, 42, 57, 99, 123, 2024, 2025, 3024, 4096`; it writes the paired results and descriptive mean and sample standard deviation into `experiment_results.json` and produces `multiseed_metrics.png`. Both models use the same split and feature network within each run. Checkpoints are selected on validation predictive NLL; test partitions are not used for tuning.

## Results in brief

Across ten repeated stratified holdouts, the deterministic MLP reached `96.64 ± 0.84%` test accuracy and VBLL reached `96.96 ± 0.72%`. The average paired accuracy change (VBLL minus MLP) was `+0.31 ± 0.63` percentage points, with VBLL more accurate in five of ten runs. VBLL's mean NLL, Brier score, ECE, and entropy-based error AUROC were not consistently better. The ten-run ECE was higher on average for VBLL even though it was lower in the seed-42 split. The report discusses the mixed result and limitations.

Mean ± sample SD describes these ten runs. Their test partitions overlap in source examples, so this is not an independent-sample confidence interval and should not be read as a formal significance test.

## Source code webpage

Direct link to the main generated experiment: [vbll_digits_classification.py](https://github.com/6657debug/bayesian-pattern-recognition/blob/main/vbll_digits_classification.py). The full project and report are in this [repository](https://github.com/6657debug/bayesian-pattern-recognition).
