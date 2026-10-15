# Lab3 — completed baseline experiments

Open `Lab3.ipynb` in Jupyter. All 5 code cells have executed successfully; all results are embedded, including 7 figures (examples and 6 confusion matrices).

Keep `BloodMNIST.npz` beside the notebook to rerun. From the Lab3 directory:

```powershell
python -m pip install -r requirements.txt
python -c "import nbformat; from nbclient import NotebookClient; p='Lab3.ipynb'; n=nbformat.read(p,as_version=4); NotebookClient(n,timeout=1800).execute(); nbformat.write(n,p)"
```

## Observed accuracy

| Classifier | Balanced 5,000 / 1,000 | Full 11,959 / 3,421 |
|---|---:|---:|
| KNN (k=3) | 68.70% | 71.50% |
| Decision tree | 61.30% | 65.16% |
| SGD (max_iter=250) | 66.40% | 69.57% |

All other estimator settings are defaults. NumPy random seeds are fixed. RGB images are flattened to 2,352 raw pixel features. Balanced sampling selects 625 training and 125 testing examples per class, without replacement, from separate official splits. The validation split is not used for training or evaluation.

The notebook includes per-class precision/recall/F1, macro/weighted summaries, confusion matrices, training and prediction timing, actual SGD epoch counts, and discussion. Class IDs 0–7 are retained without guessing an external name mapping.

Validation passed: notebook schema; all code cells executed; no error outputs; total dataset size; sample balance/uniqueness/reproducibility; feature shapes; six evaluations; metric ranges; confusion matrix support and accuracy agreement. No training warnings occurred. SGD converged after 123 and 182 epochs under the default tolerance.

The two experiments change both training size/balance and the testing population, so differences cannot be attributed solely to training size. Results may vary across library/platform versions. The source PDF and dataset were not modified. No pre-existing Lab3 implementation or tests were present.
