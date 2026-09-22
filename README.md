# Credit Score Classifier — AWS Deployment

Individual project: the AWS deployment variant of the credit-tier classifier — trained via
a SageMaker-style pipeline and served through an EC2-hosted front end.

## Open this first
- [`train.ipynb`](./train.ipynb) — training notebook for the cloud pipeline.
- `train.py` / `pipeline.py` / `preprocessing.py` / `transformer.py` — the training pipeline, structured as separate OOP modules.
- `evaluation.py` — held-out evaluation.

## Result
Same modeling approach as the [local deployment](https://github.com/ivanwilliaml/credit-score-app):
best model (LightGBM), **Macro-F1 0.722**, ROC-AUC 0.885.

No dataset is committed here — see the notebook for the source.
