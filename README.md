# 5-day3-assignment
Unsupervised Learning - Breno, Diego, Sindhuja, Soumya

## Reusable clustering model

The trained model is saved as `models/melbourne_clustering.joblib`, with training
details in `models/melbourne_clustering.json`. It was trained on the 12,078 rows
in `data/melbourne_house_prices_clean.csv` using the current notebook's numeric
workflow: median imputation, standard scaling, PCA (95% variance), and four
K-Means clusters (seed 42, 10 initializations). Six components retain 95.95% of
the variance. This is a fresh fit using the saved clean CSV; it does not export
an active notebook session or guarantee matching labels from older runs.

Install the model's dependencies:

```powershell
python -m pip install -r requirements.txt
```

Assign clusters to a CSV:

```powershell
python clustering_model.py predict --input data/melbourne_house_prices_clean.csv --output cluster_predictions.csv
```

Input columns: `Landsize` (square metres), `Rooms`, `HouseAge` (years at sale),
`Distance` (km to CBD), `Bathroom`, `Car`, `Price` (AUD), and `Bedroom2`.
All eight columns must be present; blank numeric values use training medians.
Additional columns are preserved in the output, which adds `cluster_id`.
Price **is used** in the current model, despite older notebook comments.
Property type and coordinates are excluded. This model assigns clusters; it
does not predict price. IDs are arbitrary and may change after retraining.

Retrain and verify that the saved model reproduces its training assignments:

```powershell
python clustering_model.py train
```

Only load joblib files from trusted sources. Use the pinned dependency versions
when loading this artifact. The model, illustrated model card and cluster profiles
are published at [Hugging Face](https://huggingface.co/SindhujaDavuluri-IAAC/melbourne-housing-pca-kmeans-clustering).
This is a downloadable model release; no hosted prediction endpoint is provided.

## Publish to Hugging Face

Team 5 organisation release:

- [Model and illustrated model card](https://huggingface.co/iaac-maai/day3-team5-melbourne-housing-clustering)
- [Cleaned and cluster-labelled dataset](https://huggingface.co/datasets/iaac-maai/day3-team5-melbourne-housing-clustering)

Rebuild with `python publish_team_release.py`; upload both organisation repositories
with `python publish_team_release.py --publish`. Cluster IDs are recomputed using
the existing fitted model so the dataset and graphics match the released artifact.

The published model name is `melbourne-housing-pca-kmeans-clustering`.
Prepare its model file, metadata, model card, prediction script and dependencies:

```powershell
python publish_model.py
```

Publish to your Hugging Face account (requires a token with write access):

```powershell
python -m pip install -r requirements-publish.txt
hf auth login
python publish_model.py --publish --namespace YOUR_USERNAME
```

Or publish directly from the training script:

```powershell
python clustering_model.py publish --namespace YOUR_USERNAME
```

The repository will be `YOUR_USERNAME/melbourne-housing-pca-kmeans-clustering`.
Add `--private` to create a private repository; existing repository visibility
is preserved. Authentication can also use the `HF_TOKEN` environment variable.
Without `--publish`, the script only prepares local files in `model_release/`.
Uploads include the model, code, model card, metadata, licence, cluster summary
and three graphics regenerated from the saved model. Training records are excluded.
