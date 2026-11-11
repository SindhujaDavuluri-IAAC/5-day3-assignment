"""Package the Melbourne clustering model and optionally publish to Hugging Face."""
import argparse
import json
import shutil
from pathlib import Path

from clustering_model import ROOT

MODEL_NAME = "melbourne-housing-pca-kmeans-clustering"


def prepare_package(model_path, output_dir):
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    if not model_path.is_file():
        raise FileNotFoundError(f"Train the model first: {model_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, output_dir / f"{MODEL_NAME}.joblib")
    # Publish the dataset filename without exposing a local machine path.
    metadata["training_data"] = Path(metadata["training_data"]).name
    metadata["model_name"] = MODEL_NAME
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.copy2(ROOT / "clustering_model.py", output_dir / "clustering_model.py")
    versions = {"scikit-learn": "sklearn_version", "pandas": "pandas_version",
                "numpy": "numpy_version", "joblib": "joblib_version"}
    (output_dir / "requirements.txt").write_text(
        "".join(f"{package}=={metadata[key]}\n" for package, key in versions.items()), encoding="utf-8")
    card = f"""---
library_name: sklearn
tags:
- clustering
- kmeans
- pca
- tabular
- real-estate
model-index:
- name: {MODEL_NAME}
  results: []
---
# Melbourne Housing PCA K-Means Clustering

Groups Melbourne property sales into {metadata['clusters']} clusters. Trained on
{metadata['training_rows']:,} cleaned sales using median imputation, standard
scaling, PCA retaining at least 95% variance, and K-Means (seed 42, n_init=10).
PCA retains {metadata['pca_components']} components and
{metadata['explained_variance']:.2%} of training variance.

## Inputs

Required numeric columns, in training order: {', '.join(metadata['features'])}.
Landsize is in square metres, HouseAge is years at sale, Distance is kilometres
to Melbourne CBD, and Price is AUD. Blank numeric values use training medians.
Property type and coordinates are excluded. Price IS an input.

## Use

Download this repository's files and install `requirements.txt`, then run:

```bash
python clustering_model.py predict --model {MODEL_NAME}.joblib --input properties.csv --output predictions.csv
```

The output preserves input columns and adds `cluster_id`.

## Limits

This is exploratory clustering, not a price prediction model. Cluster IDs are
arbitrary and may change after retraining. Results reflect the historical
cleaned training sample; performance on new data has not been evaluated.
Training cluster counts: {json.dumps(metadata['cluster_counts'])}.
The original dataset and notebooks are not included in this package.
Only load trusted joblib artifacts using the recorded dependency versions.
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")
    from model_card import build_card
    extra_files = build_card(model_path, output_dir, metadata)
    return [f"{MODEL_NAME}.joblib", "model_metadata.json", "clustering_model.py", "requirements.txt", "README.md"] + extra_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "melbourne_clustering.joblib")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model_release" / MODEL_NAME)
    parser.add_argument("--namespace", help="Hugging Face username or organization")
    parser.add_argument("--publish", action="store_true", help="Upload the prepared package")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    args = parser.parse_args()
    if args.publish and not args.namespace:
        parser.error("--publish requires --namespace YOUR_USERNAME_OR_ORG")
    files = prepare_package(args.model, args.output_dir)
    print(f"Prepared {MODEL_NAME}: {args.output_dir}")
    if args.publish:
        from huggingface_hub import HfApi

        api = HfApi()  # Uses HF_TOKEN or credentials saved by hf auth login.
        repo_id = f"{args.namespace}/{MODEL_NAME}"
        api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(args.output_dir),
                          allow_patterns=files, commit_message=f"Publish {MODEL_NAME}")
        print(f"Published: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
