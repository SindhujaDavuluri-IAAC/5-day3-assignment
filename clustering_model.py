"""Train, predict, and publish Melbourne Housing PCA K-Means Clustering."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ["Landsize", "Rooms", "HouseAge", "Distance", "Bathroom", "Car", "Price", "Bedroom2"]


def inputs(data):
    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    values = data[FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Features must not contain infinity")
    return values


def train(data_path, model_path):
    data = pd.read_csv(data_path)
    values = inputs(data)
    if values.isna().all().any():
        raise ValueError("Each feature must have at least one observed value")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("kmeans", KMeans(n_clusters=4, n_init=10, random_state=42)),
    ])
    labels = model.fit_predict(values)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    restored = joblib.load(model_path)
    np.testing.assert_array_equal(restored.predict(values), labels)
    metadata = {
        "features": FEATURES,
        "training_data": str(data_path),
        "training_rows": len(data),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "clusters": 4,
        "cluster_counts": pd.Series(labels).value_counts().sort_index().to_dict(),
        "pca_components": int(model.named_steps["pca"].n_components_),
        "explained_variance": float(model.named_steps["pca"].explained_variance_ratio_.sum()),
        "notes": "Price is an input in AUD. HouseAge is age at sale in years. Type and coordinates are excluded. Cluster IDs are arbitrary and may change when retrained. This is not a price predictor.",
    }
    model_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved and verified {model_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict", "publish"])
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "melbourne_clustering.joblib")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "cluster_predictions.csv")
    parser.add_argument("--namespace", help="Hugging Face username or organization for publishing")
    parser.add_argument("--private", action="store_true", help="Create a private model repository")
    args = parser.parse_args()
    if args.command == "train":
        train(args.input or ROOT / "data" / "melbourne_house_prices_clean.csv", args.model)
    elif args.command == "publish":
        if not args.namespace:
            parser.error("publish requires --namespace YOUR_USERNAME_OR_ORG")
        from publish_model import MODEL_NAME, prepare_package
        from huggingface_hub import HfApi

        package_dir = ROOT / "model_release" / MODEL_NAME
        files = prepare_package(args.model, package_dir)
        api = HfApi()  # Uses HF_TOKEN or credentials from hf auth login.
        repo_id = f"{args.namespace}/{MODEL_NAME}"
        api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=str(package_dir),
            allow_patterns=files, commit_message=f"Publish {MODEL_NAME}",
        )
        print(f"Published: https://huggingface.co/{repo_id}")
    else:
        if args.input is None:
            parser.error("predict requires --input")
        data = pd.read_csv(args.input)
        model = joblib.load(args.model)
        data["cluster_id"] = model.predict(inputs(data))
        data.to_csv(args.output, index=False)
        print(f"Saved {len(data)} predictions to {args.output}")


if __name__ == "__main__":
    main()
