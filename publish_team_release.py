"""Publish Team 5's existing model and matching datasets to the IAAC organisation."""
import argparse
import json
import shutil

import joblib
import pandas as pd

from clustering_model import ROOT, inputs

REPO_ID = "iaac-maai/day3-team5-melbourne-housing-clustering"


def prepare():
    from publish_model import prepare_package

    source = ROOT / "model_release" / "melbourne-housing-pca-kmeans-clustering"
    prepare_package(ROOT / "models" / "melbourne_clustering.joblib", source)
    release = ROOT / "model_release" / "day3-team5-melbourne-housing-clustering"
    shutil.copytree(source, release, dirs_exist_ok=True)
    data_dir = release / "data"
    data_dir.mkdir(exist_ok=True)
    data = pd.read_csv(ROOT / "data" / "melbourne_house_prices_clean.csv")
    model = joblib.load(source / "melbourne-housing-pca-kmeans-clustering.joblib")
    data["cluster_id"] = model.predict(inputs(data))
    # Recompute IDs with the released model rather than reusing an older notebook CSV.
    data["cluster_label"] = data["cluster_id"].map(lambda i: f"Cluster {i}")
    data.to_csv(data_dir / "melbourne_house_prices_with_clusters.csv", index=False)
    shutil.copy2(ROOT / "data" / "melbourne_house_prices_clean.csv", data_dir)
    metadata = json.loads((release / "model_metadata.json").read_text(encoding="utf-8"))
    assert data["cluster_id"].value_counts().sort_index().to_dict() == {
        int(k): v for k, v in metadata["cluster_counts"].items()
    }
    metadata["repo_id"] = REPO_ID
    (release / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    card = (release / "README.md").read_text(encoding="utf-8")
    card = card.replace("SindhujaDavuluri-IAAC/melbourne-housing-pca-kmeans-clustering", REPO_ID)
    card = card.replace("Raw records, addresses and the training CSV are not distributed in this model repository.",
                        "The cleaned training CSV and a matching cluster-labelled CSV are included in `data/`.\nThe original uncleaned CSV is not included.")
    card += f"""

## Team 5 dataset

The dataset is also available as a [Hugging Face dataset](https://huggingface.co/datasets/{REPO_ID}).
`data/melbourne_house_prices_clean.csv` is the original cleaned training table.
`data/melbourne_house_prices_with_clusters.csv` adds predictions from this exact
released model; names are neutral `Cluster 0`–`Cluster 3`. Both have {len(data):,} rows.

```python
from datasets import load_dataset
data = load_dataset("{REPO_ID}", split="train")
```
"""
    (release / "README.md").write_text(card, encoding="utf-8")
    dataset = ROOT / "model_release" / "day3-team5-melbourne-housing-clustering-dataset"
    dataset.mkdir(exist_ok=True)
    shutil.copytree(data_dir, dataset / "data", dirs_exist_ok=True)
    shutil.copytree(release / "figures", dataset / "figures", dirs_exist_ok=True)
    dataset_card = f"""---
language: en
tags:
- clustering
- real-estate
- built-environment
- iaac-maai
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/melbourne_house_prices_with_clusters.csv
---
# Team 5 — Melbourne Housing Clustering

{len(data):,} cleaned Melbourne property sales with four PCA/K-Means cluster assignments.
Authors: Breno, Diego, Sindhuja, Soumya — IAAC MaAI, AI Methods I.

## Source and attribution

Derived from [Anthony Pino's Melbourne Housing Market dataset](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market),
described in the project notebook as publicly posted Domain.com.au sale results.
See the [source notebooks](https://github.com/SindhujaDavuluri-IAAC/5-day3-assignment)
for cleaning and feature engineering. Source data reuse remains subject to the
upstream dataset's terms; the software MIT licence does not relicense the data.

## What we changed

The cleaned training table has 28 columns. The labelled table adds `cluster_id`
and `cluster_label`, recomputed with the [released model](https://huggingface.co/{REPO_ID}).
The model uses Landsize, Rooms, HouseAge, Distance, Bathroom, Car, Price and Bedroom2,
with median imputation, standard scaling, PCA retaining 95% variance and four-cluster
K-Means. Price is an input. The label is descriptive only: `Cluster 0`–`Cluster 3`.

![Cluster overview](figures/cluster_overview.png)
![Cluster profiles](figures/cluster_profiles.png)

## Columns and files

- `data/melbourne_house_prices_clean.csv`: cleaned training records, 28 columns.
- `data/melbourne_house_prices_with_clusters.csv`: the same records plus two cluster columns.
- Cluster IDs are integers 0–3. See the model card for all eight feature definitions,
  units, cluster medians, pipeline configuration and limitations.
- Other columns preserve the project table, including suburb, address, sale date,
  coordinates, sale metadata and engineered attributes.

## How to load

```python
from datasets import load_dataset
sales = load_dataset("{REPO_ID}", split="train")
```

## Limitations and biases

Historical sales and cleaning choices limit representativeness. Missing-value
handling affects clustering. Cluster IDs are arbitrary; they are not validated
market segments or price predictions. This dataset has one descriptive training
split and no held-out evaluation. Do not treat clusters as valuation or lending advice.
"""
    (dataset / "README.md").write_text(dataset_card, encoding="utf-8")
    return release, dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    release, dataset = prepare()
    print(f"Prepared model and dataset: {REPO_ID}")
    if args.publish:
        from huggingface_hub import HfApi
        api = HfApi()
        for kind, folder in [("model", release), ("dataset", dataset)]:
            api.create_repo(repo_id=REPO_ID, repo_type=kind, private=False, exist_ok=True)
            api.upload_folder(repo_id=REPO_ID, repo_type=kind, folder_path=str(folder),
                              commit_message="Publish Team 5 Melbourne housing clustering model and dataset")
            info = api.repo_info(repo_id=REPO_ID, repo_type=kind)
            print(f"Verified {kind}: {info.id}; public={not info.private}; commit={info.sha}")


if __name__ == "__main__":
    main()
