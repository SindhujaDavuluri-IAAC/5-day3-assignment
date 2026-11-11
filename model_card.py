"""Build graphics and documentation directly from the released model."""
import json
import shutil

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clustering_model import ROOT, inputs


def build_card(model_path, output_dir, metadata):
    model = joblib.load(model_path)
    data = pd.read_csv(ROOT / "data" / metadata["training_data"])
    values = inputs(data)
    scores = model[:-1].transform(values)
    labels = model.predict(values)
    names = {int(k): v for k, v in metadata["cluster_names"].items()}
    counts = pd.Series(labels).value_counts().sort_index()
    if counts.to_dict() != {int(k): v for k, v in metadata["cluster_counts"].items()}:
        raise ValueError("Training data no longer matches released model counts")
    graphics = output_dir / "figures"
    graphics.mkdir(exist_ok=True)
    colors = ["#247BA0", "#E49B30", "#8F63B8", "#27977B"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#FAFAF7"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for i in counts.index:
        mask = labels == i
        axes[0].scatter(scores[mask, 0], scores[mask, 1], s=7, alpha=.35,
                        color=colors[i], label=f"{i}: {names[i]}", rasterized=True)
    variance = model.named_steps["pca"].explained_variance_ratio_
    axes[0].set(xlabel=f"PC1 ({variance[0]:.1%} variance)", ylabel=f"PC2 ({variance[1]:.1%} variance)", title="A two-component view of housing clusters")
    axes[0].legend(frameon=False, markerscale=3, fontsize=8)
    bars = axes[1].bar([f"Cluster {i}" for i in counts.index], counts, color=colors)
    axes[1].bar_label(bars, labels=[f"{n:,}\n{n/len(data):.1%}" for n in counts], padding=5)
    axes[1].set(title="How the training sales are grouped", ylabel="Property sales", ylim=(0, counts.max()*1.2))
    fig.savefig(graphics / "cluster_overview.png", dpi=170)
    plt.close(fig)
    means = values.groupby(labels).mean()
    z = (means - values.mean()) / values.std()
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(values.columns)), values.columns)
    ax.set_yticks(range(len(counts)), [f"{i}: {names[i]}" for i in counts.index])
    for i in range(len(counts)):
        for j in range(len(values.columns)):
            ax.text(j, i, f"{z.iloc[i,j]:+.1f}", ha="center", va="center",
                    color="white" if abs(z.iloc[i,j])>1.2 else "#202020")
    ax.set_title("Cluster profiles relative to the training average")
    fig.colorbar(im, ax=ax, label="Standard deviations above / below mean")
    fig.savefig(graphics / "cluster_profiles.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.bar(range(1, len(variance)+1), variance*100, color="#247BA0", label="Individual component")
    ax.plot(range(1,len(variance)+1), variance.cumsum()*100, "o-", color="#E49B30", label="Cumulative")
    ax.axhline(95, color="#777777", linestyle="--", linewidth=1)
    ax.set(xlabel="Principal component", ylabel="Explained variance (%)", title="Six components retain 95.95% of variance", ylim=(0,105))
    ax.legend(frameon=False)
    fig.savefig(graphics / "pca_variance.png", dpi=170)
    plt.close(fig)
    summary = values.assign(cluster_id=labels).groupby("cluster_id").median()
    summary.insert(0,"sales",counts)
    summary.insert(0,"cluster_label", pd.Series(names))
    summary.to_csv(output_dir / "cluster_profiles.csv")
    rows = "\n".join(f"| {i}: {names[i]} | {int(r.sales):,} | {r.Rooms:.1f} | {r.Landsize:.0f} | {r.HouseAge:.0f} | {r.Distance:.1f} | {r.Price:,.0f} |" for i,r in summary.iterrows())
    card = f"""---
language: en
license: mit
library_name: sklearn
tags:
- clustering
- kmeans
- pca
- tabular
- real-estate
- built-environment
- iaac-maai
model-index:
- name: Melbourne Housing PCA K-Means Clustering
  results: []
---
# Melbourne Housing PCA K-Means Clustering

Eight numeric property attributes, six principal components, four housing clusters.
This educational model groups **{len(data):,} Melbourne property sales** by their
physical characteristics, distance to the CBD and recorded sale price. It supports
exploratory analysis for AI Methods I, IAAC MaAI 2026/27.

![PCA cluster map and cluster sizes](figures/cluster_overview.png)

## Model at a glance

| Item | Detail |
|---|---|
| Authors | Breno, Diego, Sindhuja, Soumya |
| Algorithm | Median imputation → StandardScaler → PCA → K-Means |
| Training sample | {len(data):,} cleaned property sales |
| Inputs | 8 numeric attributes; Price is included |
| PCA | {metadata['pca_components']} components, {metadata['explained_variance']:.2%} retained variance |
| Clusters | 4; random_state=42; n_init=10 |
| Output | Integer cluster_id, 0–3 |
| Source code | [Assignment repository](https://github.com/SindhujaDavuluri-IAAC/5-day3-assignment) |

## Source and attribution

Training uses the project's `melbourne_house_prices_clean.csv`, derived from
[Melbourne Housing Market on Kaggle, Anthony Pino](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market).
The project notebook describes the original data as publicly posted Domain.com.au
sale results, with 34,857 rows and 21 columns before cleaning. The cleaning notebook
and unsupervised-learning notebook in the source repository document the preparation.
This release fits the saved cleaned CSV; it does not serialize a live notebook session.
Raw records, addresses and the training CSV are not distributed in this model repository.

## What we changed and fitted

1. Use the saved cleaned sales and select the eight numeric columns below.
2. Fill missing values with training-column medians.
3. Standardize each feature using training means and scales.
4. Keep enough PCA components to explain at least 95% of variance.
5. Fit four K-Means centroids in all six retained components.

Four clusters follow the assignment configuration. This release does not establish
that four is optimal: no k-selection comparison or stability study is reported.
The fitted preprocessing is bundled with K-Means, so prediction never refits it.
Property type, suburb, latitude and longitude are excluded. Distance remains an input.

## What the clusters look like

Values below are observed training medians; the full eight-feature table is available
in [cluster_profiles.csv](cluster_profiles.csv). Names follow the updated notebook:
smallest median land area → Compact inner-city units; highest remaining median
price → Large premium family houses; oldest remaining median age → Inner-city
heritage houses; farthest remaining median CBD distance → Affordable outer-suburb
houses. These are exploratory descriptions, not validated market segments.

| Cluster | Sales | Rooms | Land (m²) | Age (years) | CBD (km) | Price (AUD) |
|---|---:|---:|---:|---:|---:|---:|
{rows}

![Standardized cluster feature profiles](figures/cluster_profiles.png)

Red indicates an above-average cluster mean; blue indicates a below-average mean.
Price differences are partly a consequence of using Price as an input, not an
independent validation of the clusters. The PCA scatter above shows only two of six
components; apparent overlap is not a complete view of the fitted clustering space.

## PCA and evaluation

![Explained variance by PCA component](figures/pca_variance.png)

Retained variance describes dimensionality reduction, not clustering accuracy.
No ground-truth segment labels, held-out evaluation, silhouette score or deployment
performance claims are supplied. Training assignments were checked after saving and
reloading the model. Graphics and summary tables are recomputed from this artifact.

## Input columns

| Column | Meaning / unit |
|---|---|
| Landsize | Land area, square metres |
| Rooms | Recorded room count |
| HouseAge | Property age at sale, years |
| Distance | Distance to Melbourne CBD, kilometres |
| Bathroom | Bathroom count |
| Car | Car-space count |
| Price | Recorded sale price, Australian dollars |
| Bedroom2 | Recorded secondary bedroom-count field |

All eight columns must exist. Values must be numeric or missing; infinity is rejected.
Use the training feature order above when calling the pipeline directly.
HouseAge must be calculated at the sale date, not at today's date.

## How to load and use it

Install `requirements.txt` from this repository plus `huggingface_hub`, then:

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

repo_id = "SindhujaDavuluri-IAAC/melbourne-housing-pca-kmeans-clustering"
path = hf_hub_download(repo_id, "melbourne-housing-pca-kmeans-clustering.joblib")
model = joblib.load(path)
properties = pd.DataFrame([{{
    "Landsize": 450, "Rooms": 3, "HouseAge": 40, "Distance": 12,
    "Bathroom": 2, "Car": 1, "Price": 900000, "Bedroom2": 3
}}])
print(model.predict(properties).tolist())
```

Alternatively, download the repository and run:

```bash
python clustering_model.py predict --model melbourne-housing-pca-kmeans-clustering.joblib --input properties.csv --output predictions.csv
```

The CSV output preserves original columns and adds `cluster_id` and `cluster_label`.
The pipeline's `cluster_label_map_` attribute maps prediction IDs to names. Use the pinned
versions recorded in `requirements.txt` and `model_metadata.json`. Only load trusted
joblib files. Hugging Face hosts the artifact; this release provides no hosted API.

## Limitations and biases

The cleaned historical sales sample may not represent all Melbourne properties or
today's market. Cleaning, missing-value imputation, correlated room/bedroom fields,
and feature scaling influence boundaries. K-Means uses Euclidean distances and
assigns every record a cluster even when it is far outside training conditions.
IDs are arbitrary and can change on retraining. There are no calibrated confidence
scores. Use for education and exploration, not valuation, lending or automated
housing decisions. This model requires Price and cannot predict an unknown price.

## Files and reproducibility

- `melbourne-housing-pca-kmeans-clustering.joblib`: fitted four-step pipeline.
- `model_metadata.json`: features, versions, training counts and PCA information.
- `cluster_profiles.csv`: cluster medians and sample counts.
- `figures/`: three figures generated from the fitted model.
- `clustering_model.py`: CSV prediction and training implementation.
- `publish_model.py`, `model_card.py`: release preparation and publication code.
- `requirements.txt`: exact prediction dependencies.

To reproduce training, use the assignment repository's cleaned CSV and run
`python clustering_model.py train`. Cluster numbers from older notebook runs may differ.
To rebuild figures, additionally install matplotlib and run `python publish_model.py`.

## Authors and licence

Breno, Diego, Sindhuja and Soumya — IAAC MaAI. Release code and model artifact use
the assignment repository's MIT licence (see [LICENSE](LICENSE)). This does not
relicense the source dataset; consult its Kaggle terms for data reuse.
Card organization inspired by the
[IAAC daily load profiles example](https://huggingface.co/datasets/iaac-maai/day3-example-daily-load-profiles).
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")
    for name in ["LICENSE", "publish_model.py", "model_card.py"]:
        shutil.copy2(ROOT / name, output_dir / name)
    return ["cluster_profiles.csv", "figures/*.png", "LICENSE", "publish_model.py", "model_card.py"]
