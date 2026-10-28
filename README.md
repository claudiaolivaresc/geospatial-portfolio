# Geospatial and Data Science Portfolio Examples

Three small, self-contained projects. Each one demonstrates a single technique end to end, with a notebook for the analysis, a tested Python module for the logic, and a README written for a quick review.

**All data is synthetic and generated in code with fixed seeds.** The projects show how the methods behave. They are not validated analyses of real sites or satellite products, and no results here should be read as real-world findings.

| Project | Capability demonstrated | Key technical points |
|---|---|---|
| [Monte Carlo uncertainty](project-01-monte-carlo-uncertainty/) | Uncertainty propagation for a gridded estimate | Correlated sampling via Cholesky factor; independent versus fully shared spatial error; P90/P50/P10 reporting; GeoTIFF grid-alignment checks |
| [Weighted-overlay site ranking](project-02-weighted-overlay/) | Spatial multi-criteria ranking and its stability | CRS-aware distances in metres; min-max normalisation with NaN masking; weight sweep with rank-correlation and top-set overlap |
| [Unsupervised VOD clustering](project-03-vod-unsupervised/) | Clustering with leakage-safe evaluation | Spatial train/test split with buffer; scaler and PCA fitted on training data only; silhouette-based choice of k; single-band baseline; seed stability |

## Common standards

- Fixed random seeds, so every notebook regenerates its inputs and outputs exactly.
- Each project has its own `requirements.txt`, `pytest.ini` and tests for the important functions.
- Notebooks are executed before publication, and their outputs are the source of every number quoted in a README.
- Synthetic results are labelled as such. Limitations and next steps are listed in every README.

## Running

Each project is independent. From a project folder:

```bash
pip install -r requirements.txt
python -m pytest -q
cd notebooks && jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

Python 3.10 or later is required.

## Licence

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 Claudia Olivares.
