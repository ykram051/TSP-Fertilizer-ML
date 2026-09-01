# 08 — Dependency manifest

The model-serving dependencies below were read from the Python environment that
loaded and executed the saved research bundles:

| Package | Version |
|---|---:|
| Python | 3.11 |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |
| LightGBM | 4.6.0 |
| PyYAML | 6.0.1 |

`asyncua 1.1.8` is introduced only for the laboratory OPC UA integration; it
was not involved in model training. Before plant installation, the built image
must be vulnerability-scanned and approved by IT/OT.
