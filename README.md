# Stamm SDK

Python client SDK for the Stamm ML registry (FastAPI backend): authentication, read-only model
discovery, and a complete federated-learning pipeline -- data prep, two independent validation
gates, training (your own loop or ours), and aggregation -- for the models hosted there.

**Full documentation, a step-by-step guide, and per-model training examples:**
[stamm-4m.github.io/soft-sensor-FL](https://stamm-4m.github.io/soft-sensor-FL/)

## Install

```bash
pip install stamm-sdk              # auth + read-only discovery
pip install "stamm-sdk[learning]"  # + training (scikit-learn, torch, tensorflow)
```

`[learning]` pins `scikit-learn==1.5.2` exactly -- every sklearn model artifact in the registry's
catalog was saved with that version, and newer sklearn fails to unpickle some of them.

## Quick Start

```python
from stamm_sdk import Client

sdk = Client(base_url="http://188.245.241.118:8080")
sdk.login(email="you@example.com", password="your-password")

projects = sdk.ml.list_projects()
models = sdk.ml.list_models(project_id="your-project-id")

sdk.logout()
sdk.close()
```

For the federated-learning pipeline (discover a model, prepare data, validate, train, aggregate),
see the [full guide](https://stamm-4m.github.io/soft-sensor-FL/SDK_GUIDE.html) or run
[`notebooks/complete_guide.ipynb`](notebooks/complete_guide.ipynb) end to end against a live
registry.

## Implemented Modules

- `stamm_sdk.core`: config and authenticated HTTP session
- `stamm_sdk.registry`: ML endpoint wrappers, artifact bundle, storage manager
- `stamm_sdk.schemas`: pydantic schemas for model/federation metadata
- `stamm_sdk.tracking`: local experiment logger with optional remote push
- `stamm_sdk.data`: feature-mapping and synthetic-data helpers for FL data prep
- `stamm_sdk.validation`: Gate 1 (raw-data: range + PSI drift) and Gate 2 (post-training:
  cosine-similarity collusion detection for gradient models, accuracy-outlier detection for trees)
- `stamm_sdk.learning`: `FederatedClient` (load/train/extract a local update, Manual or Automatic
  mode), `FederatedAggregator` (FedAvg for gradient models, tree-ensemble merge for Random
  Forest/CART), `run_federation_round` (end-to-end orchestration)

## Environment Variables

- `STAMM_API_BASE_URL`: default API base URL
- `STAMM_API_TIMEOUT`: request timeout in seconds

