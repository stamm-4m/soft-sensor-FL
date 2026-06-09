# Stamm SDK

Lightweight Python SDK to interact with the Stamm ML registry API (FastAPI backend), focused on authentication and read-only registry discovery.

## Install

Install from PyPI:

```bash
pip install stamm-sdk
```

## Quick Start

```python
from stamm_sdk import Client

sdk = Client(base_url="http://188.245.241.118:8080")
sdk.login(email="you@example.com", password="your-password")

projects = sdk.ml.list_projects()
print(projects)

project_id = "your-project-id"

project_info = sdk.ml.get_project_info(project_id)
print(project_info)

variables = sdk.ml.get_variables(project_id)
print(variables)

models = sdk.ml.list_models(project_id)
print(models)

models_full = sdk.ml.list_models_full(project_id)
print(models_full)

sdk.logout()
sdk.close()
```

## Implemented Modules

- `stamm_sdk.core`: config and authenticated HTTP session
- `stamm_sdk.registry`: ML endpoint wrappers, artifact bundle, storage manager
- `stamm_sdk.schemas`: pydantic schemas for model/federation metadata
- `stamm_sdk.tracking`: local experiment logger with optional remote push
- `stamm_sdk.learning`: centralized/federated stubs for training orchestration

## Environment Variables

- `STAMM_API_BASE_URL`: default API base URL
- `STAMM_API_TIMEOUT`: request timeout in seconds

