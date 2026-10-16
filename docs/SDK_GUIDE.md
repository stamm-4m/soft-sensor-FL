# Stamm SDK — Complete Guide

This is the full reference for `stamm-sdk`: what every module does, which imports you need for
which task, and a worked path from "just installed it" to "ran a full federated-learning round."
If you want copy-pasteable code instead of prose, see `notebooks/complete_guide.ipynb` — it's the
hands-on companion to this document and runs every step below against the real registry.

For a model-by-model breakdown of what's actually trainable today (and what isn't, and why), see
`docs/MODEL_CATALOG.md`. This guide is about the SDK itself; that one is about the data you'll point
it at.

---

## 1. What this SDK is

A Python client for the Stamm ML registry (a FastAPI backend that stores bioprocess soft-sensor
models, their metadata, and federated-learning bookkeeping). It does three things:

1. **Talks to the registry over HTTP** — auth, discovering projects/models, reading a model's
   feature schema, downloading its trained artifact, writing predictions back.
2. **Prepares your local data** to match what a given model expects — renaming columns, validating
   it, padding it out with synthetic rows if you don't have enough.
3. **Runs (or helps you run) the federated-learning pipeline** — local training, two independent
   validation gates before anything gets trusted, and aggregation into a new global model.

Nothing in this SDK invents a model from scratch. Every training path starts from a model that
already exists in the registry.

---

## 2. Installation

```bash
pip install stamm-sdk
```

That's enough for authentication and read-only registry discovery (`Client`, `sdk.ml.list_*`,
`sdk.ml.get_schemas`, `sdk.ml.predict`). It pulls in `httpx` and `pydantic` — intentionally
lightweight, nothing ML-related.

Training needs the optional `learning` extra, which adds `scikit-learn`, `torch`, `tensorflow`, and
`joblib`:

```bash
pip install "stamm-sdk[learning]"
```

Without it, calling `FederatedClient.load_model()` or `.train()` raises a clear `RuntimeError`
telling you to install the extra — it doesn't fail with a confusing `ImportError` buried in a
traceback, because both heavy imports are deferred to inside the functions that actually need them.

> **scikit-learn must be exactly `1.5.2`.** The extra pins it (`scikit-learn==1.5.2`, not `>=`) —
> every sklearn artifact in the live registry's catalog was saved with that version, and scikit-learn
> doesn't guarantee model persistence across versions (its own docs say so). This isn't theoretical:
> the GBM and HistGradientBoosting models in this catalog hard-fail
> (`ModuleNotFoundError: No module named '_loss'`) on scikit-learn 1.9.1 — confirmed live. Other
> models (RandomForest, DecisionTree, SVR) happened to still load on 1.9.1 with only a cosmetic
> `InconsistentVersionWarning`, but that's luck, not something to rely on. If something else in your
> environment needs a different scikit-learn version, isolate this SDK's `[learning]` install into
> its own virtualenv rather than loosening the pin.

---

## 3. Quick start — authenticate

```python
from stamm_sdk import Client

sdk = Client(base_url="http://188.245.241.118:8080")   # or STAMM_API_BASE_URL env var
sdk.login(email="you@example.com", password="your-password")

# ... do things ...

sdk.logout()
sdk.close()
```

`Client` is a context manager too:

```python
with Client(base_url="...") as sdk:
    sdk.login(email=..., password=...)
    ...
```

Everything registry-related hangs off `sdk.ml`, which is an `APIClient` instance. `sdk.session` is
the underlying authenticated HTTP session — you'll need it directly when constructing an
`APIClient`/`FederatedClient` pair for training (see §6).

---

## 4. Discovering what you can work with

```python
from stamm_sdk.registry.api_client import APIClient   # only needed if building APIClient by hand;
                                                         # sdk.ml already *is* one

# Every model across every project you belong to (server-side lab-filtered, no args needed):
models = sdk.ml.list_available_models()
# -> [{"project_id": "P0001", "slug": "0001_python_penicillin_RF",
#      "name": "Random Forest", "algorithm": "random_forest", "status": "deployed"}, ...]

# One model's full feature schema, normalized regardless of how the registry's YAML stored it:
schema = sdk.ml.get_schemas("P0001", "0001_python_penicillin_RF")
schema.algorithm     # "random_forest"
schema.framework      # "sklearn" | "pytorch" | "keras" | "r"  -- see §6 for what each means for training
schema.output_name    # "penicillin_concentration"
schema.features       # list[FeatureSpec]: name, type, lag, required, range_min, range_max
```

`framework` is detected from the model's own declared packages/language, not guessed from the
algorithm name — `algorithm="rnn"` turned out to mean Keras for one real model in this registry, not
PyTorch, and guessing wrong would have made `load_model()` call the wrong deserializer. See
`docs/MODEL_CATALOG.md` for exactly which models work with which framework today.

Other read endpoints, useful for exploring a project before you commit to a model:

```python
sdk.ml.get_project_info("P0001")
sdk.ml.get_variables("P0001")
sdk.ml.get_references("P0001")
sdk.ml.list_models_full("P0001")     # richer per-model listing than list_available_models()
```

---

## 5. Preparing your data

### 5.1 Map your columns onto the model's feature names

```python
from stamm_sdk.data import map_input_columns, validate_mapped_data

my_raw_data = pd.DataFrame({"T_sensor": [...], "pH_reading": [...]})   # your own column names

column_map = {"T_sensor": "temperature", "pH_reading": "pH"}
mapped = map_input_columns(my_raw_data, schema, column_map)

report = validate_mapped_data(mapped, schema)
report.passed             # bool
report.missing_features   # list[str] -- required features that never got mapped
report.type_mismatches    # list[str]
```

`map_input_columns` never guesses or fuzzy-matches — if you don't map a column, it stays unmapped,
and `validate_mapped_data` will tell you exactly what's missing instead of silently training on an
incomplete feature set.

### 5.2 Not enough rows? Generate synthetic ones

```python
from stamm_sdk.data import generate_synthetic_rows

synthetic = generate_synthetic_rows(schema, mapped, target_n=200, seed=42)
# -> only the NEW rows, uniform-random within each feature's real [range_min, range_max],
#    each tagged is_synthetic=True so they stay traceable

combined = pd.concat([mapped.assign(is_synthetic=False), synthetic], ignore_index=True)
```

If a feature has no catalog range, this falls back to your real data's own min/max; if there's
neither, it raises rather than inventing a bound from nothing.

---

## 6. Training

Both modes start from a model that already exists in the registry — the SDK never invents an
architecture — and both go through the same `FederatedClient`:

```python
from stamm_sdk.learning import FederatedClient
from stamm_sdk.registry.api_client import APIClient

api = APIClient(session=sdk.session)      # wraps the same authenticated session as sdk.ml
client = FederatedClient(api)
```

- **Automatic (Mode B)** — `client.train(project_id, model_id, data, hyperparams=...)` loads the
  model and runs its training loop entirely inside the SDK. Simplest path; `hyperparams` means
  different things depending on the model's real class (see §6 of
  **[Training examples](TRAINING_EXAMPLES.md)** for exactly which keys are valid for each one).
- **Manual (Mode A)** — `client.load_model(project_id, model_id)` hands you the real, deserialized
  object (a fitted sklearn estimator, a `torch.nn.Module`, a Keras model) and steps back entirely;
  you train it however you want, then call `client.extract_update(...)` once to package your result
  into the same shape Gate 2/aggregation expect.

`load_model()`/`train()` raise `RuntimeError` immediately (not a confusing crash three calls later)
if the model has no artifact registered yet, or if `schema.framework == "r"` (not implemented --
use `sdk.ml.predict()` instead). `"sklearn"`, `"pytorch"`, and `"keras"` are all real, working
paths.

**→ For a complete, runnable example of both modes against every supported model class --
Random Forest, Decision Tree, SVM, Gradient Boosting, HistGradientBoosting, and the real Keras
LSTM in this registry -- see [Training examples](TRAINING_EXAMPLES.md).** That page is organized
as tabs (Manual / Automatic) per model, each one confirmed against the live registry, including the
caveats that don't apply uniformly (SVM's aggregation gap, the exact scikit-learn pin GBM/HGB need,
Keras's 3D reshape, PyTorch being implemented but unverified against a real artifact).

---

## 7. Validating before trusting anything (the two gates)

### Gate 1 — raw data, before training

```python
from stamm_sdk.validation import validate_raw_data

report = validate_raw_data(combined, schema, reference=None)   # reference is optional
report.passed
report.out_of_range    # {feature_name: count of rows outside its physical range}
report.psi_scores      # {feature_name: PSI vs. reference} -- only computed if you pass a reference DataFrame
report.reasons
```

Out-of-range checking always runs. PSI (population-stability-index drift detection) only runs when
you pass a `reference` DataFrame of known-good prior data — without one, that part is simply skipped,
not assumed to pass.

### Gate 2 — the trained result, before aggregating

Dispatches on `update.kind`:

```python
from stamm_sdk.validation import validate_gradient_update, validate_tree_update

# kind == "gradient" (linear models, PyTorch):
report = validate_gradient_update(update, round_updates, history)
report.passed
report.norm                # ||update|| -- flagged if outside median +/- 3*MAD across this round's updates
report.cosine_to_peers      # {participant_id: cosine_similarity} -- near-1.0 means likely collusion (FoolsGold-style)
report.drift_vs_history     # cosine similarity vs. the previous accepted global model

# kind == "trees" (Random Forest / other sklearn ensembles):
report = validate_tree_update(update, round_updates, reference_data, output_name)
report.passed
report.accuracy             # this update's score() against reference_data
report.accuracy_vs_round    # z-score vs. this round's other participants -- flags outliers in EITHER direction
                              # (suspiciously bad AND suspiciously perfect both get flagged)
```

Why two different gates: cosine similarity on a weight vector only makes sense for models with a
fixed-size parameter vector. A Random Forest has no such vector to compare — the tree path instead
scores each contribution's actual predictive performance against a shared reference set.

`reference_data` for the tree path is filtered down to the model's own `feature_names_in_`
internally — safe to pass a DataFrame with extra bookkeeping columns (`is_synthetic`, timestamps,
whatever); they're ignored rather than causing a sklearn "feature names don't match" error.

---

## 8. Aggregating a round

```python
from stamm_sdk.learning import FederatedAggregator

result = FederatedAggregator().aggregate(
    [update_site_a, update_site_b, update_attacker],   # all same `.kind`
    history=[],                                          # previous AggregationResults, oldest first
    reference_data=combined,     # required for kind=="trees"; ignored for "gradient"
    output_name=output_name,     # required for kind=="trees"
)

result.accepted_participants      # who passed Gate 2
result.rejected_participants      # {participant_id: reason string}
result.global_payload             # the new global model: a weighted-mean dict (gradient) or a merged forest (trees)
result.performance_vs_previous    # trees only -- new global model's score() on reference_data
result.direction_ok               # sanity check: new result isn't a regression/divergence vs. history
```

Gradient path: weighted mean of accepted updates' payloads (weight = `dataset_size` share), plus a
direction check against the previous round's global payload.

Trees path: merges each accepted participant's own fitted trees into one bigger forest, each
contributing a `dataset_size`-proportional share of their trees — not weight averaging, since there's
no equivalent of "averaging" two separately-grown decision trees.

Raises `RuntimeError` if every update in the round failed Gate 2 — aggregation never silently
produces a result from zero trusted inputs.

---

## 9. Running + persisting a full round

```python
from stamm_sdk.learning import run_federation_round

result = run_federation_round(
    api, federation_id="<uuid of a Federation row already created>",
    round_number=1,
    updates=[update_site_a, update_site_b],
    history=[],
    reference_data=combined, output_name=output_name,
    global_model_id=None,   # pass the new soft_sensors.id once you've registered the aggregated
                              # model, so the persisted round/contribution rows point at it
)
```

This runs `FederatedAggregator().aggregate()` and then writes the outcome back to the registry:
one `federation_rounds` row, plus one `model_contributions` row per accepted participant (weighted by
their share of the accepted dataset size). It does **not** register the aggregated model itself as a
new `soft_sensors` row — that's a separate decision (new version vs. in-place update) left to you;
pass its id in as `global_model_id` once you've made that call, or omit it to persist the round
"unpublished" for inspection first.

A `Federation` row has to exist first (plain CRUD, same pattern as everything else):

```python
federation = api.session.request("POST", "/api/v1/federations/", json={
    "slug": "penicillin_titer_demo", "name": "Penicillin titer demo federation",
    "aggregation_strategy": "fedavg_trees",
})
```

---

## 10. Import reference — "I want to do X, what do I import?"

| I want to... | Import |
|---|---|
| Connect and log in | `from stamm_sdk import Client` |
| List/discover models, get a schema, download an artifact, raw CRUD calls | `sdk.ml` (already an `APIClient`) — or `from stamm_sdk.registry.api_client import APIClient` to build one standalone |
| Typed feature/model schema objects | `from stamm_sdk.schemas import FeatureSpec, ModelSchema` |
| Rename columns / check they're all there | `from stamm_sdk.data import map_input_columns, validate_mapped_data` |
| Generate synthetic training rows | `from stamm_sdk.data import generate_synthetic_rows` |
| Gate 1 (raw data validation + PSI) | `from stamm_sdk.validation import validate_raw_data, compute_psi` |
| Gate 2, gradient-capable models | `from stamm_sdk.validation import validate_gradient_update, compute_cosine_similarity, compute_norm_bound` |
| Gate 2, tree ensembles | `from stamm_sdk.validation import validate_tree_update` |
| Load/train a registry model | `from stamm_sdk.learning import FederatedClient` |
| Aggregate a round | `from stamm_sdk.learning import FederatedAggregator` |
| Run + persist a full round in one call | `from stamm_sdk.learning import run_federation_round` |
| The data contracts (`LocalUpdate`, `AggregationResult`, `LoadedModel`) | `from stamm_sdk.fl_types import LocalUpdate, AggregationResult, LoadedModel` (or `from stamm_sdk.learning import ...`, re-exported) |
| Local experiment logging | `from stamm_sdk.tracking import ExperimentLogger` |
| Package a model + metadata for local storage | `from stamm_sdk.registry import ArtifactBundle, StorageManager` |

---

## 11. Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `STAMM_API_BASE_URL` | Default registry URL if you don't pass `base_url=` to `Client()` | `http://188.245.241.118:8080` |
| `STAMM_API_TIMEOUT` | Request timeout (seconds) | `30.0` |

---

## 12. What's not implemented yet

- `CentralizedTrainer` (`stamm_sdk.learning.centralized_trainer`) — still a literal stub.
- R models can't be `load_model()`-ed (clear error, not a crash) — use `predict()` for inference on
  those.
- Keras models only support `seq_len == 1` (one timestep per row) in `train()` — a real
  multi-timestep sequence model would need genuine time-ordered data grouping, which
  `generate_synthetic_rows`/`map_input_columns` don't produce; `train()` raises clearly rather than
  guessing if it ever sees `seq_len > 1`.
- `FederatedAggregator` only has the two paths described above (gradient, trees) — kernelized SVMs
  and anything else without `.coef_`/`get_weights()` or `.estimators_` aren't covered by either yet
  (see `docs/MODEL_CATALOG.md` §3 for the specific SVM case this bit us on).
- There's no live PyTorch model in this registry today to verify the `"pytorch"` path against — it's
  implemented and should work the same way the Keras path does, just unconfirmed against a real
  artifact.

See `docs/MODEL_CATALOG.md` for exactly which real models in the registry hit which of these limits
today.
