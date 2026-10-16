# Model catalog — what's really loadable, and with what hyperparameters

Generated from live queries against `188.245.241.118:8080` (`GET /{project_id}/metadata/{model_id}`)
plus actually downloading and loading every artifact with `FederatedClient.load_model()`, not just
reading metadata. "Loadable" below means confirmed working in this session, not assumed from the
`algorithm` field — see the ecoli-RF finding, which looked fine on paper and wasn't.

> **Requires `scikit-learn==1.5.2` exactly.** Every sklearn artifact in this catalog (every row
> below marked sklearn) was saved with that version, and `stamm-sdk[learning]` now pins it exactly
> for that reason — see "GBM / HGB" below for what happens on a newer version. This is enforced by
> the package install, not just a note.

## How framework is detected

`APIClient.get_schemas()` no longer guesses framework from `algorithm` (that's what originally
caused the LSTM mix-up below). It reads `model_description.language` / `model_description.packages`
instead:

- any language `"R"` → `framework="r"`
- any package/class mentioning `tensorflow`/`keras` → `framework="keras"`
- any package/class mentioning `torch` → `framework="pytorch"`
- otherwise → `framework="sklearn"`

`load_model()` supports `"sklearn"`, `"pytorch"`, and `"keras"`. It raises a clear `RuntimeError` for
`"r"` only (would need `rpy2`) instead of silently trying the wrong deserializer.

## P0001 — IndPenSim (penicillin)

| slug | declared algorithm | real framework / class | `load_model()` | hyperparameters you can pass to `train()` | historical values (from registry) |
|---|---|---|---|---|---|
| `0001_python_penicillin_RF` | random_forest | sklearn `RandomForestRegressor` | ✅ works | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, ... (any `RandomForestRegressor` param) | `n_estimators=5`, `max_depth=7` |
| `0002_R_penicillin_RF` | random_forest | **R** (`ranger` package) | ❌ not supported (R) | n/a — use `predict()`, which proxies to the R API | `max_tree_depth=7`, `number_of_trees=5` |
| `0003_R_penicillin_CUBIST` | custom | **R** (`Cubist` package) | ❌ not supported (R) | n/a — `predict()` only | `committees=1`, `instance_based_corrections=3`, `min_number_instances_per_leaf=12000` |
| `0004_R_penicillin_CART` | decision_tree | **R** (`rpart` package) | ❌ not supported (R) | n/a — `predict()` only | `max_tree_depth=7` |
| `0005_python_penicillin_CART` | decision_tree | sklearn `DecisionTreeRegressor` | ✅ works | `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`, ... | `max_depth=7` |
| `0006_R_penicillin_M5` | custom | **R** (`RWeka` package) | ❌ not supported (R) | n/a — `predict()` only | `min_number_instances_per_leaf=12000` |
| `0007_python_penicillin_SVM` | svm | sklearn `SVR` | ✅ loads, but see ⚠️ below | `C`, `kernel`, `epsilon`, `gamma` | `C=1`, `kernel="rbf"`, `epsilon=0.01` |
| `0008_python_penicillin_GBM` | gradient_boosting | sklearn `GradientBoostingRegressor` | ⚠️ **only with sklearn==1.5.2** — see below | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, ... | `n_estimators=300`, `max_depth=5`, `learning_rate=0.1` |
| `0009_python_penicillin_LSTM` | rnn | **Keras/TensorFlow** `Sequential(LSTM, Dropout, BatchNormalization, Dense)` | ✅ works — see ⚠️ below | `epochs`, `lr`, `batch_size` (fixed generic loop, not passed to the model's own constructor) | `epochs=100`, `batch_size=512`, optimizer Adam `lr=0.001`, early-stopping `patience=10` |
| `0010_python_penicillin_HGB` | gradient_boosting | sklearn `HistGradientBoostingRegressor` | ⚠️ **only with sklearn==1.5.2** — see below | `max_iter` (**not** `n_estimators`), `max_depth`, `learning_rate`, `l2_regularization`, ... | recorded as `n_estimators=300` historically, which is the wrong param name for this class — real param is `max_iter` |
| `biomass_indpensim_dt_v1` | decision_tree | — | no artifact (draft) | n/a | — |
| `penicillin_titer_nn_v1` | neural_network | — | no artifact (draft, placeholder metrics) | n/a | — |
| `fed_penicillin_global` | rnn | — | no artifact (FL output placeholder) | n/a | — |

## P0002 — Bioindustry E.coli

| slug | declared algorithm | real framework / class | `load_model()` | hyperparameters | historical values |
|---|---|---|---|---|---|
| `0001_python_ecoli_CART` | decision_tree | sklearn `DecisionTreeRegressor` | ✅ works | `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion` | `max_depth=5` |
| `0002_python_ecoli_RF` | **random_forest** (declared) | sklearn, **but the real artifact is a `DecisionTreeRegressor`**, not a `RandomForestRegressor` — see 🐛 below | ✅ loads, wrong type | n/a until the data bug is fixed | metadata says `model="RandomForestRegressor"`, `number_of_trees=5` |
| `ecoli_biomass_mlp_v1` | neural_network | — | no artifact (draft, placeholder metrics) | n/a | — |
| `ecoli_nanobody_svm_v1` | svm | — | no artifact (draft, placeholder metrics) | n/a | — |
| `fed_ecoli_global` | linear_regression | — | no artifact (FL output placeholder) | n/a | — |

## Real bugs found while building this table

### 🐛 `0002_python_ecoli_RF`'s artifact doesn't match its own metadata
`load_model('P0002', '0002_python_ecoli_RF')` returns a `DecisionTreeRegressor(max_depth=5)` — the
exact same object `0001_python_ecoli_CART` loads. The metadata says `RandomForestRegressor`. The
`artifact_path` for this model is almost certainly pointing at the wrong file (or a copy-paste from
the CART entry). Worth reporting to Carlos — this is a data/registration issue, not something fixable
client-side.

### ⚠️ GBM / HGB artifacts fail to unpickle on modern scikit-learn
Both were saved with scikit-learn **1.5.2**. Loading them with scikit-learn **1.9.1** (what's
currently installed wherever we tested) raises `ModuleNotFoundError: No module named '_loss'` —
sklearn restructured its internal gradient-boosting loss module between those versions, and
`joblib`/`pickle` persistence across sklearn versions [is explicitly not
guaranteed](https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations).
RF/CART/SVR happened to still load fine (only a cosmetic `InconsistentVersionWarning`), but that's
luck, not a guarantee, for those either. **`stamm-sdk[learning]` now pins `scikit-learn==1.5.2`
exactly** (not `>=`) in `pyproject.toml`, so a plain `pip install stamm-sdk[learning]` gets the
right version automatically — this note is about *why*, not an action you need to take yourself
unless something else in your environment forces a different scikit-learn version.

### ✅ `0009_python_penicillin_LSTM` is Keras/TensorFlow, not PyTorch — now supported
`algorithm="rnn"` previously mapped straight to `framework="pytorch"` in `get_schemas()`, which would
have made `load_model()` call `torch.load()` on a `.keras` file and fail. Framework detection now
reads the real `model_description.packages` (`keras.layers.LSTM`, `tensorflow`) instead of guessing
from the algorithm name, and `load_model()`/`train()` both have a real Keras path (needs
`tensorflow>=2.16`, part of the `[learning]` extra).

Two things worth knowing if you touch this model directly:
- **Loading needs a real file on disk, not an in-memory stream** -- the native `.keras` format is a
  zip archive; `load_model()` writes the downloaded bytes to a temp file first rather than trying
  `tf.keras.models.load_model()` on a `BytesIO`, which doesn't work for this format.
- **Input shape is `(batch, 1, 8)`, not `(batch, 8)`** -- confirmed via `model.summary()` against the
  real artifact. This LSTM treats each row as a length-1 "sequence," not a real multi-step time
  window. `train()` reshapes your flat feature DataFrame into that 3D shape automatically; it raises
  a clear error instead of guessing if a *different* Keras model in this registry ever turns out to
  need `seq_len > 1` (real time-ordered sequencing isn't implemented).
- It's `kind="gradient"` once trained (Keras models expose real weight tensors via `get_weights()`),
  so it goes through the cosine-similarity Gate 2 path, not the tree-accuracy one.

### ⚠️ `0007_python_penicillin_SVM` loads, but isn't FL-ready
It loads fine as a plain `SVR`, no bundled scaler. But with the historical `kernel="rbf"`, `SVR` has
no `.coef_` (kernelized SVMs don't expose a flat weight vector — only a *linear*-kernel SVR would).
`FederatedClient.extract_update()` currently treats "no `.coef_`" as "this must be a tree ensemble"
(`kind="trees"`), which is wrong for SVR and would crash in `_aggregate_trees()` (SVR has no
`.estimators_`). **Don't run this model through the FL pipeline (`train()`/`extract_update()`
followed by aggregation) yet** — `predict()` against it directly is fine, just not the federated
training path.

## Worked example — training the one model that's fully supported today

`0001_python_penicillin_RF` is the only model in this catalog with zero caveats: real sklearn
artifact, loads cleanly on current scikit-learn, and its lack of `.coef_` correctly routes it through
the tree-ensemble FL path (which is what it actually is). See
`notebooks/federated_learning_demo.ipynb` for the full pipeline against it, and
`notebooks/model_catalog_demo.ipynb` for hyperparameter examples across every loadable model in this
table.
