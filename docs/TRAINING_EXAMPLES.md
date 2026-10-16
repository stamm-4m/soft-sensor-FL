# Training examples — Manual vs. Automatic, per model

Every model below is shown both ways:

- **Automatic (Mode B)** — `client.train(...)` does everything: loads the model, runs its training
  loop, packages the result.
- **Manual (Mode A)** — `client.load_model(...)` hands you the real object and steps back; you train
  it however you want, then call `client.extract_update(...)` once at the end.

Both arrive at the same `LocalUpdate` shape. The *code* for each is what differs by model family —
this page is the single place that shows all of them side by side.

Every example below was actually run against the live registry (`188.245.241.118:8080`) while this
page was written, not written from the docstrings alone — see each model's notes for what that
turned up.

## Setup shared by every example on this page

```python
from stamm_sdk.learning import FederatedClient
from stamm_sdk.registry.api_client import APIClient

api = APIClient(session=sdk.session)
client = FederatedClient(api)
project_id = "P0001"

# `data` stands in for your own prepared DataFrame (after §5's mapping/
# synthetic-rows steps) -- one column per feature, plus the target column.
# X/y below are just that split out, for the Manual examples.
```

---

## Random Forest — `0001_python_penicillin_RF`

The one model in this registry with zero rough edges. `RandomForestRegressor` under the hood.

`verbose` is a real sklearn constructor parameter (not something this SDK adds) -- it prints
tree-building progress while it runs. There's no `metadata["loss_history"]` to inspect afterward
though: Random Forest trees are built independently (bagging), not as a sequence that converges, so
there's nothing comparable to a loss curve to capture.

`````{tab-set}
````{tab-item} Automatic (Mode B)
```python
model_id = "0001_python_penicillin_RF"

update = client.train(
    project_id, model_id, data,
    hyperparams={
        "n_estimators": 50,
        "max_depth": 6,
        "min_samples_leaf": 2,
        "verbose": 1,
    },
    participant_id="site_A",
)
print(update.kind, update.dataset_size)   # "trees" <n>
```
````

````{tab-item} Manual (Mode A)
```python
model_id = "0001_python_penicillin_RF"
feature_names = [f.name for f in schema.features]
X, y = data[feature_names], data[schema.output_name]

loaded = client.load_model(project_id, model_id)
loaded.native_model.set_params(n_estimators=50, max_depth=6, min_samples_leaf=2, verbose=1)
loaded.native_model.fit(X, y)   # your own call -- could be a validation split, warm_start, anything

update = client.extract_update(
    loaded, loaded.native_model, dataset_size=len(X), participant_id="site_B",
)
print(update.kind)   # "trees"
```
````
`````

---

## Decision Tree / CART — `0005_python_penicillin_CART`

`DecisionTreeRegressor`. Same "no loss curve" situation as Random Forest, for the same reason --
a single tree isn't a converging sequence either.

`````{tab-set}
````{tab-item} Automatic (Mode B)
```python
model_id = "0005_python_penicillin_CART"

update = client.train(
    project_id, model_id, data,
    hyperparams={"max_depth": 6, "min_samples_leaf": 3, "criterion": "squared_error"},
    participant_id="demo",
)
```
````

````{tab-item} Manual (Mode A)
```python
model_id = "0005_python_penicillin_CART"
feature_names = [f.name for f in schema.features]
X, y = data[feature_names], data[schema.output_name]

loaded = client.load_model(project_id, model_id)
loaded.native_model.set_params(max_depth=6, min_samples_leaf=3, criterion="squared_error")
loaded.native_model.fit(X, y)

update = client.extract_update(loaded, loaded.native_model, dataset_size=len(X), participant_id="demo")
```
````
`````

---

## SVM — `0007_python_penicillin_SVM`

`SVR`. Trains and loads fine either way -- but with the registry's own `kernel="rbf"`, the fitted
`SVR` has no `.coef_` (only a *linear*-kernel SVR would), so `extract_update()` tags it
`kind="trees"`, which is **wrong** for an SVM.

```{important}
Don't feed this model's `LocalUpdate` into `FederatedAggregator.aggregate()` yet -- it would crash
looking for `.estimators_`, which `SVR` doesn't have. Training and `predict()` are both fine; only
the aggregation step isn't ready for this model class (see `MODEL_CATALOG.md`).
```

`````{tab-set}
````{tab-item} Automatic (Mode B)
```python
model_id = "0007_python_penicillin_SVM"

update = client.train(
    project_id, model_id, data,
    hyperparams={"C": 2.0, "kernel": "rbf", "epsilon": 0.01, "gamma": "scale"},
    participant_id="demo",
)
```
````

````{tab-item} Manual (Mode A)
```python
model_id = "0007_python_penicillin_SVM"
feature_names = [f.name for f in schema.features]
X, y = data[feature_names], data[schema.output_name]

loaded = client.load_model(project_id, model_id)
loaded.native_model.set_params(C=2.0, kernel="rbf", epsilon=0.01, gamma="scale")
loaded.native_model.fit(X, y)

update = client.extract_update(loaded, loaded.native_model, dataset_size=len(X), participant_id="demo")
```
````
`````

---

## Gradient Boosting — `0008_python_penicillin_GBM`

`GradientBoostingRegressor`. This artifact was saved with scikit-learn 1.5.2 and hard-fails to
unpickle on newer versions (`ModuleNotFoundError: No module named '_loss'`) -- `stamm-sdk[learning]`
pins `scikit-learn==1.5.2` exactly for this reason, so this just works if you installed the extra
normally.

Unlike the bagging models above, boosting really is a sequence of stages that converges: `verbose=1`
prints real `Iter / Train Loss / Remaining Time` progress, **and** `update.metadata["loss_history"]`
comes back populated afterward from sklearn's own `train_score_` (the per-stage training deviance --
the real equivalent of a Keras loss curve, from a completely different framework).

`````{tab-set}
````{tab-item} Automatic (Mode B)
```python
model_id = "0008_python_penicillin_GBM"

update = client.train(
    project_id, model_id, data,
    hyperparams={"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05,
                 "subsample": 0.8, "verbose": 1},
    participant_id="demo",
)
print(update.metadata["loss_history"][:3], "...", update.metadata["loss_history"][-3:])
```
````

````{tab-item} Manual (Mode A)
```python
model_id = "0008_python_penicillin_GBM"
feature_names = [f.name for f in schema.features]
X, y = data[feature_names], data[schema.output_name]

loaded = client.load_model(project_id, model_id)
loaded.native_model.set_params(
    n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, verbose=1,
)
loaded.native_model.fit(X, y)

# train_score_ is the same per-stage deviance train() captures automatically
# in Mode B -- in Mode A you read it yourself if you want it.
loss_history = list(loaded.native_model.train_score_)

update = client.extract_update(
    loaded, loaded.native_model, dataset_size=len(X), participant_id="demo",
    metadata={"loss_history": loss_history},
)
```
````
`````

```{note}
`0010_python_penicillin_HGB` is the same family (`HistGradientBoostingRegressor`) but has **no
`n_estimators`** -- the equivalent there is `max_iter`. Everything else on this page about GBM
(the sklearn pin, the `verbose`/`loss_history` behavior) applies to it too.
```

---

## LSTM (Keras/TensorFlow) — `0009_python_penicillin_LSTM`

A real Keras model, not PyTorch -- `algorithm="rnn"` looks like it should mean PyTorch, but the real
declared packages (`keras.layers.LSTM`, `tensorflow`) say otherwise (confirmed via
`schema.framework == "keras"`). Needs the `[learning]` extra's `tensorflow` dependency.

```{important}
**The 3D reshape is only automatic in Mode B.** `train()` reshapes your flat `(rows, features)`
data into whatever shape the model's own `input_shape` declares (confirmed `(batch, 1, 8)` against
the real artifact) *before* calling `model.fit()`. In Mode A, `load_model()` hands you the raw
Keras model and does **not** reshape anything for you -- you do it yourself, as shown in the Manual
tab below. Forgetting this is the most likely way to hit a shape error training this model by hand.
```

`````{tab-set}
````{tab-item} Automatic (Mode B)
```python
model_id = "0009_python_penicillin_LSTM"

update = client.train(
    project_id, model_id, data,
    hyperparams={"epochs": 3, "lr": 0.001, "batch_size": 8, "verbose": 1},
    participant_id="demo",
)
print(update.kind)                           # "gradient" -- Keras exposes real weight tensors
print(len(update.payload))                   # 14 weight tensors
print(update.metadata["loss_history"])       # per-epoch loss, captured regardless of verbose
```
````

````{tab-item} Manual (Mode A)
```python
import tensorflow as tf

model_id = "0009_python_penicillin_LSTM"
feature_names = [f.name for f in schema.features]
X, y = data[feature_names], data[schema.output_name]

loaded = client.load_model(project_id, model_id)

# train() does this reshape for you in Mode B -- by hand, you need it
# yourself: this model's input_shape is (None, 1, 8), one timestep per row.
X_arr = X.to_numpy(dtype="float32").reshape(-1, 1, len(feature_names))
y_arr = y.to_numpy(dtype="float32")

loaded.native_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
history = loaded.native_model.fit(X_arr, y_arr, epochs=3, batch_size=8, verbose=1)

update = client.extract_update(
    loaded, loaded.native_model, dataset_size=len(X), participant_id="demo",
    metadata={"loss_history": history.history["loss"]},
)
print(update.kind)   # "gradient"
```
````
`````

---

## PyTorch

```{warning}
**Unverified against a real model.** There's no live PyTorch artifact in this registry today --
the one model whose `algorithm` suggested PyTorch (`0009_python_penicillin_LSTM`) turned out to be
Keras instead (see above). The code path below exists and mirrors the Keras one, but hasn't been
run against a real `torch.nn.Module` from this registry. Treat it as the intended shape, not a
confirmed-working example, until a real PyTorch model shows up in the catalog.
```

`````{tab-set}
````{tab-item} Automatic (Mode B)
```python
update = client.train(
    project_id, model_id, data,
    hyperparams={"epochs": 10, "lr": 1e-3, "batch_size": 32, "verbose": True},
    participant_id="demo",
)
# update.metadata["loss_history"] -- mean loss per epoch, same as the Keras path
```
````

````{tab-item} Manual (Mode A)
```python
import torch

loaded = client.load_model(project_id, model_id)
feature_names = [f.name for f in schema.features]
X, y = data[feature_names], data[schema.output_name]

X_t = torch.tensor(X.to_numpy(dtype="float32"))
y_t = torch.tensor(y.to_numpy(dtype="float32")).reshape(-1, 1)

optimizer = torch.optim.Adam(loaded.native_model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()
loaded.native_model.train()
for _ in range(10):
    optimizer.zero_grad()
    loss = loss_fn(loaded.native_model(X_t), y_t)
    loss.backward()
    optimizer.step()

update = client.extract_update(
    loaded, loaded.native_model, dataset_size=len(X), participant_id="demo",
)
```
````
`````

---

## R models — no training path, `predict()` only

`0002_R_penicillin_RF`, `0003_R_penicillin_CUBIST`, `0004_R_penicillin_CART`,
`0006_R_penicillin_M5` all report `schema.framework == "r"`. `load_model()` refuses immediately and
clearly (would need `rpy2`, not attempted by this SDK) -- there's no Manual/Automatic split to show
for these, only inference:

```python
r_schema = sdk.ml.get_schemas(project_id, "0002_R_penicillin_RF")
one_row = {f.name: (f.range_min + f.range_max) / 2 for f in r_schema.features}
prediction = sdk.ml.predict(project_id, "0002_R_penicillin_RF", one_row)
```
