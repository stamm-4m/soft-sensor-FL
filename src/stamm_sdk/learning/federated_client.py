"""Load an existing model from the registry and produce a LocalUpdate
from it, either by training it yourself (Mode A) or letting the SDK run
the whole local-training loop (Mode B).

Both modes start from the same registry-hosted model -- the SDK never
invents a model architecture; it only loads/trains what's already
registered (see load_model()). torch/scikit-learn/tensorflow are
optional dependencies (`pip install stamm-sdk[learning]`) -- all three
are imported lazily, only when a method that actually needs them is
called, so the read-only discovery part of the SDK stays lightweight.

Supported frameworks today: "sklearn", "pytorch", "keras". "r" is
detected but not loadable (no rpy2) -- use predict() for inference on
those instead.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd

from stamm_sdk.registry.api_client import APIClient
from stamm_sdk.fl_types import LoadedModel, LocalUpdate

_LEARNING_EXTRA_HINT = "pip install 'stamm-sdk[learning]'"


def _flatten_state(native_model: Any, framework: str) -> dict[str, Any]:
    """Pull a named {param_name: array} snapshot out of a loaded model,
    for the weight-delta math in extract_update(). PyTorch modules
    expose state_dict(); Keras models expose get_weights() (a plain list,
    so we index it); sklearn linear models expose .coef_/.intercept_.
    Returns {} for anything else (tree ensembles) -- extract_update()
    reads that as "no weight vector, this is a 'trees' update" instead.
    """
    if framework == "pytorch":
        return {k: v.detach().cpu().numpy() for k, v in native_model.state_dict().items()}
    if framework == "keras":
        return {f"w{i}": w for i, w in enumerate(native_model.get_weights())}
    if hasattr(native_model, "coef_"):
        state: dict[str, Any] = {"coef_": native_model.coef_}
        if hasattr(native_model, "intercept_"):
            state["intercept_"] = native_model.intercept_
        return state
    return {}


class FederatedClient:
    def __init__(self, api: APIClient) -> None:
        self.api = api

    def load_model(self, project_id: str, model_id: str) -> LoadedModel:
        """Fetch the model's schema + trained artifact from the registry
        and deserialize it into a ready-to-use native object (a
        torch.nn.Module instance, or a fitted sklearn estimator).

        Raises RuntimeError if the model has no artifact registered yet
        -- a legitimate, reachable "draft" state (see penicillin_titer_nn_v1
        in our catalog) -- there's nothing to load in that case.

        Convention assumed for pytorch artifacts: saved as the full
        module via torch.save(model), not just a state_dict -- a bare
        state_dict can't be loaded without the class definition, which
        the registry doesn't have.
        """
        schema = self.api.get_schemas(project_id, model_id)
        row = self.api.get_model_metadata(project_id, model_id)
        # FIX: get_model_metadata() is the YAML-backed metadata shape, not
        # the flat soft_sensors DB row -- there's no top-level
        # "artifact_path" field here. The artifact's filename (when one
        # exists) lives at model_description.config_files.model_file,
        # confirmed against the live registry (e.g. "0001_[python]_penicillin_RF.pkl").
        config_files = (row.get("model_description") or {}).get("config_files") or {}
        if not config_files.get("model_file"):
            raise RuntimeError(
                f"model {model_id!r} has no trained artifact yet "
                f"(model_description.config_files.model_file is empty) -- nothing to load"
            )

        framework = schema.framework or "sklearn"
        if framework == "r":
            raise RuntimeError(
                f"model {model_id!r} uses framework='r', which this SDK can't "
                f"load locally (would need rpy2) -- use predict() instead, "
                f"which proxies through the registry's own R runtime"
            )

        artifact_bytes = self.api.get_model_artifact(project_id, model_id)

        if framework == "pytorch":
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    f"load_model() for a pytorch model needs torch installed -- {_LEARNING_EXTRA_HINT}"
                ) from exc
            native_model = torch.load(BytesIO(artifact_bytes), weights_only=False)
        elif framework == "keras":
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise RuntimeError(
                    f"load_model() for a keras model needs tensorflow installed -- {_LEARNING_EXTRA_HINT}"
                ) from exc
            # Keras's native ".keras" format is a zip archive -- loading
            # reliably needs a real file path, not an in-memory stream.
            # Confirmed against the real artifact (0009_python_penicillin_LSTM):
            # tf.keras.models.load_model() on a BytesIO is not supported
            # for this format, so write to a temp file first.
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
                tmp.write(artifact_bytes)
                tmp.flush()
                native_model = tf.keras.models.load_model(tmp.name)
        else:
            # FIX: sklearn models in this registry are saved with joblib,
            # not plain pickle -- confirmed against a real artifact
            # (0001_python_penicillin_RF.pkl): plain pickle.loads() didn't
            # error, it just silently returned the wrong object (an
            # internal numpy array from partway through the stream)
            # instead of the RandomForestRegressor. joblib.load() handles
            # the same stream correctly regardless of the ".pkl" extension
            # -- sklearn's own docs recommend joblib for persistence.
            try:
                import joblib
            except ImportError as exc:
                raise RuntimeError(
                    f"load_model() for a sklearn model needs joblib installed -- {_LEARNING_EXTRA_HINT}"
                ) from exc
            native_model = joblib.load(BytesIO(artifact_bytes))

        return LoadedModel(
            model_id=model_id, project_id=project_id, schema=schema,
            framework=framework, native_model=native_model,
            initial_payload=_flatten_state(native_model, framework),
        )

    def extract_update(self, loaded: LoadedModel, trained_model: Any, dataset_size: int,
                        round_number: int = 0, participant_id: str = "local",
                        metadata: dict[str, Any] | None = None) -> LocalUpdate:
        """Mode A: wrap a model the USER trained themselves (their own
        PyTorch loop, or a direct sklearn .fit() call, run outside the
        SDK) into the same LocalUpdate shape Gate 2 / aggregation expect.

        For weight-capable models (pytorch / linear), payload is the
        WEIGHT DELTA (trained - initial) -- that's what FedAvg /
        cosine-similarity math operates on, not the raw trained weights.
        For tree ensembles there's no delta concept, so payload is the
        trained model itself and kind is "trees" (detected by
        loaded.initial_payload being empty -- see _flatten_state).

        metadata is yours to use for whatever you want to carry alongside
        the update (e.g. a training loss curve) -- train()'s Mode B path
        populates it with one for Keras/PyTorch (see _train_keras/
        _train_pytorch's `history` return value); nothing reads it back
        downstream today, it's just attached to the LocalUpdate for your
        own inspection/logging.
        """
        if not loaded.initial_payload:
            return LocalUpdate(
                participant_id=participant_id, round_number=round_number,
                dataset_size=dataset_size, kind="trees", payload=trained_model,
                metadata=metadata or {},
            )

        trained_state = _flatten_state(trained_model, loaded.framework)
        delta = {
            k: trained_state[k] - loaded.initial_payload[k]
            for k in trained_state if k in loaded.initial_payload
        }
        return LocalUpdate(
            participant_id=participant_id, round_number=round_number,
            dataset_size=dataset_size, kind="gradient", payload=delta,
            metadata=metadata or {},
        )

    def train(self, project_id: str, model_id: str, data: pd.DataFrame,
              hyperparams: dict[str, Any] | None = None,
              round_number: int = 0, participant_id: str = "local",
              output_name: str | None = None) -> LocalUpdate:
        """Mode B: load the model and run its local-training loop
        entirely inside the SDK.

        hyperparams, by framework:
          pytorch:  epochs (default 10), lr (default 1e-3), batch_size (default 32),
                    verbose (default False) -- prints loss per epoch when True
          keras:    epochs (default 10), lr (default 1e-3), batch_size (default 32),
                    verbose (default 0) -- passed straight to Keras's own
                    model.fit(verbose=...): 0 silent, 1 progress bar, 2 one line/epoch
          sklearn:  passed straight to estimator.set_params(**hyperparams)

        For keras/pytorch, the returned LocalUpdate.metadata["loss_history"]
        is a list of per-epoch loss values, regardless of verbose -- useful
        for plotting convergence even when you didn't want console output.

        output_name overrides loaded.schema.output_name -- get_schemas()
        can only infer the target column when the registry's own
        soft_sensors.outputs is a plain list of strings; pass this
        explicitly for any model where that didn't resolve (checked
        below, instead of letting .fit() fail three layers down with a
        confusing "y cannot be None").
        """
        loaded = self.load_model(project_id, model_id)
        hyperparams = hyperparams or {}
        feature_names = [f.name for f in loaded.schema.features]
        resolved_output = output_name or loaded.schema.output_name
        if not resolved_output:
            raise ValueError(
                f"model {model_id!r} has no output_name in its registry metadata -- "
                f"pass output_name= explicitly"
            )
        X = data[feature_names]
        y = data[resolved_output]

        loss_history: list[float] = []
        if loaded.framework == "pytorch":
            trained_model, loss_history = self._train_pytorch(loaded.native_model, X, y, hyperparams)
        elif loaded.framework == "keras":
            trained_model, loss_history = self._train_keras(loaded.native_model, X, y, hyperparams)
        else:
            trained_model = loaded.native_model
            if hyperparams:
                # sklearn's own estimators already support "verbose" as a
                # real constructor param on several classes (e.g.
                # GradientBoostingRegressor, RandomForestRegressor) -- no
                # special-casing needed, it just flows through set_params()
                # like any other hyperparameter.
                trained_model.set_params(**hyperparams)
            trained_model.fit(X, y)
            # GradientBoostingRegressor (and similar staged/boosting
            # estimators) record training deviance per boosting stage in
            # train_score_ -- the closest sklearn equivalent to an
            # epoch-by-epoch loss curve. Bagging estimators (Random
            # Forest) have no equivalent: trees are built independently,
            # not as a sequence that converges, so there's nothing
            # comparable to capture for those.
            if hasattr(trained_model, "train_score_"):
                loss_history = list(trained_model.train_score_)

        return self.extract_update(
            loaded, trained_model, dataset_size=len(data),
            round_number=round_number, participant_id=participant_id,
            metadata={"loss_history": loss_history} if loss_history else {},
        )

    def _train_pytorch(self, model: Any, X: pd.DataFrame, y: pd.Series,
                        hyperparams: dict[str, Any]) -> tuple[Any, list[float]]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                f"train() for a pytorch model needs torch installed -- {_LEARNING_EXTRA_HINT}"
            ) from exc

        epochs = hyperparams.get("epochs", 10)
        lr = hyperparams.get("lr", 1e-3)
        batch_size = hyperparams.get("batch_size", 32)
        verbose = hyperparams.get("verbose", False)

        X_t = torch.tensor(X.to_numpy(dtype="float32"))
        y_t = torch.tensor(y.to_numpy(dtype="float32")).reshape(-1, 1)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        model.train()
        loss_history: list[float] = []
        for epoch in range(epochs):
            epoch_losses = []
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            loss_history.append(mean_loss)
            if verbose:
                print(f"  epoch {epoch + 1}/{epochs} -- loss: {mean_loss:.4f}")
        return model, loss_history

    def _train_keras(self, model: Any, X: pd.DataFrame, y: pd.Series,
                      hyperparams: dict[str, Any]) -> tuple[Any, list[float]]:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                f"train() for a keras model needs tensorflow installed -- {_LEARNING_EXTRA_HINT}"
            ) from exc

        epochs = hyperparams.get("epochs", 10)
        lr = hyperparams.get("lr", 1e-3)
        batch_size = hyperparams.get("batch_size", 32)
        verbose = hyperparams.get("verbose", 0)

        X_arr = X.to_numpy(dtype="float32")
        input_shape = model.input_shape   # e.g. (None, 1, 8) for the real LSTM in this registry
        if len(input_shape) == 3:
            seq_len, n_features = input_shape[1], input_shape[2]
            if X_arr.shape[1] != n_features:
                raise ValueError(
                    f"model expects {n_features} features per timestep, data has {X_arr.shape[1]}"
                )
            # This SDK only handles "1 timestep per row" (confirmed real
            # shape: every Keras model in this catalog treats each sample
            # as a length-1 sequence, not a genuine time-ordered window) --
            # a real multi-step sequence would need time-ordered grouping
            # that map_input_columns/generate_synthetic_rows don't produce.
            if seq_len != 1:
                raise RuntimeError(
                    f"model expects {seq_len} timesteps per sample; only "
                    f"seq_len=1 is supported by this SDK today"
                )
            X_arr = X_arr.reshape(-1, seq_len, n_features)

        y_arr = y.to_numpy(dtype="float32")

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
        history = model.fit(X_arr, y_arr, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return model, history.history.get("loss", [])
