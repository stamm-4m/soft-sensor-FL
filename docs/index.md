# stamm-sdk

Python client SDK for the Stamm ML registry: authentication and read-only model discovery, plus
a federated-learning pipeline (data prep, two validation gates, training, aggregation) for the
models hosted there.

```bash
pip install stamm-sdk              # auth + read-only discovery
pip install "stamm-sdk[learning]"  # + training (scikit-learn, torch, tensorflow)
```

New here? Start with the [guide](SDK_GUIDE.md) -- it explains every stage of the pipeline with
working code. [Training examples](TRAINING_EXAMPLES.md) has a complete Manual-vs-Automatic example
for every model in the live registry. The [model catalog](MODEL_CATALOG.md) covers which specific
models actually train today and why some don't. The [API reference](api.md) is the auto-generated
per-module/per-function detail, for when you already know what you're looking for.

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

SDK_GUIDE
TRAINING_EXAMPLES
MODEL_CATALOG
api
```
