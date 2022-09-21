# Training APIs

YDF is available in different programming languages / APIs both for model
training and model serving. Model and training configurations are
cross-compatible in between APIs. For example, it is common to develop a model
with one API (e.g., Python API) and then to deploy it with another API (e.g.,
C++ API).

The following APIs available for **model training**. The list of APIs available
for **model serving** is available on [this page](serving_apis).

## Model training

-   **CLI API**: This is the most complete and efficient API. A model can be
    trained, evaluated, analyzed, and benchmarked in only a few lines. However,
    the CLI API does not support data preprocessing i.e., data preprocessing
    should be applied before calling the CLI API.

-   **Python API / TensorFlow Decision Forests**: This API is the most
    user-friendly. This API is especially suited for small datasets with data
    preprocessing. TF-DF models are compatible both with the rest of the
    TensorFlow ecosystem (e.g., TF-Hub, TF Serving) and with the other YDF APIs.
    However, TF-DF does not support YDF model evaluation (instead, models are
    evaluated with TensorFlow) and can be slower than the other APIs.

-   **C++ API**: This API is equivalent to the CLI API. Each CLI binary has a
    corresponding function in C++ API making the conversion from CLI API to C++
    API easy during productionisation.
