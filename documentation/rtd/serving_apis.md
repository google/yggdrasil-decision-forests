# Serving APIs

Once trained, the predictions of a model can be generated using the same API
used for model training (see the list of [model training APIs](apis)). For
example, the predictions of a model trained with the TensorFlow API can be
generated with `model.predict(...)`. This solution works great for prediction
analysis and evaluation, but it is generally not suited for productionization.

Instead, various specific solutions are available to run a model in production
(called **model serving**). The choice of this solution is independent of the
API used for model development / training.

## Model serving solutions

-   **CLI API**: The CLI API makes it easy to generate offline predictions on
    small datasets.

-   **C++ API**: The C++ API is the most efficient and flexible solution. It is
    a small single-thread library. In the case of large datasets, or for online
    serving, It is up to the user to distribute the computation.

-   **TensorFlow Serving**: TF Serving is a TensorFlow product to run models
    online in large production settings. It is commonly used to run TF-DF with
    large QPS.

-   **JavaScript API**: The JS API makes it possible to run YDF models on
    webpages using WebAssembly.

-   **Go API**: The Go API is a Go native API to run YDF models in Go
    applications.
