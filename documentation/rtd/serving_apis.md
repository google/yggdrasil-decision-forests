# Serving APIs

Several solutions are available to run a model (called **model serving**).

## Model serving solutions

-   **C++ API**: The C++ API is the fastest solution to run models. In the case
    of large datasets, or for online serving, it is up to the user to distribute
    the computation.

-   **CLI API**: The `predict` command of the CLI API is the simplest solution
    to generate offline predictions in-process on a small dataset. The CLI API
    directly uses the C++ API underneath, and therefore has the same speed.

-   **TensorFlow Serving**: TF Serving is a TensorFlow product to run TensorFlow
    models online in large distributed production settings. TensorFlow can add a
    significant amount of overhead over the computation of the model. For small
    models, this overhead can be of multiple orders of magnitude i.e., runing
    the model in TF Serving is >100x slower than running the same model with the
    C++ API.

-   **JavaScript API**: The JavaScript API makes it possible to run YDF models
    on webpages using WebAssembly.

-   **Go API**: The Go API is a Go native API to run YDF models in Go
    applications. This solution is generally ~2x slower than the C++ API.
