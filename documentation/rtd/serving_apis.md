# Serving APIs

Once trained using one of the [Training APIs](apis), model predictions are
generated using one of the several available **Serving APIs** listed below or
available in the **DEPLOY A MODEL* section in the left column.

-   [C++ API](cpp_serving): The C++ API is the fastest solution to run models.
    In the case of large datasets, or for online serving, it is up to the user
    to distribute the computation.

-   [CLI API](cli_commands): The `predict` command of the CLI API is the
    simplest solution to generate offline predictions in-process on a small
    dataset. The CLI API directly uses the C++ API underneath, and therefore has
    the same speed.

-   [TensorFlow Serving](tf_serving): TF Serving is a TensorFlow product to run
    TensorFlow models online in large distributed production settings.
    TensorFlow can add a significant amount of overhead over the computation of
    the model. For small models, this overhead can be of multiple orders of
    magnitude i.e., running the model in TF Serving is >100x slower than running
    the same model with the C++ API.

-   [JavaScript API](js_serving): The JavaScript API makes it possible to run
    YDF models on webpages using WebAssembly.

-   [Go API](go_serving): The Go API is a Go native API to run YDF models in Go
    applications. This solution is generally ~2x slower than the C++ API.

-   [TensorFlow Decision Forests Python API](https://www.tensorflow.org/decision_forests/tutorials/predict_colab):
    The `predict` function of the TensorFlow Decision Forests library generates
    predictions in Python. This API follows the Keras API. This API is simple to
    use but significantly slower than other APIs.
