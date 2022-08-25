# Yggdrasil / TensorFlow Decision Forests in Javascript

The Yggdrasil Decision Forests Javascript port is a Javascript + WebAssembly
library to run (i.e. generate the predictions) of Yggdrasil Decision Forests
models on the web.

This library is compatible with TensorFlow Decision Forests models.

## Usage example

The [example/](example) directory (i.e.
`yggdrasil_decision_forests/port/javascript/example/example.html`) contains a
running example of YDF model inference in a webpage. The following section
details how this example work. A live example is available
[here](https://achoum.github.io/yggdrasil_decision_forests_js_example/example.html).

**Note:** The shell commands below should be run from the Yggdrasil root
directory i.e. the directory containing `CHANGELOG.md`.

**Step 1**

First, train and save to disk a Yggdrasil Decision Forests model using one of
the available APIs. See the
[Yggdrasil TensorFlow Decision Forests page](../../README.md) for more details.

-   C++:
    [user manual](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/user_manual.md),
    [example](https://github.com/google/yggdrasil-decision-forests/blob/main/examples/beginner.cc).
-   CLI:
    [user manual](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/user_manual.md),
    [example](https://github.com/google/yggdrasil-decision-forests/blob/main/examples/beginner.sh).
-   Python (with TensorFlow Decision Forests):
    [website](https://www.tensorflow.org/decision_forests),
    [examples](https://www.tensorflow.org/decision_forests/tutorials). The
    Yggdrasil Decision Forests model is located in the `assets` subdirectory of
    the TensorFlow model.

Note: Yggdrasil Decision Forests model is a directory containing a
`data_spec.pb` file.
[Here](https://github.com/google/yggdrasil-decision-forests/tree/main/yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt)
is an example of model.

If the size of the model or its inference speed is important to you, the
following suggestions can help optimizing it:

1.  Gradient Boosted Trees models are both smaller and faster than Random Forest
    models. If both have the same quality, prefer a Gradient Boosted Trees
    model.

2.  The number of trees (controlled with the `num_trees`) parameter impacts the
    size of Random Forest models.

3.  If you don't expect to interpret the model, use the
    `keep_non_leaf_label_distribution=False` hyperparameter.

4.  Always, if you don't expect to interpret the model, use the
    `keep_non_leaf_label_distribution=False` advanced hyperparameter.

**Step 2**

Archive the model in a zip file.

```shell
zip -jr model.zip /path/to/my/model
```

For this example, we can use one of the unit test models:

```shell
zip -jr model.zip yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt
```

**Step 3**

Download the Yggdrasil Decision Forest Javascript port library using one of the
options:

-   Download a [pre-compiled binaries]() (Not yet available).
-   Compile the library from source by running:

```shell
# Compile Yggdrasil Decision Forest Webassembly inference
# The result is available at "bazel-bin/yggdrasil_decision_forests/port/javascript/ydf.zip"
# Note: You can skip the "--config=lto --config=size" flags to speed-up compilation at the expense of a larger binary.
bazel build -c opt --config=lto --config=size --config=wasm //yggdrasil_decision_forests/port/javascript:create_release
```

If you get the error `Exception: FROZEN_CACHE is set` during the compilation,
you need to disable the frozen cache (i.e. set `FROZEN_CACHE=False` in the
`emscripten_config` Emscripten configuration file located in the Bazel cache
e.g. `~/.cache/bazel/.../external/emsdk/emscripten_toolchain`). See details
[here](https://github.com/emscripten-core/emsdk/issues/971). Alternatively, you
can remove the `--config=lto` statement in the compilation command.

**Step 5**

Decompress the library.

```
unzip bazel-bin/yggdrasil_decision_forests/port/javascript/ydf.zip \
 -d yggdrasil_decision_forests/port/javascript/example/ydf
```

The library is composed of two files:

```
ydf/inference.js
ydf/inference.wasm
```

**Step 4**

Add the library to the HTML header of your webpage. Also add
[JSZip](https://stuk.github.io/jszip/).

```
<!-- Yggdrasil Decision Forests -->
<script src="ydf/inference.js"></script>

<!-- JSZip -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.0/jszip.min.js"></script>
```

See `yggdrasil_decision_forests/port/javascript/example/example.html` for an
example.

**Step 5**

In Javascript, load the library :

```js
let ydf = null;
YggdrasilDecisionForests().then(function (m) {
  ydf = m;
  console.log("The library is loaded");
});
```

Then, load the model from an url:

```js
let model = null;
ydf.loadModelFromUrl("https://path/to/my/model.zip").then((loadedModel) => {
  model = loadedModel;

  console.log("The model is loaded");
  console.log("The input features of the model are:", model.getInputFeatures());
});
```

Compute predictions with the model:

```js
let examples = {
  feature_1: [1, null, 3], // "null" represents a missing value.
  feature_2: ["cat", "dog", "tiger"],
  };
let predictions = model.predict(examples);
```

Finally, unload the model:

```js
model.unload();
model = null;
```

**Step 6**

Start a http(s) server:

```shell
# Start a http server with python.
(cd yggdrasil_decision_forests/port/javascript/example && python3 -m http.server)
```

Open the webpage `http://localhost:8000/example.html`.

**Step 7**

In this example, you can see three buttons:

-   **Load model:** Downloads and load the model in memory.
-   **Apply model:** Apply the model on the toy examples specified in the `Input
    examples` text area.
-   **Unload model:** Unload the model from memory.
