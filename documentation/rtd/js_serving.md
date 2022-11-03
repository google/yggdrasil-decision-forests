# Serving models on the Web with JS/Wasm

The JavaScript API makes it possible to run an Yggdrasil Decision Forests model
or a TensorFlow Decision Forests model in a webpage.

## Usage example

The following example shows how to run a Yggdrasil model in a webpage. The
result of this example is viewable
[here](https://achoum.github.io/yggdrasil_decision_forests_js_example/example.html).
The source code of the example is available
[here](https://github.com/google/yggdrasil-decision-forests/tree/main/yggdrasil_decision_forests/port/javascript/example).

### Step 1: Prepare the model

The JavaScript API requires a Zipped Yggdrasil model.

Optionally, the meta-data of the model can be
[removed](https://ydf.readthedocs.io/en/latest/improve_model.html#remove-model-meta-data)
to make the model smaller with the `edit_model` or the `pure_serving_model=True`
argument.

**Train and prepare a model for JS with the CLI API**

```shell
# Install the CLI API. See "CLI API / Quick Start" or " CLI API / Install" sections for more details.
wget https://github.com/google/yggdrasil-decision-forests/releases/download/1.0.0/cli_linux.zip
unzip cli_linux.zip

# Train the model (see Quick Start)
./train ... --output=model

# Remove the meta-data from the model (makes the model smaller)
./edit_model --input=/tmp/model --output=/tmp/model_pure --pure_serving=true

# Zip the model.
zip -j /tmp/model.zip /tmp/model_pure/*

# /tmp/model.zip can be used directly in Javascript.
```

**Train and prepare a model for JS with TensorFlow Decision Forests**

```python
# Load TF-DF (see TF-DF tutorial)
import tensorflow_decision_forests as tfdf

# Load a dataset in a Pandas dataframe.
train_df = pd.read_csv("project/train.csv")

# Convert the dataset into a TensorFlow dataset.
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="my_label")

# Train a Random Forest model.
model = tfdf.keras.GradientBoostedTreesModel(pure_serving_model=True)
model.fit(train_ds)

# Export the model to a SavedModel.
model.save("/tmp/tfdf_model")

# Zip the model.
zip -j /tmp/model.zip /tmp/tfdf_model/assets/*

# /tmp/model.zip can be used directly in Javascript.
```

``` {note}
See the [convert model](convert_model) page on how to import models from other formats.
```

``` {note}
**Warning:** The model zip file should be a *flat* zip file: The model files
should be located at the **root** of the zip file. If using the
[zip tool](https://linux.die.net/man/1/zip), a flat file can be created with the `-j` option.
```

### Step 2: Install the YDF Javascript library

Download the latest version of the YDF Javascript library on the
[Yggdrasil release page](https://github.com/google/yggdrasil-decision-forests/releases).
For example,
[this](https://github.com/google/yggdrasil-decision-forests/releases/download/js_0.2.5_rc1/ydf.zip)
is the YDF Javascript library for YDF 0.2.5.

The files of your webpage can be structured as follows:

-   `index.html` : The webpage running the model (see step 3).
-   `model.zip` : The model created in step 1.
-   `ydf/` :The content of the zip file downloaded above.
    -   `inference.js`
    -   `inference.wasm`

### Step 3: Create the webpage

Add the YDF Javascript wrapper to your html:

```html
<!-- Yggdrasil Decision Forests -->
<script src="ydf/inference.js"></script>

<!-- JSZip -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.0/jszip.min.js"></script>
```

Then, also in the HTML header or in the body, load the YDF library:

```javascript
<script>
let ydf = null;
YggdrasilDecisionForests().then(function (m) {
  ydf = m;
  console.log("The library is loaded and ready to be used!");
});
</script>
```

Once the library is loaded, load the model:

``` {note}
In practice, you will likely load the model next to the `console.log("The library...` call above.
```

```javascript
let model = null;
ydf.loadModelFromUrl("model.zip").then((loadedModel) => {
  model = loadedModel;
  console.log("The model is loaded");
});
```

Once, the model is loaded, you can make predictions:

```javascript
let examples = {
  feature_1: [1, null, 3], // "null" represents a missing value.
  feature_2: ["cat", "dog", "tiger"],
  };
let predictions = model.predict(examples);
```

``` {note}
The input of the `predict` function should be the same as the input features used to train the model.
```

Finally, when you are done using the model, unload the model:

```javascript
model.unload();
model = null;
```

## Compile YDF Javascript library from source

To compile, the YDF Javascript library from the source, install the dependencies
required to compile YDF (see
[CLI / Compile from source](cli_install.md#compile-from-source)), and then run
the following command:

```shell
# Download YDF
git clone https://github.com/google/yggdrasil-decision-forests.git
cd yggdrasil-decision-forests


# Compile Yggdrasil Decision Forest WebAssembly inference engine
# The result is available at "bazel-bin/yggdrasil_decision_forests/port/javascript/ydf.zip"
# Note: You can skip the "--config=lto --config=size" flags to speed-up compilation at the expense of a larger binary.
bazel build -c opt --config=lto --config=size --config=wasm //yggdrasil_decision_forests/port/javascript:create_release

# The YDF library is available at: bazel-bin/yggdrasil_decision_forests/port/javascript/ydf.zip
```

If the compilation fails with the error `Exception: FROZEN_CACHE is set`,
disable the Emscripten frozen cache. This is done by setting
`FROZEN_CACHE=False` in the `emscripten_config` Emscripten configuration file
located in the Bazel cache
`~/.cache/bazel/.../external/emsdk/emscripten_toolchain`. See here
[here](https://github.com/emscripten-core/emsdk/issues/971) for more details.
