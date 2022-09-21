# Serving model in JavaScript

The JavaScript API makes it possible to run a YDF model in a webpage using
WebAssembly.

## Usage example

The following example shows how to run a Yggdrasil model in a webpage. The
result of this example is viewable
[here](https://achoum.github.io/yggdrasil_decision_forests_js_example/example.html).
The source code of the example is available
[here](https://github.com/google/yggdrasil-decision-forests/tree/main/yggdrasil_decision_forests/port/javascript/example).

### Step 1: Prepare the model

The JavaScript API requires a Zipped Yggdrasil model. If available in another
format, models might need to be [converted](convert_model) first.

In this example, we will use a pre-existing YDF model trained on the Adult
dataset.

```shell
# Download the model
git clone https://github.com/google/yggdrasil-decision-forests.git
MODEL_PATH="yggdrasil-decision-forests/yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt"
```

**(Optional**) By default, YDF models contain meta-data used for model
interpretation and debugging. This meta-data is not used for model inference and
can be discarded to decrease the model size using the `edit_model` tool:

```shell
# Install the CLI API. See "CLI API / Quick Start" or " CLI API / Install" sections for more details.
wget https://github.com/google/yggdrasil-decision-forests/releases/download/1.0.0/cli_linux.zip
unzip cli_linux.zip

# Remove the meta-data from the model
./edit_model --input=${MODEL_PATH} --output=/tmp/my_model --pure_serving=true

# Look at the size of the model
du -h ${MODEL_PATH}
du -h /tmp/my_model
```

Results:

```
528K yggdrasil-decision-forests/yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt
264K /tmp/my_model
```

We can now compress the model in a zip file:

```shell
zip -r model.zip /tmp/my_model
du -h model.zip
```

Results:

```
96K     model.zip
```

The model size is 96KB.

### Step 2: Install the YDF Javascript library

Download the YDF Javascript library on the
[Yggdrasil release page](https://github.com/google/yggdrasil-decision-forests/releases),
or [compile it from source](#compile-ydf-javascript-library-from-source). For
example,
[this](https://github.com/google/yggdrasil-decision-forests/releases/download/js_0.2.5_rc1/ydf.zip)
is the YDF Javascript library for YDF 0.2.5.

In the rest of this example, we assume that the content of this zip file was
extracted in a `ydf` directory next to the webpage and the model:

-   `index.html` : The webpage running the model (see step 3).
-   `model.zip` : The model created in step 1.
-   `ydf/` :The content of the zip file downloaded above.
    -   `inference.js`
    -   `inference.wasm`

### Step 3: Create the webpage

In the HTML header of a webpage, download the YDF and
[Zip](https://stuk.github.io/jszip/) libraries:

```html
<!-- Yggdrasil Decision Forests -->
<script src="ydf/inference.js"></script>

<!-- JSZip -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.0/jszip.min.js"></script>
```

Then, load the YDF library:

```javascript
let ydf = null;
YggdrasilDecisionForests().then(function (m) {
  ydf = m;
  console.log("The library is loaded");
});
```

As an example, the code to load the YDF library can be set in a `<script>`
section in the HTML header:

```html
<script>
let ydf = null;
YggdrasilDecisionForests().then(function (m) {
  ydf = m;
  console.log("The library is loaded");
});
</script>
```

Once the library is loaded, load the model:

```javascript
let model = null;
ydf.loadModelFromUrl("model.zip").then((loadedModel) => {
  model = loadedModel;
  console.log("The model is loaded");
});
```

Then, generate predictions with the model:

```javascript
let examples = {
  feature_1: [1, null, 3], // "null" represents a missing value.
  feature_2: ["cat", "dog", "tiger"],
  };
let predictions = model.predict(examples);
```

Final, you are done using the model, unload the model:

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
