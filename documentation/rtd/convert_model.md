# Model formats

Models exist in multiple formats:

-   **Yggdrasil model**: A directory containing a Yggdrasil model. The directory
    is recognizable by a `<optional prefix>data_spec.pb` file. This is the
    native format of the library. This format is used with the CLI, C++, and Go
    APIs.

-   **Zipped Yggdrasil model**: A zip file containing a Yggdrasil model. This
    format is used with the JavaScript serving API.

-   **A TensorFlow Decision Forests model**: This is a
    [TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model)
    containing one or more Yggdrasil models and recognizable by a
    `saved_model.pb` file . A SavedModel itself is a directory and the Yggdrasil
    models are stored in one of its subdirectories. The directory containing the
    Yggdrasil models (currently named `assets`). This format is used with the
    TensorFlow Decision Forests API.

Models can be "converted" between formats:

-   Yggdrasil model &rarr; Zipped Yggdrasil model: Zip the model directory.

-   Zipped Yggdrasil model &rarr; Yggdrasil model: Unzip the model directory.

-   TensorFlow Decision Forests model &rarr; Yggdrasil model: The Yggdrasil
    model is contained in the `assets` sub-directory of the TensorFlow Decision
    Forests model directory. See details in the the next section.

-   (Zipped) Yggdrasil model &rarr; TensorFlow Decision Forests model: This
    conversion requires some code. See the next section.

## Convert a a TensorFlow Decision Forests model to a Yggdrasil model

A TensorFlow Decision Forests model directory contains a sub-dirctory containing
a Yggdrasil model. In the next example, we create a TF-DF model in `/tmp/model`,
and show the Yggdrasil model in `/tmp/model/assets`.

```shell
import tensorflow_decision_forests as tfdf

# Train a TF-DF model (without any pre-processing)
model = tfdf.keras.GradientBoostedTreesModel()
model.fit(...)

# Export the model as a TF Saved Model
# Note: /tmp/model/assets is an Yggdrasil Decision Forests model
model.save("/tmp/model")

# Show the structure of the TF SavedModel.
!tree /tmp/model
# /tmp/model
# ├── assets
# │   ├── <prefix>data_spec.pb
# │   ├── <prefix>done
# │   ├── <prefix>gradient_boosted_trees_header.pb
# │   ├── <prefix>header.pb
# │   └── <prefix>nodes-00000-of-00001
# ├── keras_metadata.pb
# ├── saved_model.pb
# └── variables
#     ├── variables.data-00000-of-00001
#     └── variables.index
```

Yggdrasil tools can be used. The next example use the `show_model` Yggdrasil
command on a TF-DF model:

```shell
# Show the model structure using Yggdrasil tool box
!yggdrasil_decision_forests/cli/show_model --model=/tmp/model/assets
```

An Yggdrasil model contained in a TF-DF model differ from the TF-DF model in the
following points:

-   Any TensorFlow preprocessing (e.g., using of the `preprocessor` model
    constructor argument) is not taken into account.

-   Categorical integer features need to be offset by 1. Therefore, for
    simplicity of use, it is better to use categorical string features.

## Convert a Yggdrasil model to a TensorFlow Decision Forests model

The conversion of an Yggdrasil model to to a TensorFlow Decision Forests is done
using the `tfdf.keras.yggdrasil_model_to_keras_model()` function available in in
TensorFlow Decision Forests.

The following example converts a YDF model into a TensorFlow Decision Forests
model:

```python
# Prepare and load the model with TensorFlow
import tensorflow_decision_forests as tfdf

tfdf.keras.yggdrasil_model_to_keras_model(<path to ydf model>, <path to tfdf-model>)
```

By default, `yggdrasil_model_to_keras_model` creates a model with raw tensor
inputs. The type of the input tensor is determined by the semantics of the
features. For example, numerical features are always encoded as float32. You can
change this signature using the `input_model_signature_fn` argument.

The following example converts a YDF model while making sure that `feature_1` is
encoded into a int64 tensor.

```python
def custom_model_input_signature(inspector) -> Any:

    # Default signature.
    input_spec = tfdf.keras.build_default_input_model_signature(inspector)

    # Override the type of feature_1.
    input_spec["feature_1"] = tf.TensorSpec(shape=[None], dtype=tf.int64)
    return input_spec

tfdf.keras.yggdrasil_model_to_keras_model(
    ygg_model_path,
    tfdf_model_path,
    input_model_signature_fn=custom_model_input_signature)
```

In YDF, pre-integerized categorical features (i.e., categorical features
represented as integers) are 1-indexed. The value 0 (zero) is reserved for the
out-of-vocabulary item. However, many TensorFlow tools assume that categorical
integer features are 0-indexed. For this reason, TF-DF models are 0-indexed and
contain an implicit +1 offset to the YDF model underneath. This offset does not
impact users in most cases. However, when converting models, depending on how
this model will be used, removing this offset might be better. This is done with
the `disable_categorical_integer_offset_correction` argument.
