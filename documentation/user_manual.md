# User Manual

This page is an in-depth user introduction to Yggdrasil Decision Forests (YDF).
It is complementary to the beginner example available in `examples/`.

## Table of Contents

<!--ts-->

*   [User Manual](#user-manual)
    *   [Table of Contents](#table-of-contents)
    *   [Interfaces](#interfaces)
    *   [Dataset](#dataset)
    *   [Dataset path and format](#dataset-path-and-format)
    *   [Learners and Models](#learners-and-models)
    *   [Distributed Training](#distributed-training)
        *   [GRPC distribute implementation (recommended)](#grpc-distribute-implementation-recommended)
        *   [TF_DIST distribute implementation](#tf_dist-distribute-implementation)
    *   [Meta-Learner](#meta-learner)
    *   [Model/Learner Evaluation](#modellearner-evaluation)
    *   [Experiment](#experiment)
    *   [Model Analysis](#model-analysis)
        *   [Variable Importances](#variable-importances)
            *   [Model agnostic](#model-agnostic)
            *   [Decision Forests specific](#decision-forests-specific)
    *   [Manual Tuning of Hyper-parameters](#manual-tuning-of-hyper-parameters)
        *   [Best first global growing strategy for GBT](#best-first-global-growing-strategy-for-gbt)
        *   [Oblique splits for GBT and RF](#oblique-splits-for-gbt-and-rf)
        *   [Random Categorical splits for GBT and RF](#random-categorical-splits-for-gbt-and-rf)
        *   [Hessian splits for GBT](#hessian-splits-for-gbt)
        *   [Number of trees for RF and GBT](#number-of-trees-for-rf-and-gbt)
        *   [Disabling the validation dataset for GBT](#disabling-the-validation-dataset-for-gbt)
        *   [Disabling winner take all for RF](#disabling-winner-take-all-for-rf)
        *   [Super learners](#super-learners)
    *   [Automated Tuning of Hyper-parameters](#automated-tuning-of-hyper-parameters)
    *   [Feature Engineering](#feature-engineering)
    *   [Model Inference](#model-inference)
        *   [Fast engine](#fast-engine)
        *   [Serving TensorFlow Decision Forests model](#serving-tensorflow-decision-forests-model)
    *   [Advanced features](#advanced-features)

<!--te-->

## Interfaces

YDF has two interfaces: C++ and the command line interface (CLI). Both
interfaces are equivalent and compatible.

YDF is also available in Python in TensorFlow Decision Forests (TF-DF). TF-DF
models are compatible with YDF models, and vice versa. For example, you can
train a model in TF-DF, and then run it using the Yggdrasil C++ API.

## Dataset

An **attribute** (also known as a *feature*, *variable* or *column*) refers to
one particular piece of information in a tabular dataset. For example, the
*age*, *name* and *country of birth* of a person are attributes. Different
attributes can have different types (or *semantics*). For example, the *age* is
*numerical* while the *country* is categorical.

An **example** is a mapping between **attribute** names (stored as a string) and
**attribute** values (storage depends on the value type).

Note: For efficiency reasons, **examples** are not stored as string dictionaries
(e.g. std::map).

An **Example** (in Python notation):

```python
{"attribute_1": 5.1, "attribute_2": "CAT", "attribute_3": [1, 2, 3]}
```

A **dataset** is a list of **examples**. Datasets are stored in memory (e.g.
vertical dataset, list of proto::Example) or on disk (e.g. csv, tfrecords of
tf.examples). An example in a dataset is unambiguously identified by its
**index**.

A **data specification** (or dataspec for short) is a list of attribute
definitions that indicates how a **dataset** is semantically understood. The
definition of an attribute contains its name, semantic type, and type-dependent
meta-information (e.g. the mean value of a numerical attribute, the vocabulary
of a categorical attribute).

A dataspec is stored as a protobuffer (`dataset/data_spec.proto`) and can be
printed in a readable way using the `show_dataspec` command.

A **Dataspec** (raw proto):

```proto
columns {
    name: "attribute_1"
    type: NUMERICAL
    numerical {
        mean: 10
    }
}
columns {
    name: "attribute_2"
    type: CATEGORICAL
    categorical {
        items {
          key: "CAT"
          value { index: 1 }
        }
        items {
            key: "COT"
            value { index: 2 }
        }
    }
}
columns {
    name: "attribute_3"
    type: CATEGORICAL_SET
    categorical_set {
        is_already_integerized: true
    }
}
```

**Dataspec** (pretty print):

```
Number of records: 1
Number of columns: 3

Number of columns by type:
    NUMERICAL: 1 (33%)
    CATEGORICAL: 1 (33%)
    CATEGORICAL_SET: 1 (33%)

Columns:

NUMERICAL: 6 (40%)
    0: "feature_1" NUMERICAL mean:38.5816

CATEGORICAL: 1 (33%)
    1: "feature_2" CATEGORICAL has-dict vocab-size:2 zero-ood-items most-frequent:"CAT" 1 (50%)

CATEGORICAL_SET: 1 (33%)
    2: "feature_3" CATEGORICAL_SET integerized

Terminology:
    nas: Number of non-available (i.e. missing) values.
    ood: Out of dictionary.
    manually-defined: Attribute which type is manually defined by the user i.e. the type was not automatically inferred.
    tokenized: The attribute value is obtained through tokenization.
    has-dict: The attribute is attached to a string dictionary e.g. a categorical attribute stored as a string.
    vocab-size: Number of unique values.
```

The representation of an attribute is not always sufficient to determine its
semantic type. For example, csv files stores all attribute types (including
numerical ones) as strings. Another example, integer values are often
interpreted as numerical or categorical (i.e. enum) values.

Dataspecs are created automatically by the dataspec inference tool (e.g.
`cli/infer_dataspec`). If an automatically-inferred dataspec is incorrect (e.g.
an integer value is interpreted as a numerical instead of a categorical), one
can use a **dataspec guide** (see `dataset/data_spec.proto`) to configure the
dataspec inference, to enforce the semantic type of a column.

Following is an example of dataspec guide that forces attributes starting with
"foo" to be detected as categorical.

```proto
column_guides {
    column_name_pattern: "^foo.*"
    type: CATEGORICAL
}
```

The following two examples show the inference of a dataspec using the CLI and
C++ API:

**CLI**

```shell
infer_dataspec --dataset=csv:/my/dataset.csv --output=dataspec.pbtxt

# Human readable description of the dataspec.
show_dataspec --dataspec=dataspec.pbtxt
```

**C++**

```c++
using namespace yggdrasil_decision_forests;
dataset::proto::DataSpecification data_spec;
dataset::CreateDataSpec("csv:/my/dataset.csv", false, guide, &data_spec);

# Human
description of the dataspec.
std::cout << dataset::PrintHumanReadable(data_spec);
```

## Dataset path and format

The format of a dataset is specified by a path prefix and a colon (":").

Example:

-   `csv:/path/to/my/dataset.csv`
-   `tfrecord+tfe:/path/to/my/dataset.rtfe`

By default, two formats are supported: Csv and TensorFlowRecord of
tensorflow.Example compressed with GZIP.

Dataset formats rely on a registration mechanism: Users can restrict or extend
the list of supported formats with dependency injection (there is a registration
mechanism). All the officially supported formats can be injected at once with
the dependency
`//third_party/yggdrasil_decision_forests/dataset:all_dataset_formats`.

Dataset paths also support sharding, globbing and comma separation.

Example:

-   `csv:/path/to/my/dataset.csv@10`
-   `csv:/path/to/my/dataset_*.csv`
-   `csv:/path/to/my/dataset_1.csv,/path/to/my/dataset_1.csv`

## Learners and Models

YDF relies on the **Learner and Model** abstraction: A **learner** is a function
that consumes a dataset, and outputs a model. A **model** is a function that
consumes an example, and outputs a prediction.

<p align="center">
<img src="image/learner_and_model.png" style="margin:auto; display:block; width:330px;" />
</p>

In this definition, a learning algorithm and a set of hyper-parameters is **a
learner**, and a trained model is **a model**.

A **model** can be stored in memory or on disk. Information about a model (e.g.
input features, label, validation score, learner) can be displayed using the
**show_model** command.

A **model** also contains metadata. Metadata is not used to compute predictions.
Instead, users or tools can query this meta data for analysis (e.g. feature
importances, training losses).

A model can be **compiled** into an **(serving) engine**. An **engine** is a
model without metadata and with an internal structure optimized for fast
inference. While **models** are platform agnostic, **engines** are tied to
specific hardware (e.g. require a GPU or AVX512 instructions). Engine inference
is orders of magnitude faster than the standard *model* inference.

The method `model.BuildFastEngine()` will compile the model into the most
efficient engine on the current hardware. The `cli/benchmark_inference` tool can
measure the inference speed of a model and the compatible serving engines.

A **learner** is instantiated from a
[TrainingConfig](../yggdrasil_decision_forests/learner/abstract_learner.proto)
proto. The **TrainingConfig** defines the label, input features, task, learner
type and hyper-parameters for the learning.

The **generic hyper-parameters** (GHP) is an alternative representation of the
Training Config stored as a map of string to values. GHP are suited for
automated hyper-parameter optimization and generally used on top of an existing
minimal Training Configuration. GHP are used by TensorFlow Decision Forests.

The [learner](learners.md) page lists the official learners and they
hyper-parameters.

Optionally, a learner can be configured with a **deployment specification**. A
[deployment specification](../yggdrasil_decision_forests/learner/abstract_learner.proto)
defines the computing resources to use during the training. This includes the
number of threads or the number of workers (in case of distributed training).
Changing the deployment configuration (or not providing any) can impact the
speed of training, but it cannot impact the model: Two models trained with the
same learner, but different deployment specifications are guaranteed to be
equal.

By default, learners are configured to be trained locally using 6 threads.

Finally, the **logging directory** of a learner is an optional directory that
specifies where the learner exports information about the learning
process--information that is not necessary for the model to operate. This
information can be composed of plots, tables and HTML reports, and helps one to
understand the model. The logging directory of a learner does not have any
defined structure as different learners can export different types of
information.

The following two examples show the training of a model using the CLI and C++
API (see above how to create the *dataspec* in the `dataspec.pbtxt` file):

**CLI**

```shell

cat <<EOF > hparams.pbtxt
task: CLASSIFICATION
label: "income"
learner: "GRADIENT_BOOSTED_TREES"
EOF

train \
  --dataset=csv:/my/dataset.csv \
  --dataspec=dataspec.pbtxt \
  --config=hparams.pbtxt \
  --output=/my/model

# Human description of the model.
show_model --model=dataspec.pbtxt
```

**C++**

```c++
using namespace yggdrasil_decision_forests;

dataset::proto::DataSpecification data_spec = ... // Previous example.

// Configure the training.
model::proto::TrainingConfig config;
config.set_task(model::proto::CLASSIFICATION);
config.set_label("income");
config.set_learner("GRADIENT_BOOSTED_TREES");

// Instantiate the learner.
std::unique_ptr<model::AbstractLearner> learner;
CHECK_OK(model::GetLearner(config, &learner));

// Train
std::unique_ptr<model::AbstractModel> model = learner->Train("csv:/my/dataset.csv", data_spec);

// Export the model to disk.
CHECK_OK(model::SaveModel("/my/model", model.get()));

// Human description of the model.
std::string description;
model->AppendDescriptionAndStatistics(false, &description);
std::cout << description;
```

To minimize binary sizes and allow the support custom learning algorithms and
models, models and learners rely on a registration mechanism: Users can restrict
or extend the list of supported model and learners with dependency injection.

All the official supported models and learners can be respectively injected with
the dependencies `yggdrasil_decision_forests/model:all_models` and
`yggdrasil_decision_forests/learners:all_learners`.

## Distributed Training

*Distributed training* refers to the training a model with the use of multiple
computers. The distribution of computation, memory and the network IO in between
the machines depends on each distributed algorithm. See specific algorithm
documentation for more details.

When distributed training is used, the process running the user code (i.e.
running the `learner->Train()` function) is the manager. Unless specified
otherwise, the manager process runs close to no computation.

The following learners support distributed training:

-   `DISTRIBUTED_GRADIENT_BOOSTED_TREES`: an exact distributed implementation of
    the GRADIENT_BOOSTED_TREES learner. See
    [the DGBT manual](learner_distributed_gradient_boosted_trees.md) for
    specific details.

Distributed training is enabled by configuring a `distribute` execution engine
and the `cache_path` field in the
[DeploymentConfig](../yggdrasil_decision_forests/learner/abstract_learner.proto)
. Different `distribute` execution engines are available. They don't impact the
effective trained model.

### `GRPC` distribute implementation (recommended)

Each worker is a process running the
`yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_worker_main`
binary. Workers are communicating through GRPC. This option is recommended for
Yggdrasil users. It is up to the user to start the `:grpc_worker_main` processes
and configure the socket addresses in the DeploymentConfig proto.

For example:

DeploymentConfig proto:

```
cache_path: "/directory/accessible/to/all/the/workers"
num_threads: 32 # Each worker will run 32 threads.
try_resume_training: true # Allow training to be interrupted and resumed.

distribute {
  implementation_key: "GRPC"
  [yggdrasil_decision_forests.distribute.proto.grpc] {
    socket_addresses {
      # Configure the 3 workers.
      addresses { ip: "192.168.0.10" port: 1001 }
      addresses { ip: "192.168.0.11" port: 1001 }
      addresses { ip: "192.168.0.12" port: 1001 }
      }
    }
  }
```

See
[yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.proto](../yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.proto)
for more details.

### `TF_DIST` distribute implementation

Each worker is a generic TensorFlow parameter server with TF-DF custom ops. One
such rule is pre-configured:
`tensorflow_decision_forests/tensorflow/distribute:tensorflow_std_server`. This
implementation is best suited for TensorFlow Decision Forest users. The workers
can be configured either by socket addresses or with the
[TF_CONFIG](https://www.tensorflow.org/guide/distributed_training#setting_up_the_tf_config_environment_variable)
variable

See
[tensorflow_decision_forests/tensorflow/distribute/tf_distribution.proto](http://google3/third_party/tensorflow_decision_forests/tensorflow/distribute/tf_distribution.proto)
for more details.

## Meta-Learner

The **Learner and Model** abstraction allows the development of generic tools
that operates on top of the learner concept. Some of these tools, called
**meta-learners**, are learners (i.e. dataset in, model out) that rely on
sub-learners.

For example, an ensemble learner, an hyper-parameter tuner, a feature-selector
or a calibrator can all be implemented as learners and used in the same way as
any learner (see section on learner).

Importantly, learners and meta-learners can be combined. For example, an
ensembler on top of a hyper-parameter tuner on top of a calibrator.

## Model Analysis

The
[show_model](../yggdrasil_decision_forests/cli/show_model.cc)
tool shows the following information about a model:

-   The input features and labels of the model.
-   The variable importance of each features.
-   The detailed or summarized structure of the model (e.g. number of nodes).
-   The compatible serving engine.
-   The training logs.

Alternatively, models can be visualized and analysed programmatically directly
using the
[TensorFlow Decision Forests python inspector](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/inspector/make_inspector)
. See the
[Inspect and debug](https://www.tensorflow.org/decision_forests/tutorials/advanced_colab)
colab for more details. An Yggdrasil model can be converted into a Tensorflow
Decision Forests model using the
[tfdf.keras.core.yggdrasil_model_to_keras_model](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/core/yggdrasil_model_to_keras_model)
function. Calling `model.summary()` shows the same information as the
`show_model` CLI.

### Variable Importances

Variable importances can be obtained with CLI (`show_model`), the C++ API
(`model.GetVariableImportance()`) and the Python API
(`tfdf.inspector.make_inspector(path).variable_importances()`).

The available variable importances are:

#### Model agnostic

`MEAN_{INCREASE,DECREASE}_IN_{metric}`: Estimated metric change from removing a
feature using
[permutation importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html)
. Depending on the learning algorithm and hyper-parameters, the VIs can be
computed with validation, cross-validation or out-of-bag. For example, the
`MEAN_DECREASE_IN_ACCURACY` of a feature is the drop in accuracy (the larger,
the most important the feature) caused by shuffling the values of a features.
For example, `MEAN_DECREASE_IN_AUC_3_VS_OTHERS` is the expected drop in AUC when
comparing the label class "3" to the others.

#### Decision Forests specific

`SUM_SCORE`: Sum of the split scores using a specific feature. The larger, the
most important.

`NUM_AS_ROOT`: Number of root nodes using a specific feature. The larger, the
most important.

`NUM_NODES`: Number of nodes using a specific feature. The larger, the most
important.

`MEAN_MIN_DEPTH`: Average minimum depth of the first occurence of a feature
across all the tree paths. The smaller, the most important.

## Manual Tuning of Hyper-parameters

The learner default hyper-parameters are set to reasonable default values or to
the values used in the original publication. To ensure the stability of training
configuration, default hyper-parameter values are never changed. Therefore, new
(generally better) learner features are by default always disabled.

This section list some of the hyper-parameters that often improve the quality of
the models:

### Best first global growing strategy for GBT

By default, trees are built with a greedy recursive local algorithm. Growing the
tree globally can improve the performances. In this case, the maximum number of
nodes plays an important role.

**Training config:**

```proto
decision_tree {
    growing_strategy_best_first_global {
      max_num_nodes: 64
    }
  }
```

**Generic hyper-parameter:**

```python
{growing_strategy = "BEST_FIRST_GLOBAL",
                    max_num_nodes = 64}
```

### Oblique splits for GBT and RF

By default, trees are "orthogonal" i.e. each split/condition tests a single
feature. Enabling oblique trees can improve the performances.

Oblique trees are more expensive to train. The num_projections_exponent
parameter play an important role for the training time and final model quality.

**Training config:**

```proto
decision_tree {
    sparse_oblique_split {
       num_projections_exponent : 2
       normalization: NONE
    }
  }
```

**Generic hyper-parameter:**

```python
{split_axis = "SPARSE_OBLIQUE",
              sparse_oblique_num_projections_exponent = 2,
                                                        sparse_oblique_normalization = 64}
```

### Random Categorical splits for GBT and RF

By default, categorical splits are learned with the CART algorithm. Training
categorical split with the Random algorithm can improve the model performances
at the expense of model size.

**Training config:**

```proto
decision_tree {
    categorical {
      random {}
    }
  }
```

**Generic hyper-parameter:**

```python
{categorical_algorithm = "RANDOM"}
```

### Hessian splits for GBT

By default, splits are trained with a first order approximation of the gradient.
Second order approximation can improve the performances.

**Training config:**

```proto
use_hessian_gain: true
```

**Generic hyper-parameter:**

```python
{use_hessian_gain = "true"}
```

### Number of trees for RF and GBT

Increasing the number of trees is likely to improve the quality of the model.

**Training config:**

```proto
num_trees: 2000
```

**Generic hyper-parameter:**

```python
{num_trees = 2000}
```

### Disabling the validation dataset for GBT

By default, GBT extracts a sample of the training dataset to build a validation
dataset (default to 10%). For small datasets, might be important and training
without validation (and early stopping) can improve the model. This can both
improve or hurt the model.

**Training config:**

```proto
validation_set_ratio: 0.0
early_stopping: NONE
```

**Generic hyper-parameter:**

```python
{validation_ratio = 0.0, early_stopping = "NONE"}
```

### Disabling winner take all for RF

By default, each tree in a RF is voting for a single class. When disabling
winner take all, each tree is voting for a distribution of classes. This
generally improve the model.

**Training config:**

```proto
winner_take_all_inference: false
```

**Generic hyper-parameter:**

```python
{winner_take_all = "false"}
```

### Super learners

Following are examples of GBT and RF training configuration with all the method
listed above:

```proto
learner: "GRADIENT_BOOSTED_TREES"

[yggdrasil_decision_forests.model.gradient_boosted_trees.proto.gradient_boosted_trees_config] {
  num_trees: 1000
  use_hessian_gain: true
  validation_set_ratio: 0.0
  early_stopping: NONE
  decision_tree {
    growing_strategy_best_first_global { max_num_nodes: 64 }
    sparse_oblique_split {}
    categorical { random {} }
  }
}
```

```proto
learner: "RANDOM_FOREST"

[yggdrasil_decision_forests.model.random_forest.proto.random_forest_config] {
  num_trees: 1000
  winner_take_all_inference: false
  decision_tree {
    sparse_oblique_split {}
    categorical { random {} }
  }
}

```

## Automated Tuning of Hyper-parameters

The optimal hyper-parameter value depends on the dataset and problem. They
should be tuned using cross-validation to maximize the performance of the model.
Automated hyper-parameter tuning is not yet published in YDF: users are expected
to write they own hyper-parameter tuning loop.

## Feature Engineering

Improving the features is often more impactful than improving the model. For the
sake of simplicity (YDF does only one thing) and execution efficiency (direct
code execution is often much faster than intermediate representation execution),
YDF core library does not contain customizable processing components with the
exception of text tokenization.

## Model Inference

Model inference is possible in one of two ways:

-   The *slow generic* inference with the `model.Predict()`.
-   The *fast engine* inference with the `model.BuildFastEngine()`.

In the majority of cases, the *fast engine* solution is preferable. The slow
generic is only useful when the model is changing often (e.g. during training)
or for unconventional models without specialized engines.

### Fast engine

Engines are injected with a registration mechanism similarly to the model and
learners. The rule
`//third_party/yggdrasil_decision_forests:all_inference_engines` injects all the
official engines. Calling `model.BuildFastEngine()` iterates over all the
compatible engines, and returns the faster one.

All engines available are thread safe, and generally benefit from running on
large batches of examples. Some of the engines use SIMD instructions: Make sure
to compile your binary and package your MPM with the flag `--copt=-mavx2`.

The list of compatible engine is available with the `cli/show_model` tool with
the `--engines` flag.

The following is an example show the initialization of a fast engine:

```c++
// Load the model.
std::unique_ptr<model::AbstractModel> model;
CHECK_OK(LoadModel("/path/to/model", &model));

// Create a fast engine.
//
// If no fast engine is available, the model cannot be served using the fast
// engine. The error will explain why. Non-fast engine compatible models can
// be served using the slow engine (see "slow engine" section below).
const auto engine = model->BuildFastEngine().value();
const auto& features = engine->features();

// Index the input features of the model.
//
// This operation should be done during the model loading.
const auto feature_age = features.GetNumericalFeatureId("age").value();
const auto feature_country = features.GetCategoricalFeatureId("country").value();
const auto feature_text = features.GetCategoricalSetFeatureId("text").value()

// At this point, "model" can be discarded.
```

The following example runs one example through the model.

```c++
// Allocate memory for 10 examples. This chunk of memory can be cached and
// re-used for speed-sensitive code.
auto examples = engine->AllocateExamples(10);

// Set all the values to be missing. The values may then be overridden by the
// "Set*" methods.
// If all the values are set with "Set*" methods, "FillMissing" can be skipped.
examples->FillMissing(features);

// Prepare one example.
const int example_idx = 0;
examples->SetNumerical(example_idx, feature_age, 30, features);
examples->SetCategorical(example_idx, feature_country, "UK", features);
examples->SetCategoricalSet(example_idx, feature_text,
std::vector<std::string>{
"hello", "world"
},
features);

// Run the model on the first example.
//
// Note: When possible, prepare and run multiple examples at a time.
std::vector<float> predictions;
engine->Predict(*examples, /*num_examples=*/1, &predictions);

// The semantic of "predictions" depends on the model (e.g. probabilities,
// regressive values).
```

**Note:** Categorical features can be represented as string or dense integers.
In some cases, it is better to use the "string" version, even when the feature
is effectively stored as an integer.

To minimize binary sizes, *models* and *engines* are not linked by default. It
is up to the caller to inject the dependencies to the specific model/engine (or
to include all the canonical ones using helpers):

```python
# Dependency to the Random Forest model.
// third_party / yggdrasil_decision_forests / model / random_forest

# Dependency to the Gradient Boosted Decision Tree model.
// third_party / yggdrasil_decision_forests / model / gradient_boosted_trees

# Dependency to ALL the canonical models.
// third_party / yggdrasil_decision_forests / model: all_models

# Dependency to all the canonical engines.
// third_party / yggdrasil_decision_forests: all_inference_engines
```

Note: If you get the following error: `No compatible engine available for model
RANDOM_FOREST. 1) Make sure the corresponding engine is added as a dependency,

The input example can be specified in different ways:

-   Feature by feature using the `examples->Set{Type}(example_idx, feature,
    value)` methods (like in the example above).

-   From an Yggdrasil example proto (i.e.
    `yggdrasil_decision_forests::dataset::proto::Example`) and using the
    `examples->FromProtoExample(example_proto)` method.

-   From a TensorFlow example proto (i.e. `tensorflow::Example`) and using the
    `examples->FromTensorflowExample(tf_example_proto)` method.

Note: TensorFlow example proto are inefficient. If inference speed is important
to you, the other methods are more suited.

### Serving TensorFlow Decision Forests model

[TensorFlow Decision Forests](https://github.com/tensorflow/decision-forests)
models are TensorFlow Saved model containing an Yggdrasil Decision Forests in
their `assets` subdirectory. Therefore, the Yggdrasil C++ API can be used to run
TF-DF models directly.

Following is a Colab example that trains a TF-DF model, and use it with the
Yggdrasil DF toolbox.

```python
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
# │   ├── data_spec.pb
# │   ├── done
# │   ├── gradient_boosted_trees_header.pb
# │   ├── header.pb
# │   └── nodes-00000-of-00001
# ├── keras_metadata.pb
# ├── saved_model.pb
# └── variables
#     ├── variables.data-00000-of-00001
#     └── variables.index

# Show the model structure using Yggdrasil tool box
!yggdrasil_decision_forests/cli/show_model --model=/tmp/model/assets
```

Note that the Yggdrasil model in a TF-DF model does not include any of the
pre-processing done in the TensorFlow graph. If any pre-processing is applied,
you have feed the pre-processed example to the Yggdrasil C++ API.

For example, if a TF-DF model is trained with a TF-Hub text embedding module in
the pre-processing stage, you have to feed the embeddings to the Yggdrasil C++
API.

## Advanced features

Most of the YDF advanced documentation is written in `.h` header (for the
library) and `.cc` headers (for CLI tools).
