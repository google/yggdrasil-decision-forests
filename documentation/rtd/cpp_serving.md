# Serving models in C++

The C++ API is the most efficient solution to run models.

## API

Model inference is divided into three stages:

**Stage 1**

-   The model is loaded in memory.

**Stage 2**

-   The model is compiled for fast inference. The result of the compilation is
    called an *engine*. After this stage, the model can be discarded as only the
    engine is necessary for inference.

-   The input features of the model are indexed.

-   Optionally, the memory used to store the examples and predictions is
    pre-allocated.

**Stage 3**

-   Examples are assembled. The engine is used to generated predictions.

All engines are thread safe: The same engine can be called to make predictions
in parallel from different threads at the same time without need for mutex
protection. Unless documented, engines are not multi-threaded.

The following code illustrates the three stages:

``` {note}
For a full working example with handeling of absl's status,
check
[this example](https://github.com/google/yggdrasil-decision-forests/blob/main/examples/beginner.cc).
```

**Stage 1**

```c++
#include "yggdrasil_decision_forests/learner/learner_library.h"

namespace ydf = yggdrasil_decision_forests;

// Load the model.
std::unique_ptr<ydf::model::AbstractModel> model;
CHECK_OK(LoadModel("/path/to/model", &model));
```

This was easy :).

**Stage 2**

```c++
// Compile the model into an inference engine.
const auto engine = model->BuildFastEngine().value();

// Index the input features of the model.
//
// For efficiency reasons, it is important to index the input features when
// loading the model, and not when generating predictions.
const auto& features = engine->features();
const auto feature_age = features.GetNumericalFeatureId("age").value();
const auto feature_country = features.GetCategoricalFeatureId("country").value();
const auto feature_text = features.GetCategoricalSetFeatureId("text").value()

// At this point, "model" can be discarded.
model.reset(nullptr);
```

**Stage 3**

The following example runs a batch of 2 examples through the model.

```c++
// Allocate memory for 10 examples. Alternatively, for speed-sensitive code,
// the "examples" object can be allocated in the stage 2 and reused everytime.
auto examples = engine->AllocateExamples(10);

// Set all the values to be missing. The values may then be overridden by the
// "Set*" methods. If all the values are set with "Set*" methods, "FillMissing"
// can be skipped.
examples->FillMissing(features);

// Prepare one example.
examples->SetNumerical(/*example_idx=*/0, feature_age, 30, features);
examples->SetCategorical(/*example_idx=*/0, feature_country, "UK", features);
examples->SetCategoricalSet(/*example_idx=*/0, feature_text,
  std::vector<std::string>{"hello", "world"}, features);

// Prepare another example.
examples->SetNumerical(/*example_idx=*/1, feature_age, 30, features);
examples->SetCategorical(/*example_idx=*/1, feature_country, "UK", features);
examples->SetCategoricalSet(/*example_idx=*/1, feature_text,
  std::vector<std::string>{"hello", "world"}, features);

// Run the model on the two examples.
//
// Note: When possible, prepare and run multiple examples at a time.
std::vector<float> predictions;
engine->Predict(*examples, /*num_examples=*/2, &predictions);
```

The semantic of `predictions` depends on the model (e.g. probabilities,
regressive values). `engine->NumPredictionDimension()` is the number of
predictions item for each example. For example, in the case of a three classes
classification, `engine->NumPredictionDimension()=3` and the predictions vector
contains the probability of each class for each example as follow (example
major, prediction minor):

```
probablity example 0 class 0
probablity example 0 class 1
probablity example 0 class 2
probablity example 1 class 0
probablity example 1 class 1
probablity example 1 class 2
```

## Example format

In the example above, the feature values are set using the `Set*` functions.
This solution is the most efficient. Alternatively, input examples can be set
from Yggdrasil or TensorFlow Examples:

-   From an Yggdrasil example proto (i.e.
    `yggdrasil_decision_forests::dataset::proto::Example`) and using the
    `examples->FromProtoExample(example_proto)` method.

-   From a TensorFlow example proto (i.e. `tensorflow::Example`) and using the
    `examples->FromTensorflowExample(tf_example_proto)` method.

``` {note}
TensorFlow example protos are very inefficient as the string name of each
feature is encoded for each example. If inference speed is important for your
project, use one of the other methods.
```

## Compilation

In most cases, simply link all the available YDF models and engines with the
following Bazel build rules:

```
# Dependency to all the canonical models.
//third_party/yggdrasil_decision_forests/model:all_models

# Dependency to all the canonical engines.
//third_party/yggdrasil_decision_forests:all_inference_engines
```

Alternatively, if binary size is an issue, only link the model and engine used
by your model. For example, the following rule links the random forest model
`//third_party/yggdrasil_decision_forests/model/random_forest`.

The error `No compatible engine available for model ...` indicates that you
forgot the link the engine for your model.
