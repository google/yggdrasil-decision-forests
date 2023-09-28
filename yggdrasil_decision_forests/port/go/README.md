# Port of Yggdrasil / TensorFlow Decision Forests for Go (Golang)

With this library you can run inference on Yggdrasil or Tensorflow Decision
Forests models.

This is still EXPERIMENTAL, there is a chance of the API still changing. If this
works for most, and no one sees any issues, we'll remove this notice and commit
to this API.

## Native inference in Go (Golang)

This directory contains an implementation of Yggdrasil Decision Forests
inference written in Go. The API is similar to the
[C++ inference API](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/rtd/cli_user_manual.md).

### Limitations

-   Only a subset of the models are supported. Currently only binary
    classification gradient boosted trees models are supported.
    Reach out to the team if you need Go support to some other type of model.
-   Currently, this Go implementation is slower (~2x) than the C++
    implementation. The difference of speed can be measured by comparing both
    the c++ and go benchmarks (`cli/benchmark_inference` and
    `ports/go/cli/benchmark`). This is in large part because this implementation
    uses the straight forward implementation. Again reach out to the team
    if this becomes a bottleneck, it could be sped up. 

### Usage example

We assume an existing model trained with one of the APIs. For example, a model
trained with
[TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests).

**Prepare the model**

The first steps are to load the model and get a "serving" (inference) engine
for your model. These steps only need to be done once per model.

```go
import (
    model_io "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/io/canonical"
    "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/serving"
)

// Loads the model in memory.
model, err := model_io.LoadModel(runfiles.Path(modelPath))

// Compile / optimize the model for serving. After this statement, "model" can
// be discarded, only the prepared "engine" is needed.
engine, err := serving.NewEngine(model)

// When running a model trained with the TensorFlow Decision Forests API (in
// TensorFlow python, as opposed from the command-line), use the
// `NewEngineWithCompatibility` method instead. This is a temporary
// compatibility issue, and will be resolved soon, when "NewEngine" will
// automatically apply the correct compatibility.
engine, err := serving.NewEngineWithCompatibility(model, example.CompatibilityTensorFlowDecisionForests)

// Indices for the input features of the model.
// If "used" is false, the feature is not used by the model.
featureAge, used := engine.Features().NumericalFeatures["age"]
featureCountry, used := engine.Features().CategoricalFeatures["country"]
```

With the model loaded, and the engine created one can do as many inferences as 
desired, using the code below. Only the engine is needed now, the model can be
discarded.

Inference involves resetting the input features variable to the default value
("missing value" in this case, or any other value), setting the features one
by one, then calling the engine to do a (or a batch of) inference.

```go
// Allocate a batch of 10 examples. A batch can be re-used for speed-sensitive
// code. In this case, "examples.Clear()" should be called in between usages.
examples := engine.AllocateExamples(10)

// Allocate the predictions for of 10 examples.
// Note: In this example "predictions" is a []float32 with 10 elements.
predictions := engine.AllocatePredictions(10)

// Set all the feature values as missing. Values can be overridden with the
// "Set*" methods. If all the values are set with "Set*" methods, "FillMissing"
// can be skipped.
examples.FillMissing()

// Set the value of just two examples.
examples.SetNumerical( /*example_idx=*/ 0, featureAge, 30)
examples.SetCategorical( /*example_idx=*/ 0, featureCountry, "UK")
examples.SetNumerical( /*example_idx=*/ 1, featureAge, 40)
examples.SetCategorical( /*example_idx=*/ 1, featureCountry, "JP")

// Generate the predictions of the model on the first two examples.
engine.Predict(examples, /*num_examples=*/ 2, predictions)

// We assume a binary classification model. In this case, the prediction
// contains one value for each example. This value is the probability of the
// positive class.
assert(engine.OutputDim() == 1)
fmt.Println("Probability of true class: %v and %v",
    predictions[0], predictions[1])
```

See `cli/benchmark_inference.go` for a more detailed example.
