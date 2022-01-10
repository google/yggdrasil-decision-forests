<p align="center">
<img src="documentation/image/logo.png"  />
</p>

**Yggdrasil Decision Forests** (**YDF**) is a collection of state-of-the-art
algorithms for the training, serving and interpretation of **Decision Forest**
models. The library is developed in C++ and available in C++, CLI
(command-line-interface, i.e. shell commands) and in TensorFlow under the name
[TensorFlow Decision Forests](https://github.com/tensorflow/decision-forests)
(TF-DF).

Developing models in TF-DF and productionizing them (possibly including
re-training) in C++ with YDF allows both for a flexible and fast development and
an efficient and safe serving.

## Usage example

Train, evaluate and benchmark the speed of a model in a few shell lines with the
CLI interface:

```shell
# Training configuration
echo 'label:"my_label" learner:"RANDOM_FOREST" ' > config.pbtxt
# Scan the dataset
infer_dataspec --dataset="csv:train.csv" --output="spec.pbtxt"
# Train a model
train --dataset="csv:train.csv" --dataspec="spec.pbtxt" --config="config.pbtxt" --output="my_model"
# Evaluate the model
evaluate --dataset="csv:test.csv" --model="my_model" > evaluation.txt
# Benchmark the speed of the model
benchmark_inference --dataset="csv:test.csv" --model="my_model" > benchmark.txt
```

(see the [examples/beginner.sh](examples/beginner.sh) for more details)

or use the C++ interface:

```c++
auto dataset_path = "csv:/train@10";
// Training configuration
TrainingConfig train_config;
train_config.set_learner("RANDOM_FOREST");
train_config.set_task(Task::CLASSIFICATION);
train_config.set_label("my_label");
// Scan the dataset
DataSpecification spec;
CreateDataSpec(dataset_path, false, {}, &spec);
// Train a model
std::unique_ptr<AbstractLearner> learner;
GetLearner(train_config, &learner);
auto model = learner->Train(dataset_path, spec);
// Export the model
SaveModel("my_model", model.get());
```

(see the [examples/beginner.cc](examples/beginner.cc) for more details)

or use the Keras/Python interface of TensorFlow Decision Forests:

```python
import tensorflow_decision_forests as tfdf
import pandas as pd
# Load the dataset in a Pandas dataframe.
train_df = pd.read_csv("project/train.csv")
# Convert the dataset into a TensorFlow dataset.
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="my_label")
# Train the model
model = tfdf.keras.RandomForestModel()
model.fit(train_ds)
# Export a SavedModel.
model.save("project/model")
```

(see
[TensorFlow Decision Forests](https://github.com/tensorflow/decision-forests)
for more details)

## Documentation & Resources

The following resources are available:

-   [Examples](examples) (C++ and CLI)
-   [User manual](documentation/user_manual.md)
-   [CLI one pager](documentation/cli.txt)
-   [Developer manual](documentation/developer_manual.md)
-   [List of learning algorithms](documentation/learners.md)
-   [Issue tracker](https://github.com/google/yggdrasil-decision-forests/issues)
-   [Known issues](documentation/known_issues.md)
-   [Changelog](CHANGELOG.md)
-   [TensorFlow Decision Forest](https://github.com/tensorflow/decision-forests)

## Installation from pre-compiled binaries

Download one of the
[build releases](https://github.com/google/yggdrasil-decision-forests/releases),
and then run `examples/beginner.{sh,bat}`.

## Installation from Source

On linux, install
[Bazel](https://docs.bazel.build/versions/4.0.0/getting-started.html) and run:

```shell
git clone https://github.com/google/yggdrasil-decision-forests.git
cd yggdrasil_decision_forests
bazel build //yggdrasil_decision_forests/cli:all --config=linux_cpp17 --config=linux_avx2

# Then, run the example:
examples/beginner.sh
```

See the [installation](documentation/installation.md) page for more details,
troubleshooting and alternative installation solutions.

Yggdrasil was successfully compiled and run on:

-   Linux Debian 5
-   Windows 10
-   MacOS 10
-   Raspberry Pi 4 Rev 2

Inference of Yggdrasil models is also available on:

-   *[Experimental; No support]* Arduino Uno R3 (see
    [project](https://github.com/achoum/ardwino-tensorflow-decision-forests))

**Note:** Tell us if you were able to compile and run Yggdrasil on any other
architecture :).

## Long-time-support commitments

### Inference and serving

-   The serving code is isolated from the rest of the framework (i.e., training,
    evaluation) and has minimal dependencies.
-   Changes to serving-related code are guaranteed to be backward compatible.
-   Model inference is deterministic: the same example is guaranteed to yield
    the same prediction.
-   Learners and models are extensively tested, including integration testing on
    real datasets; and, there exists no execution path in the serving code that
    crashes as a result of an error; Instead, in case of failure (e.g.,
    malformed input example), the inference code returns a util::Status.

### Training

-   Hyper-parameters' semantic is never modified.
-   The default value of hyper-parameters is never modified.
-   The default value of a newly-introduced hyper-parameter is set in such a way
    that the hyper-parameter is effectively disabled.

## Quality Assurance

The following mechanisms will be put in place to ensure the quality of the
library:

-   Peer-reviewing.
-   Unit testing.
-   Training benchmarks with ranges of acceptable evaluation metrics.
-   Sanitizers.

## Contributing

Contributions to TensorFlow Decision Forests and Yggdrasil Decision Forests are
welcome. If you want to contribute, make sure to review the
[user manual](documentation/user_manual.md),
[developer manual](documentation/developer_manual.md) and
[contribution guidelines](CONTRIBUTING.md).

## Credits

TensorFlow Decision Forests was developed by:

-   Mathieu Guillame-Bert (gbm AT google DOT com)
-   Jan Pfeifer (janpf AT google DOT com)
-   Sebastian Bruch (sebastian AT bruch DOT io)
-   Arvind Srinivasan (arvnd AT google DOT com)

## License

[Apache License 2.0](LICENSE)
