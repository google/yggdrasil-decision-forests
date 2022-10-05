<p align="center">
<img src="documentation/image/logo.png"  />
</p>

**Yggdrasil Decision Forests** (**YDF**) is a collection of state-of-the-art
algorithms for the training, serving, and interpretation of **Decision Forest**
models. The library is available in C++, CLI (command-line-interface, i.e. shell
commands), in TensorFlow under the name
[TensorFlow Decision Forests](https://github.com/tensorflow/decision-forests)
(TF-DF), Go and in Javascript (inference only). YDF is supported on Linux,
Windows, macOS, Raspberry, and Arduino (experimental).

See [the documentation 📕](https://ydf.readthedocs.org/) more details.

## Usage example

Train, evaluate, and benchmark the speed of a model in a few shell lines with
the CLI interface:

```shell
# Download YDF.
wget https://github.com/google/yggdrasil-decision-forests/releases/download/1.0.0/cli_linux.zip
unzip cli_linux.zip
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

(based on [examples/beginner.sh](examples/beginner.sh))

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

(based on [examples/beginner.cc](examples/beginner.cc))

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
[TensorFlow Decision Forests](https://github.com/tensorflow/decision-forests))

## Google IO Presentation

Yggdrasil Decision Forests powers TensorFlow Decision Forests.

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/5qgk9QJ4rdQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</center>

## Documentation & Resources

The following resources are available:

-   [📕 Documentation](https://ydf.readthedocs.io/en/latest/)
-   [Issue tracker](https://github.com/google/yggdrasil-decision-forests/issues)
-   [Changelog](https://ydf.readthedocs.io/en/latest/ydf_changelog.html)
-   [TensorFlow Decision Forest](https://github.com/tensorflow/decision-forests)
-   [Long time support](https://ydf.readthedocs.io/en/latest/lts.html)

## Contributing

Contributions to TensorFlow Decision Forests and Yggdrasil Decision Forests are
welcome. If you want to contribute, check the
[contribution guidelines](CONTRIBUTING.md).

## Credits

Yggdrasil Decision Forests and TensorFlow Decision Forests are developed by:

-   Mathieu Guillame-Bert (gbm AT google DOT com)
-   Jan Pfeifer (janpf AT google DOT com)
-   Sebastian Bruch (sebastian AT bruch DOT io)
-   Richard Stotz (richardstotz AT google DOT com)
-   Arvind Srinivasan (arvnd AT google DOT com)

## License

[Apache License 2.0](LICENSE)
