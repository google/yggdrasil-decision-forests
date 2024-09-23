<p align="center">
<img src="documentation/public/docs/image/logo_v2.png"  />
</p>

[![PyPI](https://img.shields.io/pypi/v/ydf.svg?style=flat-square)](https://pypi.org/project/ydf/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Static](https://img.shields.io/static/v1?label=docs&message=stable&style=flat-square)](https://ydf.readthedocs.io/)
[![Static](https://img.shields.io/static/v1?label=docs&message=dev&style=flat-square)](https://ydf.readthedocs.io/en/latest/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/ydf?style=flat-square)](https://pepy.tech/project/ydf)

**YDF** (Yggdrasil Decision Forests) is a library to train, evaluate, interpret,
and serve Random Forest, Gradient Boosted Decision Trees, CART and Isolation
forest models.

See the [documentation](https://ydf.readthedocs.org/) for more information on
YDF.

## Installation

To install YDF from [PyPI](https://pypi.org/project/ydf/), run:

```shell
pip install ydf -U
```

## Usage example

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/yggdrasil-decision-forests/blob/main/documentation/public/docs/tutorial/usage_example.ipynb)

```python
import ydf
import pandas as pd

# Load dataset with Pandas
ds_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset/"
train_ds = pd.read_csv(ds_path + "adult_train.csv")
test_ds = pd.read_csv(ds_path + "adult_test.csv")

# Train a Gradient Boosted Trees model
model = ydf.GradientBoostedTreesLearner(label="income").train(train_ds)

# Look at a model (input features, training logs, structure, etc.)
model.describe()

# Evaluate a model (e.g. roc, accuracy, confusion matrix, confidence intervals)
model.evaluate(test_ds)

# Generate predictions
model.predict(test_ds)

# Analyse a model (e.g. partial dependence plot, variable importance)
model.analyze(test_ds)

# Benchmark the inference speed of a model
model.benchmark(test_ds)

# Save the model
model.save("/tmp/my_model")
```

Example with the C++ API.

```c++
auto dataset_path = "csv:train.csv";

// List columns in training dataset
DataSpecification spec;
CreateDataSpec(dataset_path, false, {}, &spec);

// Create a training configuration
TrainingConfig train_config;
train_config.set_learner("RANDOM_FOREST");
train_config.set_task(Task::CLASSIFICATION);
train_config.set_label("my_label");

// Train model
std::unique_ptr<AbstractLearner> learner;
GetLearner(train_config, &learner);
auto model = learner->Train(dataset_path, spec);

// Export model
SaveModel("my_model", model.get());
```

(based on [examples/beginner.cc](examples/beginner.cc))

## Next steps

Check the
[Getting Started tutorial 🧭](https://ydf.readthedocs.io/en/stable/tutorial/getting_started/).

## Citation

If you us Yggdrasil Decision Forests in a scientific publication, please cite
the following paper:
[Yggdrasil Decision Forests: A Fast and Extensible Decision Forests Library](https://doi.org/10.1145/3580305.3599933).

**Bibtex**

```
@inproceedings{GBBSP23,
  author       = {Mathieu Guillame{-}Bert and
                  Sebastian Bruch and
                  Richard Stotz and
                  Jan Pfeifer},
  title        = {Yggdrasil Decision Forests: {A} Fast and Extensible Decision Forests
                  Library},
  booktitle    = {Proceedings of the 29th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, {KDD} 2023, Long Beach, CA, USA, August 6-10, 2023},
  pages        = {4068--4077},
  year         = {2023},
  url          = {https://doi.org/10.1145/3580305.3599933},
  doi          = {10.1145/3580305.3599933},
}
```

**Raw**

Yggdrasil Decision Forests: A Fast and Extensible Decision Forests Library,
Guillame-Bert et al., KDD 2023: 4068-4077. doi:10.1145/3580305.3599933

## Contact

You can contact the core development team at
[decision-forests-contact@google.com](mailto:decision-forests-contact@google.com).

## Credits

Yggdrasil Decision Forests and TensorFlow Decision Forests are developed by:

-   Mathieu Guillame-Bert (gbm AT google DOT com)
-   Jan Pfeifer (janpf AT google DOT com)
-   Sebastian Bruch (sebastian AT bruch DOT io)
-   Richard Stotz (richardstotz AT google DOT com)
-   Arvind Srinivasan (arvnd AT google DOT com)

## Contributing

Contributions to TensorFlow Decision Forests and Yggdrasil Decision Forests are
welcome. If you want to contribute, check the
[contribution guidelines](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE)
