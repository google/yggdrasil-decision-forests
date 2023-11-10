<p align="center">
<img src="documentation/image/logo.png"  />
</p>

**Yggdrasil Decision Forests** (YDF) is a production-grade collection of
algorithms developed in Google Switzerland 🏔️ since 2018 for the training,
serving, and interpretation of decision forest models. YDF is available in
Python, C++, CLI, in TensorFlow under the name
[TensorFlow Decision Forests](https://github.com/tensorflow/decision-forests),
JavaScript (inference only), and Go (inference only).

To learn more about YDF, see [the documentation](https://ydf.readthedocs.org/).

For more information on the design of YDF, see our paper at KDD 2023:
[Yggdrasil Decision Forests: A Fast and Extensible Decision Forests Library](https://doi.org/10.1145/3580305.3599933).

## Key features

-   A simple API for training, evaluation and serving of decision forests
    models.
-   Supports Random Forest, Gradient Boosted Trees and Carts, and advanced
    learning algorithm such as oblique splits, honest trees, hessian and
    non-hessian scores, and global tree optimizations.
-   Train classification, regression, ranking, and uplifting models.
-   Fast model inference in cpu (microseconds / example / cpu-core).
-   Supports distributed training over billions of examples.
-   Serving in Python, C++, TensorFlow Serving, Go, JavaScript, and CLI.
-   Rich report for model description (e.g., training logs, plot trees),
    analysis (e.g., variable importances, partial dependence plots, conditional
    dependence plots), evaluation (e.g., accuracy, AUC, ROC plots, RMSE,
    confidence intervals), tuning (trials configuration and scores), and
    cross-validation.
-   Natively consumes numerical, categorical, boolean, text, and missing values.
-   Backward compatibility for model and learners since 2018.
-   Consumes Pandas Dataframes, Numpy arrays, TensorFlow Dataset and CSV files.

## Installation

To install YDF in Python from [PyPi](https://pypi.org/project/ydf/), run:

```shell
pip install ydf
```

## Usage example

Example with the Python API.

```python
import ydf
import pandas as pd

train_ds = pd.read_csv("adult_train.csv")
test_ds = pd.read_csv("adult_test.csv")

# Train a model
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

The same model can be trained in Python using TensorFlow Decision Forests as
follows:

```python
import tensorflow_decision_forests as tfdf
import pandas as pd

# Load dataset in a Pandas dataframe.
train_df = pd.read_csv("project/train.csv")

# Convert dataset into a TensorFlow dataset.
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="my_label")

# Train model
model = tfdf.keras.RandomForestModel()
model.fit(train_ds)

# Export model.
model.save("project/model")
```

## Next steps

Check the [Getting Started tutorial 🧭](tutorial/getting_started.ipynb).

## Google I/O Presentation

Yggdrasil Decision Forests powers TensorFlow Decision Forests.

<div align="center">
    <a href="https://youtu.be/5qgk9QJ4rdQ">
        <img src="https://img.youtube.com/vi/5qgk9QJ4rdQ/0.jpg"></img>
    </a>
</div>

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
