---
title: YDF documentation
hide: toc
---
#

![](image/logo_v2.png){: .center .logo}

<div class="intro_text">
YDF is a library to train, evaluate, interpret, and
serve Random Forest,<br />Gradient Boosted Decision Trees, and CART decision forest
models.
</div>

<a class="getting_started_button" href="tutorial/getting_started"> Getting
Started 🧭 </a>

<div class="arguments">

<div class="argument">
<div class="column explanation">

<div class="reason">
<div class="title">A concise and modern API</div>

<div class="text">YDF allows for for rapid prototyping and development while minimizing risk of modeling errors.</div>
</div>
</div>

<div class="column illustration">
<img src="image/code_1.png">
</div>
</div>

<div class="argument">
<div class="column illustration">
<img src="image/code_2.png">
</div>

<div class="column explanation">
    <div class="reason">
        <div class="title">Deep learning composition</div>
        <div class="text">Integrated with TensorFlow, Keras, and Vertex AI.</div>
        </div>
    </div>
</div>

<div class="argument">
<div class="column explanation">

<div class="reason">
<div class="title">Cutting-edge algorithms</div>

<div class="text">Include the latest decision forest research to ensure maximum performance.</div>
</div>
</div>

<div class="column illustration">
<img src="image/compare_quality.png">
<div class="label">
Source: <a href="https://doi.org/10.1145/3580305.3599933">Yggdrasil Decision Forests: A Fast and Extensible Decision Forests Library (KDD 2023)</a>.
</div>
</div>
</div>

<div class="argument">
<div class="column illustration">
<img src="image/compare_speed.png">
<div class="label">
Source: <a href="https://doi.org/10.1145/3580305.3599933">A Comparison of Decision Forest Inference Platforms from A Database Perspective (Arxiv 2023)</a>.
</div>
</div>

<div class="column explanation">

<div class="reason">
<div class="title">Fast inference</div>

<div class="text">Compute predictions in a few microseconds. Executed in the tens of millions of times per second in Google.</div>
</div>
</div>
</div>

</div>

## Key features

*Read our KDD 2023 paper:
[Yggdrasil Decision Forests: A Fast and Extensible Decision Forests Library](https://doi.org/10.1145/3580305.3599933).
YDF is developed by Google since 2018 and powers TensorFlow Decision Forests.*

**Modeling**

-   Train [Random Forest](py_api/RandomForestLearner), [Gradient Boosted Trees](py_api/GradientBoostedTreesLearner) and [Cart](py_api/CartLearner) models.
-   Train [classification](tutorial/classification.ipynb),
    [regression](tutorial/regression.ipynb), [ranking](tutorial/ranking.ipynb),
    and [uplifting](tutorial/uplifting.ipynb) models.
-   [Plotting of decision trees](tutorial/inspecting_trees.ipynb).
-   [Model interpretation](tutorial/model_understanding.ipynb) (variable importances, partial dependence plots,
    conditional dependence plots).
-   Prediction interpretation ([counter factual](tutorial/counterfactual.ipynb), [feature variation](tutorial/prediction_understanding.ipynb)).
-   [Model evaluation](tutorial/train_and_test.ipynb) (accuracy, AUC, ROC plots, RMSE, confidence intervals,
    [cross-validation](tutorial/cross_validation.ipynb)).
-   Model [hyper-parameter tuning](tutorial/tuning.ipynb).
-   Consume natively [numerical](tutorial/numerical_feature.ipynb),
    [categorical](tutorial/categorical_feature.ipynb), boolean, tags, text, and
    missing values.
-   Consume natively [Pandas Dataframe](tutorial/pandas.ipynb), Numpy Arrays,
    [TensorFlow Datasets](tutorial/tf_dataset.ipynb), CSV files and TensorFlow
    Records.

**Serving**

-   [Benchmark](tutorial/getting_started/#benchmark-model-speed) model inference.
-   Call models in Python, [C++](tutorial/cpp.ipynb), [Go](https://github.com/google/yggdrasil-decision-forests/tree/main/yggdrasil_decision_forests/port/go), [JavaScript](https://github.com/google/yggdrasil-decision-forests/tree/main/yggdrasil_decision_forests/port/javascript), and [CLI](cli_commands).
-   Online inference with REST API with
    [TensorFlow Serving and Vertex AI](tutorial/tf_serving.ipynb).

**Advanced modeling**

-   Model composition with [TensorFlow, Keras](tutorial/compose_with_tf.ipynb),
    and Jax (coming soon).
-   [Distributed training](tutorial/distributed_training.ipynb) over billions of
    examples and hundreds of machines.
-   Cutting-edge learning algorithm such as oblique splits, honest trees,
    hessian scores, global tree optimizations, optimal categorical splits,
    categorical-set inputs, dart, extremely randomized trees.
-   [Monotonic constraints](tutorial/monotonic_feature.ipynb).
-   Consumes [multi-dimensional](tutorial/multidimensional_feature.ipynb)
    features.
-   Backward compatibility for model and learners since 2018.
-   [Edits trees](tutorial/editing_trees.ipynb) in Python.
-   [Custom loss](tutorial/custom_loss.ipynb) in Python.

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

## Next steps

Read the [🧭 Getting Started tutorial](tutorial/getting_started.ipynb). You will
learn how to train a model, interpret it, evaluate it, generate predictions,
benchmark its speed, and export it for serving.

Ask us questions on
[Github](https://github.com/google/yggdrasil-decision-forests). Googlers can
join the [internal YDF Chat](http://go/ydf-chat).

Read the [TF-DF to YDF Migration guide](tutorial/migrating_to_ydf.ipynb) to
convert a TensorFlow Decision Forests pipeline into a YDF pipeline.
