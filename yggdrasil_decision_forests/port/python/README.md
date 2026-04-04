# YDF - Yggdrasil Decision Forests for Python

**YDF** is a library for training, serving, and interpreting decision forest
models. It acts as a lightweight, efficient wrapper around the C++
[Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests)
library.

It provides fast access to core methods along with advanced features for model
import/export, evaluation, and inspection.

YDF is the official successor to
[TensorFlow Decision Forests (TF-DF)](https://github.com/tensorflow/decision-forests)
and is recommended for new projects due to its superior performance and
features.

## Installation

Install YDF from PyPI:

```bash
pip install ydf
```

For detailed build instructions, see [INSTALLATION.md](INSTALLATION.md).

## Usage Example

```python
import ydf
import pandas as pd

# Load dataset
ds_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset"
train_ds = pd.read_csv(f"{ds_path}/adult_train.csv")
test_ds = pd.read_csv(f"{ds_path}/adult_test.csv")

# Train a Gradient Boosted Trees model
model = ydf.GradientBoostedTreesLearner(label="income").train(train_ds)

# Evaluate the model
print(model.evaluate(test_ds))

# Save the model
model.save("my_model")

# Load the model
loaded_model = ydf.load_model("my_model")
```

## Documentation

For more information, visit the
[YDF Documentation](https://ydf.readthedocs.io/).

Frequently asked questions are available in the
[FAQ](https://ydf.readthedocs.io/en/latest/faq/).

