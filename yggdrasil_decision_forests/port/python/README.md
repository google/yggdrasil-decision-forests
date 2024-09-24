# Port of Yggdrasil / TensorFlow Decision Forests for Python

The Python port of Yggdrasil Decision is a light-weight wrapper around Yggdrasil
Decision Forests. It allows direct, fast access to YDF's methods and it also
offers advanced import / export, evaluation and inspection methods. While the
package is called YDF, the wrapping code is sometimes lovingly called *PYDF*.

YDF is the successor of
[Tensorflow Decision Forests](https://github.com/tensorflow/decision-forests) 
(TF-DF). TF-DF is still maintained, but new projects should choose YDF for
improved performance, better model quality and more features.

## Installation

To install YDF, in Python, simply grab the package from pip:

```
pip install ydf
```

For build instructions, see INSTALLATION.md.

## Usage Example

```python
import ydf
import pandas as pd

ds_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset"
train_ds = pd.read_csv(f"{ds_path}/adult_train.csv")
test_ds = pd.read_csv(f"{ds_path}/adult_test.csv")

model = ydf.GradientBoostedTreesLearner(label="income").train(train_ds)

print(model.evaluate(test_ds))

model.save("my_model")

loaded_model = ydf.load_model("my_model")
```

## Frequently Asked Questions

See the [FAQ](https://ydf.readthedocs.io/en/latest/faq/) in the documentation.

