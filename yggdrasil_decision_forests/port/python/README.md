# Port of Yggdrasil / TensorFlow Decision Forests for Python

The Python port of Yggdrasil Decision is a light-weight wrapper around Yggdrasil
Decision Forests. It allows direct, fast access to YDF's methods and it also
offers advanced import / export, evaluation and inspection methods. While the
package is called YDF, the wrapping code is sometimes lovingly called *PYDF*.

It is not a replacement for its sister project 
[Tensorflow Decision Forests](https://github.com/tensorflow/decision-forests) 
(TF-DF). Instead, it complements TF-DF for use cases that cannot be solved 
through the Keras API.

## Installation

To install YDF, in Python, simply grab the package from pip:

```
pip install ydf
```

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

## Compiling & Building

To build the Python port of YDF, install Bazel, GCC 9 and run the following
command from the root of the port/python directory in the YDF repository

```sh
PYTHON_BIN=python3.9
./tools/test_pydf.sh
./tools/build_pydf.sh $PYTHON_BIN
```

Browse the `tools/` directory for more build helpers.

## Frequently Asked Questions

*   **Is it PYDF or YDF?** The name of the library is simply ydf, and so is the
    name of the corresponding Pip package. Internally, the team sometimes uses
    the name *PYDF* because it fits so well.
*   **What is the status of PYDF?** PYDF is currently in Alpha development. Most
    parts already work well (training, evaluation, predicting, export), some new
    features are yet to come. The API surface is mostly stable but may still 
    change without notice.
*   **Where is the documentation for PYDF?** The documentation is
    available on https://ydf.readthedocs.org.
*   **How should I pronounce PYDF?** The preferred pronunciation is 
    "Py-dee-eff" / ˈpaɪˈdiˈɛf (IPA)

