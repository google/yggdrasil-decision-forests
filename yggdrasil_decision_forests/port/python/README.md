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

## Compiling & Building

To build the Python port of YDF, install GCC-9 and run the following command
from the root of the port/python directory in the YDF repository

```sh
PYTHON_BIN=python3.9
./tools/test_pydf.sh
./tools/build_pydf.sh $PYTHON_BIN
```

## Frequently Asked Questions

*   **Is it PYDF or YDF?** The name of the library is simply ydf, and so is the
    name of the corresponding Pip package. Internally, the team sometimes uses
    the name *PYDF* because it fits so well.
*   **What is the status of PYDF?** PYDF is currently in Alpha development. Some
    parts still work well (training models and generating predictions), others
    are yet to be added. The API surface may still change without notice.
*   **How should you pronounce PYDF?** The preferred pronunciation is 
    "Py-dee-eff" / ˈpaɪˈdiˈɛf (IPA)

