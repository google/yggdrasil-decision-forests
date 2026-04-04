# Frequently Asked Questions

![](image/cpu_forest.png){: .center}

## About YDF

### What are Decision Forests?

Decision Forests are a family of machine learning models that are easy to train
and excel at tasks involving tabular data. To learn more about the mathematics
behind them, we recommend
[Google's Decision Forests online class](https://developers.google.com/machine-learning/decision-forests).

### I want to use YDF. Where should I start?

The best place to start is with the YDF Python API in a Notebook or Colab
environment. The Python API is the most comprehensive and user-friendly way to
use YDF.

Models are portable across different APIs. For example, you can train a model
using the Python API and then export it to C++, Go, or JavaScript for efficient
serving.

### Who created YDF and when?

YDF was created and is actively developed by Google engineers in Switzerland.

Key milestones in YDF's development:

*   2017: Created as an experimental library in Google Research.
*   2018: Moved to production with a C++ API and CLI. Today, YDF models at
    Google are served tens of millions of times per second.
*   2019: Released TensorFlow Decision Forests for TensorFlow 1.
*   2020: Released TensorFlow Decision Forests for TensorFlow 2 (Keras API).
*   2021: Open-sourced and presented at Google I/O.
*   2023: Presented at KDD23 (paper).
*   2023: Launch of the standalone YDF Python API.
*   2026: Launch of the Tensorflow Export library YDF-TF.

## YDF and TF-DF

### What is YDF-TF

[YDF-TF](http://pypi.org/project/ydf-tf) is a Python library packaging the
TensorFlow inference ops of YDF. These custom ops are necessary to

-   Export YDF models to TensorFlow SavedModel format, and
-   Load exported YDF models with TensorFlow.

YDF-TF can be installed with

```bash
pip install ydf-tf
```

The package can also be included in the YDF installation by installing

```bash
pip install ydf[tensorflow]
```

### Should I use YDF or TensorFlow Decision Forests (TF-DF)?

For new projects, you should prefer YDF over TF-DF in all cases.

YDF's Python API and TF-DF share the same core C++ engine, so many features and
the models they produce are similar. However, TF-DF is constrained by its
integration with the Keras 2 API and TensorFlow. As a result, TF-DF is generally
slower, has a larger memory footprint, and is less flexible than the standalone
YDF Python API.

If you need to export to TensorFlow, install the ydf-tf package from pip.

### What is the status of TF-DF? Can I still use it?

for new use cases, we believe that YDF is a better choice than TF-DF. If export
to Tensorflow SavedModel is necessary, use the ydf-tf library to add support.
TF-DF is a production-grade library that is supported and deployed in many
products. It is still maintained, but might not receive quick compatibility
updates for new versions of TensorFlow.

Please contact the YDF team with any blockers preventing the migration from
TF-DF to YDF.

### What is the status of the Python API?

YDF's Python API has reached feature parity with TF-DF and is under active
development to introduce even more features.

### How are YDF and TF-DF different?

Both libraries are developed by the same team and use the same C++ training
code, meaning models trained by either library will be identical. YDF is the
successor to TF-DF and is significantly more feature-rich, efficient, and easier
to use.

|                  | Yggdrasil Decision Forests            | TensorFlow        |
:                  :                                       : Decision Forests  :
| ---------------- | ------------------------------------- | ----------------- |
| Model            | model.describe() produces a rich HTML | model.describe()  |
: Description      : or text report.                       : produces a simple :
:                  :                                       : text report. Does :
:                  :                                       : not work on a     :
:                  :                                       : model loaded from :
:                  :                                       : disk.             :
| Model Evaluation | model.evaluate(ds) returns a          | Each metric must  |
:                  : comprehensive evaluation report with  : be configured     :
:                  : accuracy, AUC, ROC plots, confidence  : with              :
:                  : intervals, etc.                       : model.compile()   :
:                  :                                       : before calling    :
:                  :                                       : model.evaluate(). :
:                  :                                       : Cannot produce    :
:                  :                                       : ROC curves or     :
:                  :                                       : confidence        :
:                  :                                       : intervals. Cannot :
:                  :                                       : evaluate ranking  :
:                  :                                       : or uplifting      :
:                  :                                       : models.           :
| Model Analysis   | model.analyze(ds) produces a rich     | Not available.    |
:                  : HTML report with variable             :                   :
:                  : importances, Partial Dependence Plots :                   :
:                  : (PDPs), and Conditional Expectation   :                   :
:                  : Plots (CEPs).                         :                   :
| Model            | model.benchmark(ds) measures model    | Not available.    |
: Benchmarking     : inference speed.                      :                   :
| Cross-validation | learner.cross_validation(ds) performs | Not available.    |
:                  : cross-validation and returns a rich   :                   :
:                  : evaluation report.                    :                   :
| Anomaly          | Available via Isolation Forest and    | Not available.    |
: Detection        : its extensions.                       :                   :
| Python Model     | model.predict(ds) supports multiple   | model.predict(ds) |
: Serving          : input formats (e.g., file paths,      : operates on       :
:                  : Pandas DataFrame, dictionary of NumPy : TensorFlow        :
:                  : arrays, TensorFlow Dataset).          : Datasets. You may :
:                  :                                       : need to adapt     :
:                  :                                       : feature dtypes    :
:                  :                                       : when predicting   :
:                  :                                       : with a model      :
:                  :                                       : loaded from disk. :
| TensorFlow       | model.to_tensorflow_saved_model(path) | model.save(path,  |
: Serving / Vertex : creates a valid SavedModel with an    : signature)        :
: AI               : automatically generated signature.    : requires the      :
:                  :                                       : model signature   :
:                  :                                       : to be written     :
:                  :                                       : manually.         :
| Other Model      | Models are directly usable in C++,    | Export the model  |
: Serving          : Python, Go, and JavaScript. You can   : to a TensorFlow   :
:                  : also generate serving code; for       : SavedModel using  :
:                  : example, model.to_cpp() generates C++ : model.save(),     :
:                  : serving code.                         : then use the      :
:                  :                                       : TensorFlow C++    :
:                  :                                       : API.              :
:                  :                                       : Alternatively,    :
:                  :                                       : export the model  :
:                  :                                       : to the YDF        :
:                  :                                       : format.           :
| Performance      | On small datasets, training is up to  | On small          |
:                  : 5x faster. For all dataset sizes,     : datasets,         :
:                  : model inference is up to 1000x faster : significant time  :
:                  : than TF-DF.                           : is often spent    :
:                  :                                       : reading data with :
:                  :                                       : the TensorFlow    :
:                  :                                       : Dataset pipeline. :
| Library Size     | The YDF library is ~12B.              | The TF-DF library |
:                  :                                       : is ~12MB, but it  :
:                  :                                       : requires          :
:                  :                                       : TensorFlow, which :
:                  :                                       : is ~600MB.        :
| Error Messages   | Error messages are short, high-level, | Error messages    |
:                  : and actionable.                       : are often long,   :
:                  :                                       : hard to           :
:                  :                                       : understand, and   :
:                  :                                       : related to        :
:                  :                                       : internal          :
:                  :                                       : TensorFlow        :
:                  :                                       : details like      :
:                  :                                       : Tensor shapes.    :

## Common Modeling Questions

### Adding a new column to my data changes the model. Why?

YDF training is deterministic, meaning that training a model twice on the exact
same data with the same YDF version on the same machine will produce an
identical model.

However, many parts of the training process are stochastic (random). For
example, a learner might randomly sample features using a parameter like
attribute_sampling_ratio. This process uses a pseudo-random number generator
initialized by a seed. If you add a column (even an empty one) or reorder
existing columns, you change the input to this random process, which can lead to
a different model—similar to changing the training seed.

### Can I train YDF models on a GPU or TPU?

Most YDF training and inference is done on CPU and does not use GPU or TPU.
Training of YDF models with vector sequences can use GPU. YDF models exported to
JAX can be run on GPU or TPU, thought experiments show that YDF+JAX models don't
run significantly faster than on CPU. Contact the team if you have an impactful
use case for this.

## Miscellaneous

### Which architectures does YDF support?

YDF provides pre-compiled binaries for these architectures:

*   manylinux_2_17 x86_64 (Linux)

*   macOS arm64 (Apple Silicon)

We also publish Windows binaries, though typically with a delay after a new
release.

The following architectures will likely work if you compile from source, but we
do not provide pre-compiled binaries for them at this time:

*   manylinux2014 aarch64 (Linux)

*   macOS Intel

### My model performs differently than a model from library XYZ. Why?

While most decision forest libraries implement similar algorithms, they rarely
produce identical results. The primary reason for these differences is the
choice of default hyperparameter values. For example, YDF's Gradient Boosted
Trees learner trains 300 trees with a maximum depth of 6 by default, while other
libraries may use different defaults.

YDF's strength lies in the number of available techniques and advanced features.
For a detailed comparison of model performance across libraries, please see our
paper from KDD 2023:
[Yggdrasil Decision Forests: A Fast and Extensible Decision Forests Library](https://doi.org/10.1145/3580305.3599933).

### Can I use the ydf.experimental module?

Any API under `ydf.experimental` should not be used in production. These
features are subject to change, may be unstable, or could have poor performance.

### Is the library called PYDF or YDF?

The official name of the C++ library is Yggdrasil Decision Forests, and the Pip
package is ydf. Internally, the development team sometimes calls the Python
wrapper PYDF as a nickname.

### How should I pronounce PYDF?

The preferred pronunciation is "Py-dee-eff" (/ˈpaɪˈdiˈɛf/ IPA). But since it's
an internal nickname, you don't really have to pronounce it at all.

### My question isn't answered here!

For bugs or feature requests, please open an issue on
[Github](https://github.com/google/yggdrasil-decision-forests/issues). You can
also contact the core development team at
[decision-forests-contact@google.com](mailto:decision-forests-contact@google.com).
