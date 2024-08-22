# Frequently Asked Questions

![](image/cpu_forest.png){: .center}

## About YDF

### What are Decision Forests?

Decision Forests are easy-to-train machine learning models that excel with
tabular data. To dive into the mathematics behind them, check out
[Google's Decision Forests online class](https://developers.google.com/machine-learning/decision-forests).

### I want to use YDF. Where should I start?

Start with the YDF Python API in an a Notebook or a Colab. The Python API is the
most comprehensive (alongside the C++ API) and user-friendly way to use YDF.

Models trained with one API can be used with other APIs. For example, a model
trained with the Python API can be exported to C++, Go or JavaScript for
serving.

### Who and when was YDF created?

YDF was created and continues to be developed by Google engineers in
Switzerland.

Some milestones of YDF's development:

-   2017: Creation as an experimental library in Google Research
-   2018: Move to production as C++ API and CLI interface. YDF models are called
    tens of millions of times every second.
-   2019: Release of Tensorflow Decision Forests for TensorFlow 1
-   2020: Release of Tensorflow Decision Forests for TensorFlow 2 (Keras API)
-   2021: Open-sourcing and presentation at Google I/O
-   2023: [Presentation](https://doi.org/10.1145/3580305.3599933) at KDD23.
-   2023: Development of the standalone Python API.

## YDF and TF-DF

### Should I use the YDF or TensorFlow Decision Forests (TF-DF)?

You should prefer YDF over TF-DF in nearly all cases.

YDF's Python API and TF-DF share the same C++ backend and, as a consequence,
many of their features and the models they produce are the same. However, TF-DF
is more constrained by implementing the Keras 2 API and using TensorFlow
internally. As a consequence, TF-DF is slower, bigger and less flexible than
YDF's Python API.

### What is the status of TF-DF? Can I still use TF-DF?

TF-DF is a production-grade library, supported and deployed in many products. It
can still be used and it is actively maintained by the YDF team. However, we
believe that the YDF Python API is a better fit for the majority of new use
cases (see comparison below).

### What is the status of the Python API?

YDF's Python API has reached feature parity with TF-DF and is in active
development for more features.

### How are YDF and TF-DF different?

Both libraries are developed by the same team and use the same training code,
which means that models trained by either library will be identical. **YDF is
the successor of TF-DF and it is both significantly more feature-rich,
efficient, and easier to use than TF-DF**.

&nbsp;                         | Yggdrasil Decision Forests                                                                                                                                                                 | TensorFlow Decision Forests
------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------
Model description              | `model.describe()` produces rich html or text model description report.                                                                                                                    | `model.describe()` produces simple text report. `model.describe()` does not work if applied on a model loaded from disk.
Model evaluation               | `model.evaluate(ds)` evaluates a model and returs a rich model evaluation report with accuracy, AUC, ROC plots, confidence intervals, etc.                                                 | Each evaluation metric needs to be configured with `model.compile()` before calling `model.evalute()`. Cannot produce ROC or confidence intervals. Cannot evaluate ranking and uplifting models.
Model analysis                 | `model.analyze(ds)` produces a rich model analysis html report with variable importances, PDPs and CEPs.                                                                                   | Not available
Model benchmarking             | `model.benchmark(ds)` measures the model inference speed.                                                                                                                                  | Not available
Cross-validation               | `learner.cross_validation(ds)` performs a cross-validation and return a rich model evaluation report.                                                                                      | Not available
Anomaly detection              | Available with Isolation Forests and its extensions.                                                                                                                                       | Not available
Python model serving           | `model.predict(ds)` makes predictions. Support many dataset formats (file paths, pandas dataframe, dictionary of numpy arrays, TensorFlow Dataset).                                        | `model.predict(ds)` on TensorFlow Datasets. When calling `model.predict(ds)` on a model loaded from disk, you might need to adapt feature dtypes.
TensorFlow Serving / Vertex AI | `model.to_tensorflow_saved_model(path)` create a valid SavedModel. The SavedModel signature is build automatically.                                                                        | `model.save(path, signature)`. The model signature should be written manually.
Other model serving            | Model directly available in C++, Python, CLI, Go and Javascript. You can also use utilities to generate serving code: For example, call `model.to_cpp()` to generate the C++ serving code. | Call `model.save(path, signature)` to generate a TensorFlow SaveModel, and use the TensorFlow C++ API to run the model in C++. Alternatively, export the model to YDF.
Training speed                 | On a small dataset, training up to 5x faster than TensorFlow Decision Forests. On all dataset sizes, model inference is up to 1000x faster than TensorFlow Decision Forests.               | On a small dataset, most of the time is spent in TensorFlow dataset reading.
Library loading speed          | The YDF library is ~9MB.                                                                                                                                                                   | The TF-DF library is ~12MB, but it requires TensorFlow which is ~600MB.
Error messages                 | Short, high level and actionable error messages.                                                                                                                                           | Long and hard to understand error messages often about Tensor shapes.

## Common modeling questions

### Adding an empty column to my model changes its quality. Why?

YDF training is deterministic, with the changes in compiler optimizations and
the random number generator implementation. This means that for a given version
of YDF, training a model twice on the same data will produce the same results.

Part of the training is stochastic. For example, the attribute_sampling_ratio
parameter is used to randomly select features. This random selection is
performed using a pseudo-random number generator that is initialized with the
training seed. Adding an empty column or shuffling the columns will change the
random generation result and, consequently, the output model. This change is
similar to changing the random seed.

## Misc

### Which architectures are supported by YDF?

YDF supports the following architectures:

-   Manylinux2014 x86_64 (to be updated to a newer version of manylinux in one of 
    the next releases).
-   MaxOS Arm64

We also publish Windows binaries, usually with some delay after the release.

The following architectures probably work, but we will not release binaries for them at this time:

-   Manylinux2014 aarch64 (to be updated to a newer version of manylinux in one
    of the next releases).
-   MacOS Intel.

### My model performs worse / better than with library XYZ, why?

While decision forest libraries implement similar algorithms, they generally
produce different results. The main cause of these discrepancies is generally
the different default hyperparameter values in different libraries. For example,
YDF trains GBTs with a maximum depth of 6 using a divide-and-conquer learning
algorithm by default. Other libraries may use other default hyperparameters.

YDF shines by the number of available techniquing and features. Refer to our
paper our paper titled
[Yggdrasil Decision Forests: A Fast and Extensible Decision Forests Library](https://doi.org/10.1145/3580305.3599933)
in KDD 2023 for a comparison of model performances trained with different
libraries.

### Is it PYDF or YDF?

The name of the library is simply `ydf`, and so is the name of the corresponding
Pip package. Internally, the team sometimes uses the name *PYDF* because it fits
so well.

### How should I pronounce PYDF?

The preferred pronunciation is "Py-dee-eff" / ˈpaɪˈdiˈɛf (IPA). But since it is
an internal name, you really don't have to pronounce it at all.

### I have an important question that has not been answered by this FAQ!

You can raise issues with YDF on
[Github](https://github.com/google/yggdrasil-decision-forests/issues). You can
also contact the core development team at
[decision-forests-contact@google.com](mailto:decision-forests-contact@google.com).
