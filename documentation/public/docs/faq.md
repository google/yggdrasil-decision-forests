# Frequently Asked Questions

![](image/cpu_forest.png){: .center}

## About YDF

### How can I start using YDF?

Start with the YDF Python API in an interactive notebook. The Python API is the
most comprehensive (alongside the C++ API) and user-friendly. Models trained
with the C++ API can then be exported to other APIs. For example, a model trained
with the Python API can be exported to C++, Go or JavaScript for serving. If you
want to incorporate model training into existing software, or if you need to
train thousands or millions of models, the C++ API is a good option.

TensorFlow Decision Forests is a wrapper around YDF using the Keras API. Unless
you are restricted by an existing pipeline, use the Python API instead. All new
code should use YDF.

### Who and when was YDF created?

YDF was created and continues to be developed by Google engineers in Switzerland.

Some milestones of YDF's development:

-   2017: Creation as an experimental library in Google Research
-   2018: Move to production as C++ API and CLI interface. YDF models are called
    tens of millions of times every second.
-   2019: Release of Tensorflow Decision Forests for TensorFlow 1
-   2020: Release of Tensorflow Decision Forests for TensorFlow 2 (Keras API)
-   2021: Open-sourcing and presentation at Google I/O
-   2023: Development of the standalone Python API.

## Python: YDF and TF-DF

### Should I use the YDF or TensorFlow Decision Forests (TF-DF)?

If possible, use YDF's new Python API. See below for a comparison of the
features.

YDF's Python API and TF-DF share the same C++ backend and, as a consequence,
many of their features and the models they produce are the same. However, TF-DF
is more constrained by implementing the Keras 2 API and using TensorFlow
internally. As a consequence, TF-DF is slower, bigger and less flexible than
YDF's Python API.

Note that the YDF Python API may still see minor changes while in development.
The model format and Serving APIs have been stable since 2018, see 
[Long term support](lts.md).

### What is the status of TF-DF?  Can I still use TF-DF?

TF-DF is a production-grade library, supported and deployed in many products.
It can still be used and it is actively maintained by the YDF team. However, we
believe that the pure Python API is a better fit for the majority of use cases
(see comparison below).

### Is it PYDF or YDF?

The name of the library is simply ydf, and so is the
name of the corresponding Pip package. Internally, the team sometimes uses
the name *PYDF* because it fits so well.

### What is the status of the Python API?

YDF's Python API is currently in active development. Some parts work well
(training models, generating predictions, model analysis and evaluation, model
loading, saving and exporting), others (distributed training, model editing) are
yet to be added. 

Note that the YDF Python API may still see minor changes while in development.
The model format and Serving APIs have been stable since 2018, see 
[Long term support](lts.md).

### How are YDF and TF-DF different?

Both libraries are developed by the same team and use the same training code,
which means that models trained by either library will be identical.
**YDF is the successor of TF-DF and it is both significantly more feature-rich, efficient, and easier to use than TF-DF**.


| | Yggdrasil Decision Forests | TensorFlow Decision Forests |
|---|---|---|
| Model description | `model.describe()` produces rich model description html or text report. | `model.describe()` produces a less complete text report. `model.describe()` does not work if applied on a model loaded from disk. |
| Model evaluation | `model.evaluate(ds)` evaluates a model and returs a rich model evaluation report. Metrics can also be accessed programmatically. | Each evaluation metric needs to be configured and run manually with `model.compile()` and `model.evalute()`. No evaluation report. No confidence intervals. No metrics for ranking and uplifting models. |
| Model analysis | `model.analyze(ds)` produces a rich model analysis html report. | None |
| Model benchmarking | `model.benchmark(ds)` measures and reports the model inference speed. | None |
| Cross-validation | `learner.cross_validation(ds)` performs a cross-validation and return a rich model evaluation report. | None |
| Python model serving | `model.predict(ds)` makes predictions. | `model.predict(ds)` works sometimes. However, because of limitation in the TensorFlow SavedModel format, calling `model.predict(ds)` on a model loaded from disk might require signature engineering. |
| Other model serving | Model directly available in C++, Python, CLI, go and Javascript. You can also use utilities to generate serving code: For example, call `model.to_cpp()` to generate C++ serving code. Models can be exported to a TensorFlow SavedModel with `model.to_tensorflow_saved_model(path)`. | Call `model.save(path, signature)` to generate a TensorFlow SaveModel, and use the TensorFlow C++ API to run the model in C++. Alternatively, export the model to YDF. |
| Training speed | On a small dataset, training up to 5x faster than TensorFlow Decision Forests. On all dataset sizes, model inference is up to 1000x faster than TensorFlow Decision Forests. | On a small dataset, most of the time is spent in TensorFlow dataset reading. |
| Library loading speed | The YDF library is ~4MB. | The TF-DF library is ~4MB, but it requires TensorFlow which is ~600MB. |
| Error messages | Short, high level and actionable error messages. | Long and hard to understand error messages often about Tensor shapes. |

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

### My model performs worse / better than with library XYZ, why?

While decision forest libraries implement similar algorithms, they generally
produce different results. The main cause of these discrepancies is
generally the different default hyperparameter values in different libraries.
For example, YDF trains GBTs with a maximum depth of 6 using a
divide-and-conquer learning algorithm by default. Other libraries may use other
default hyperparameters.

YDF shines by the number of available techniquing and features. Refer to our
paper our paper titled
[Yggdrasil Decision Forests: A Fast and Extensible Decision Forests Library](https://doi.org/10.1145/3580305.3599933)
in KDD 2023 for a comparison of model performances trained with different
libraries.

### How should I pronounce PYDF?

The preferred pronunciation is "Py-dee-eff" / ˈpaɪˈdiˈɛf (IPA). But since it
is an internal name, you really don't have to pronounce it at all.

### I have an important question that has not been answered by this FAQ!

You can raise issues with YDF on
[Github](https://github.com/google/yggdrasil-decision-forests/issues). 
You can also contact the core development team at [decision-forests-contact@google.com](mailto:decision-forests-contact@google.com).

