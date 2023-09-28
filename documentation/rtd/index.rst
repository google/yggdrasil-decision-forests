.. image:: image/ydf_logo.png
  :width: 200
  :alt: Yggdrasil Decision Forests logo
  :align: center

|

**Yggdrasil Decision Forests** (YDF) is a production-grade collection of
algorithms for the training, serving, and interpretation of decision forest
models. YDF is open-source and is available in C++, command-line interface
(CLI), TensorFlow (under the name
`TensorFlow Decision Forests <https://github.com/tensorflow/decision-forests>`_ ;
TF-DF), JavaScript (inference only), and Go (inference only). YDF is supported on Linux, Windows,
macOS, Raspberry Pi, and Arduino (experimental).

For details about YDF design, read our KDD 2023 paper
`Yggdrasil Decision Forests: A Fast and Extensible Decision Forests Library <https://doi.org/10.1145/3580305.3599933>`_ .
See also the `extended version <https://arxiv.org/abs/2212.02934>`_  with
additional details about the experimental evaluation of the library.

Features
========

-   Random Forest, Gradient Boosted Trees, CART, and variations such as
    Dart, Extremely randomized trees.
-   Classification, regression, ranking and uplifting.
-   Model evaluation e.g. accuracy, auc, roc, auuc, pr-auc, confidence
    boundaries, ndgc.
-   Model analysis e.g. pdp, cep, variable importance, model plotting, structure
    analysis.
-   Native support for numerical, categorical, boolean, categorical-set (e.g.
    text) features.
-   Native support for missing values.
-   State of the art tree learning features e.g. oblique split, honest tree,
    hessian score, global tree optimization.
-   Distributed training.
-   Automatic hyper-parameter tuning.
-   Fast model inference e.g. vpred, quick-scorer extended.
-   Cross compatible API and models: C++, CLI, Go, JavaScript and Python.

See the `feature list <https://ydf.readthedocs.io/en/latest/features.html>`_ for
more details.

About TensorFlow Decision Forests
=================================

`TensorFlow Decision Forests <https://www.tensorflow.org/decision_forests>`_ is a library for training, evaluating, interpreting, and inferring decision forest models in TensorFlow.
TensorFlow Decision Forests uses Yggdrasil Decision Forests for model training.
TensorFlow Decision Forests models are compatible with Yggdrasil Decision Forests.

.. image:: image/tfdf_logo.png
  :width: 200
  :alt: TensorFlow Decision Forests logo
  :align: center


Content
=======

.. toctree::
  :maxdepth: 1

  intro_df

.. toctree::
  :maxdepth: 1
  :caption: CLI API

  Quick start <cli_quick_start>
  Installation <cli_install>
  cli_commands
  User manual <cli_user_manual>

.. toctree::
  :maxdepth: 1
  :caption: C++ API

  Example <https://github.com/google/yggdrasil-decision-forests/tree/main/examples/standalone>

.. toctree::
  :maxdepth: 1
  :caption: TF-DF / Python API

  Quick Start <https://www.tensorflow.org/decision_forests/tutorials/beginner_colab>
  Installation <https://www.tensorflow.org/decision_forests/installation>
  Tutorials <https://www.tensorflow.org/decision_forests/tutorials>

.. toctree::
  :maxdepth: 1
  :caption: Reference

  apis
  improve_model
  hyper_parameters
  early_stopping
  metrics
  What are decision forests <https://developers.google.com/machine-learning/decision-forests>
  features
  lts

.. toctree::
  :maxdepth: 1
  :caption: Deploy a model

  serving_apis
  C++ <cpp_serving>
  Go <go_serving>
  JavaScript/Wasm <js_serving>
  TF Serving <tf_serving>
  Python <https://www.tensorflow.org/decision_forests/tutorials/predict_colab>
  Benchmark <benchmark_inference>
  convert_model

.. toctree::
  :maxdepth: 1
  :caption: Resources

  ydf_changelog
  YDF on Github <https://github.com/google/yggdrasil-decision-forests>
  TF-DF on Github <https://github.com/tensorflow/decision-forests>
  TF-DF on tf.org <https://www.tensorflow.org/decision_forests>
  contributing
  credit
