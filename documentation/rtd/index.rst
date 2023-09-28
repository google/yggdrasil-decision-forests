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

News
====

- *Sep, 2023* :bdg-danger:`Release` We are releasing `Temporian <https://temporian.readthedocs.io>`_, a library for preprocessing and feature engineering of temporal data.
- *Sep, 2023* :bdg-warning:`News` Start of `our Discord <https://discord.gg/D8NK9Ac6ZF>`_ server.
- *Aug, 2023* :bdg-primary:`Publication` `Yggdrasil Decision Forests: A Fast and Extensible Decision Forests Library <https://dl.acm.org/doi/10.1145/3580305.3599933>`_ at KDD'2023 (`extended version <https://arxiv.org/abs/2212.02934>`_)
- *Jul, 2023* :bdg-danger:`Release` TensorFlow Decision Forests `1.5.0 <https://github.com/tensorflow/decision-forests/releases/tag/1.5.0>`_
- *Jul, 2023* :bdg-danger:`Release` Yggdrasil Decision Forests `1.5.0 <https://github.com/google/yggdrasil-decision-forests/releases/tag/1.5.0>`_
- *May, 2023* :bdg-success:`Presentation` `Simple ML for Sheets <https://simplemlforsheets.com/>`_ at `Google IO 2023 <https://io.google/2023/program/e695ebdd-b968-4b85-98d9-0a722892e842>`_
- *Sep, 2022* :bdg-danger:`Release` `Go API <https://ydf.readthedocs.io/en/latest/go_serving.html>`_ to run Yggdrasil Decision Forest models.
- *Jul, 2022* :bdg-primary:`Publication` `Generative Trees: Adversarial and Copycat <https://proceedings.mlr.press/v162/nock22a.html>`_ at ICML'2022
- *Jun, 2022* :bdg-danger:`Release` `JavaScript API <https://ydf.readthedocs.io/en/latest/js_serving.html>`_ to train and run Yggdrasil Decision Forest models.
- *May, 2021* :bdg-success:`Presentation` `TensorFlow Decision Forests <https://www.tensorflow.org/decision_forests>`_ at `Google IO 2021 <https://io.google/2021/session/c89df7a7-0aac-4e4d-8b5d-894492bbf406/?lng=en>`_


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
