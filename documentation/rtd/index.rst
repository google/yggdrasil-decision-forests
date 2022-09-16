
.. warning::
  This is the new documentation of YDF and TF-DF on ReadTheDocs. This documentation
  is currently being assembled, and therefore it is incomplete.
  For the documentation of Yggdrasil Decision Forests and TensorFlow Decision Forests
  visit respectively the `YDF Github page <https://github.com/google/yggdrasil-decision-forests>`_
  and the `TF-DF page <https://www.tensorflow.org/decision_forests>`_ on TensorFlow.org.


.. image:: image/ydf_logo.png
  :width: 200
  :alt: Yggdrasil Decision Forests logo
  :align: center

|

**Yggdrasil Decision Forests** (YDF) is a production grade collection algorithms, developed and used in Google, for the training, serving, and interpretation of Decision Forest
models. The library is available in C++, CLI (command-line-interface), TensorFlow (under the name TensorFlow Decision Forests;TF-DF), Javascript, and Go
(inference only). See the :doc:`features` page for the list of features.

|

.. image:: image/tfdf_logo.png
  :width: 200
  :alt: TensorFlow Decision Forests logo
  :align: center

|

**TensorFlow Decision Forests** (TF-DF) is the official port of YDF to
TensorFlow using the `Kera API <https://keras.io/>`_. TF-DF makes it easy to train
and use decision forests in the TensorFlow ecosystem.



Content
=======

.. toctree::
  :maxdepth: 1

  intro_df

.. toctree::
  :maxdepth: 1
  :caption: CLI API

  Quick start <cli_quick_start>
  cli_commands

.. toctree::
  :maxdepth: 1
  :caption: TF-DF / Python API

  Quick Star <https://www.tensorflow.org/decision_forests/tutorials/beginner_colab>
  Install <https://www.tensorflow.org/decision_forests/installation>
  Tutorials <https://www.tensorflow.org/decision_forests/tutorials>

.. toctree::
  :maxdepth: 1
  :caption: Reference

  apis
  hyper_parameters
  features
  lts

.. toctree::
  :maxdepth: 1
  :caption: Deploy a model

  serving_apis

.. toctree::
  :maxdepth: 1
  :caption: Resources

  ydf_changelog
  Google's Decision Forest class <https://developers.google.com/machine-learning/decision-forests>
  YDF on Github <https://github.com/google/yggdrasil-decision-forests>
  TF-DF on Github <https://github.com/tensorflow/decision-forests>
  TF-DF on tf.org <https://www.tensorflow.org/decision_forests>
  contributing
  credit
