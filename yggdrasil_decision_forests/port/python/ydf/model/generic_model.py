# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic YDF model definition."""

import dataclasses
import enum
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypeVar, Union

from absl import logging
import numpy as np

from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.metric import metric_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.cc import ydf
from ydf.dataset import dataset
from ydf.dataset import dataspec
from ydf.metric import metric
from ydf.model import analysis
from ydf.model import model_metadata
from ydf.model import optimizer_logs
from ydf.model import template_cpp_export
from ydf.utils import html
from ydf.utils import log
from yggdrasil_decision_forests.utils import model_analysis_pb2


@enum.unique
class Task(enum.Enum):
  """Task solved by a model.

  Usage example:

  ```python
  learner = ydf.RandomForestLearner(label="income",
                                    task=ydf.Task.CLASSIFICATION)
  model = learner.train(dataset)
  assert model.task() == ydf.Task.CLASSIFICATION
  ```
  Not all tasks are compatible with all learners and/or hyperparameters. For
  more information, please see the documentation for tutorials on the individual
  tasks.


  Attributes:
    CLASSIFICATION: Predict a categorical label i.e., an item of an enumeration.
    REGRESSION: Predict a numerical label i.e., a quantity.
    RANKING: Rank items by label values. The label is expected to be between 0
      and 4 with NDCG semantic (0: completely unrelated, 4: perfect match).
    CATEGORICAL_UPLIFT: Predicts the incremental impact of a treatment on a
      categorical outcome.
    NUMERICAL_UPLIFT: Predicts the incremental impact of a treatment on a
      numerical outcome.
    ANOMALY_DETECTION: Predicts if an instance is similar to the majority of the
      training data or anomalous (a.k.a. an outlier). An anomaly detection
      prediction is a value between 0 and 1, where 0 indicates the possible most
      normal instance and 1 indicates the most possible anomalous instance.
  """

  CLASSIFICATION = "CLASSIFICATION"
  REGRESSION = "REGRESSION"
  RANKING = "RANKING"
  CATEGORICAL_UPLIFT = "CATEGORICAL_UPLIFT"
  NUMERICAL_UPLIFT = "NUMERICAL_UPLIFT"
  ANOMALY_DETECTION = "ANOMALY_DETECTION"

  def _to_proto_type(self) -> abstract_model_pb2.Task:
    if self in TASK_TO_PROTO:
      return TASK_TO_PROTO[self]
    else:
      raise NotImplementedError(f"Unsupported task {self}")

  @classmethod
  def _from_proto_type(cls, task: abstract_model_pb2.Task):
    task = PROTO_TO_TASK.get(task)
    if task is None:
      raise NotImplementedError(f"Unsupported task {task}")
    return task


# Mappings between task enum in python and in protobuffer and vice versa.
TASK_TO_PROTO = {
    Task.CLASSIFICATION: abstract_model_pb2.CLASSIFICATION,
    Task.REGRESSION: abstract_model_pb2.REGRESSION,
    Task.RANKING: abstract_model_pb2.RANKING,
    Task.CATEGORICAL_UPLIFT: abstract_model_pb2.CATEGORICAL_UPLIFT,
    Task.NUMERICAL_UPLIFT: abstract_model_pb2.NUMERICAL_UPLIFT,
    Task.ANOMALY_DETECTION: abstract_model_pb2.ANOMALY_DETECTION,
}
PROTO_TO_TASK = {v: k for k, v in TASK_TO_PROTO.items()}


@dataclasses.dataclass(frozen=True)
class ModelIOOptions:
  """Advanced options for saving and loading YDF models.

  Attributes:
      file_prefix: Optional prefix for the model. File prefixes allow multiple
        models to exist in the same folder. Doing so is heavily DISCOURAGED
        outside of edge cases. When loading a model, the prefix, if not
        specified, is auto-detected if possible. When saving a model, the empty
        string is used as file prefix unless it is explicitly specified.
  """

  file_prefix: Optional[str] = None


@enum.unique
class NodeFormat(enum.Enum):
  # pyformat: disable
  """Serialization format for a model.

  Determines the storage format for nodes.

  Attributes:
    BLOB_SEQUENCE: Default format for the public version of YDF.
  """
  # pyformat: enable

  BLOB_SEQUENCE = enum.auto()


@dataclasses.dataclass
class InputFeature:
  """An input feature of a model.

  Attributes:
    name: Feature name. Unique for a model.
    semantic: Semantic of the feature.
    column_idx: Index of the column corresponding to the feature in the
      dataspec.
  """

  name: str
  semantic: dataspec.Semantic
  column_idx: int


class GenericModel:
  """Abstract superclass for all YDF models."""

  def __init__(self, raw_model: ydf.GenericCCModel):
    self._model = raw_model

  def name(self) -> str:
    """Returns the name of the model type."""
    return self._model.name()

  def task(self) -> Task:
    """Task solved by the model."""
    return Task._from_proto_type(self._model.task())  # pylint: disable=protected-access

  def metadata(self) -> model_metadata.ModelMetadata:
    """Metadata associated with the model.

    A model's metadata contains information stored with the model that does not
    influence the model's predictions (e.g. data created). When distributing a
    model for wide release, it may be useful to clear / modify the model
    metadata with `model.set_metadata(ydf.ModelMetadata())`.

    Returns:
      The model's metadata.
    """
    return model_metadata.ModelMetadata._from_proto_type(self._model.metadata())  # pylint:disable=protected-access

  def set_metadata(self, metadata: model_metadata.ModelMetadata):
    """Sets the model metadata."""
    self._model.set_metadata(metadata._to_proto_type())  # pylint:disable=protected-access

  def describe(
      self,
      output_format: Literal["auto", "text", "notebook", "html"] = "auto",
      full_details: bool = False,
  ) -> Union[str, html.HtmlNotebookDisplay]:
    """Description of the model.

    Args:
      output_format: Format of the display: - auto: Use the "notebook" format if
        executed in an IPython notebook / Colab. Otherwise, use the "text"
        format. - text: Text description of the model. - html: Html description
        of the model. - notebook: Html description of the model displayed in a
        notebook cell.
      full_details: Should the full model be printed. This can be large.

    Returns:
      The model description.
    """

    if output_format == "auto":
      output_format = "text" if log.is_direct_output() else "notebook"

    with log.cc_log_context():
      description = self._model.Describe(full_details, output_format == "text")
      if output_format == "notebook":
        return html.HtmlNotebookDisplay(description)
      else:
        return description

  def data_spec(self) -> data_spec_pb2.DataSpecification:
    """Returns the data spec used for train the model."""
    return self._model.data_spec()

  def __str__(self) -> str:
    return f"""\
Model: {self.name()}
Task: {self.task().name}
Class: ydf.{self.__class__.__name__}
Use `model.describe()` for more details
"""

  def benchmark(
      self,
      ds: dataset.InputDataset,
      benchmark_duration: float = 3,
      warmup_duration: float = 1,
      batch_size: int = 100,
  ) -> ydf.BenchmarkInferenceCCResult:
    """Benchmark the inference speed of the model on the given dataset.

    This benchmark creates batched predictions on the given dataset using the
    C++ API of Yggdrasil Decision Forests. Note that inference times using other
    APIs or on different machines will be different. A serving template for the
    C++ API can be generated with `model.to_cpp()`.

    Args:
      ds: Dataset to perform the benchmark on.
      benchmark_duration: Total duration of the benchmark in seconds. Note that
        this number is only indicative and the actual duration of the benchmark
        may be shorter or longer. This parameter must be > 0.
      warmup_duration: Total duration of the warmup runs before the benchmark in
        seconds. During the warmup phase, the benchmark is run without being
        timed. This allows warming up caches. The benchmark will always run at
        least one batch for warmup. This parameter must be > 0.
       batch_size: Size of batches when feeding examples to the inference
         engines. The impact of this parameter on the results depends on the
         architecture running the benchmark (notably, cache sizes).

    Returns:
      Benchmark results.
    """
    if benchmark_duration <= 0:
      raise ValueError(
          "The duration of the benchmark must be positive, got"
          f" {benchmark_duration}"
      )
    if warmup_duration <= 0:
      raise ValueError(
          "The duration of the warmup phase must be positive, got"
          f" {warmup_duration}."
      )
    if batch_size <= 0:
      raise ValueError(
          f"The batch size of the benchmark must be positive, got {batch_size}."
      )

    with log.cc_log_context():
      vds = dataset.create_vertical_dataset(
          ds,
          data_spec=self._model.data_spec(),
          required_columns=self.input_feature_names(),
      )
      result = self._model.Benchmark(
          vds._dataset, benchmark_duration, warmup_duration, batch_size  # pylint: disable=protected-access
      )
    return result

  def save(self, path, advanced_options=ModelIOOptions()) -> None:
    """Save the model to disk.

    YDF uses a proprietary model format for saving models. A model consists of
    multiple files located in the same directory.
    A directory should only contain a single YDF model. See `advanced_options`
    for more information.

    YDF models can also be exported to other formats, see
    `to_tensorflow_saved_model()` and `to_cpp()` for details.

    YDF saves some metadata inside the model, see `model.metadata()` for
    details. Before distributing a model to the world, consider removing
    metadata with `model.set_metadata(ydf.ModelMetadata())`.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train a Random Forest model
    df = pd.read_csv("my_dataset.csv")
    model = ydf.RandomForestLearner().train(df)

    # Save the model to disk
    model.save("/models/my_model")
    ```

    Args:
      path: Path to directory to store the model in.
      advanced_options: Advanced options for saving models.
    """
    # Warn if the user is trying to save to a nonempty directory without
    # prefixing the model.
    if advanced_options.file_prefix is not None:
      if os.path.exists(path):
        if os.path.isdir(path):
          with os.scandir(path) as it:
            if any(it):
              logging.warning(
                  "The directory %s to save the model to is not empty,"
                  " which can lead to model corruption. Specify an empty or"
                  " non-existing directory to save the model to, or use"
                  " `advanced_options` to specify a file prefix for the model.",
                  path,
              )

    with log.cc_log_context():
      self._model.Save(path, advanced_options.file_prefix)

  def predict(self, data: dataset.InputDataset) -> np.ndarray:
    """Returns the predictions of the model on the given dataset.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").train(train_ds)

    test_ds = pd.read_csv("test.csv")
    predictions = model.predict(test_ds)
    ```

    Args:
      data: Dataset. Can be a dictionary of list or numpy array of values,
        Pandas DataFrame, or a VerticalDataset. If the dataset contains the
        label column, that column is ignored.
    """
    with log.cc_log_context():
      # The data spec contains the label / weights /  ranking group / uplift
      # treatment column, but those are not required for making predictions.
      ds = dataset.create_vertical_dataset(
          data,
          data_spec=self._model.data_spec(),
          required_columns=self.input_feature_names(),
      )
      result = self._model.Predict(ds._dataset)  # pylint: disable=protected-access
    return result

  def evaluate(
      self,
      data: dataset.InputDataset,
      *,
      bootstrapping: Union[bool, int] = False,
      weighted: bool = False,
  ) -> metric.Evaluation:
    """Evaluates the quality of a model on a dataset.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").train(train_ds)

    test_ds = pd.read_csv("test.csv")
    evaluation = model.evaluates(test_ds)
    ```

    In a notebook, if a cell returns an evaluation object, this evaluation will
    be as a rich html with plots:

    ```
    evaluation = model.evaluate(test_ds)
    evaluation
    ```

    Args:
      data: Dataset. Can be a dictionary of list or numpy array of values,
        Pandas DataFrame, or a VerticalDataset.
      bootstrapping: Controls whether bootstrapping is used to evaluate the
        confidence intervals and statistical tests (i.e., all the metrics ending
        with "[B]"). If set to false, bootstrapping is disabled. If set to true,
        bootstrapping is enabled and 2000 bootstrapping samples are used. If set
        to an integer, it specifies the number of bootstrapping samples to use.
        In this case, if the number is less than 100, an error is raised as
        bootstrapping will not yield useful results.
      weighted: If true, the evaluation is weighted according to the training
        weights. If false, the evaluation is non-weighted. b/351279797: Change
        default to weights=True.

    Returns:
      Model evaluation.
    """

    with log.cc_log_context():
      ds = dataset.create_vertical_dataset(
          data, data_spec=self._model.data_spec()
      )

      if isinstance(bootstrapping, bool):
        bootstrapping_samples = 2000 if bootstrapping else -1
      elif isinstance(bootstrapping, int) and bootstrapping >= 100:
        bootstrapping_samples = bootstrapping
      else:
        raise ValueError(
            "bootstrapping argument should be boolean or an integer greater"
            " than 100 as bootstrapping will not yield useful results. Got"
            f" {bootstrapping!r} instead"
        )

      options_proto = metric_pb2.EvaluationOptions(
          bootstrapping_samples=bootstrapping_samples,
          task=self.task()._to_proto_type(),  # pylint: disable=protected-access
      )

      evaluation_proto = self._model.Evaluate(
          ds._dataset, options_proto, weighted=weighted
      )  # pylint: disable=protected-access
    return metric.Evaluation(evaluation_proto)

  def analyze_prediction(
      self,
      single_example: dataset.InputDataset,
  ) -> analysis.PredictionAnalysis:
    """Understands a single prediction of the model.

    Note: To explain the model as a whole, use `model.analyze` instead.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").train(train_ds)

    test_ds = pd.read_csv("test.csv")

    # We want to explain the model prediction on the first test example.
    selected_example = test_ds.iloc[:1]

    analysis = model.analyze_prediction(selected_example, test_ds)

    # Display the analysis in a notebook.
    analysis
    ```

    Args:
      single_example: Example to explain. Can be a dictionary of lists or numpy
        arrays of values, Pandas DataFrame, or a VerticalDataset.

    Returns:
      Prediction explanation.
    """

    with log.cc_log_context():
      ds = dataset.create_vertical_dataset(
          single_example, data_spec=self._model.data_spec()
      )

      options_proto = model_analysis_pb2.PredictionAnalysisOptions()
      analysis_proto = self._model.AnalyzePrediction(ds._dataset, options_proto)  # pylint: disable=protected-access
      return analysis.PredictionAnalysis(analysis_proto, options_proto)

  def analyze(
      self,
      data: dataset.InputDataset,
      sampling: float = 1.0,
      num_bins: int = 50,
      partial_depepence_plot: bool = True,
      conditional_expectation_plot: bool = True,
      permutation_variable_importance_rounds: int = 1,
      num_threads: int = 6,
  ) -> analysis.Analysis:
    """Analyzes a model on a test dataset.

    An analysis contains structual information about the model (e.g., variable
    importances), and the information about the application of the model on the
    given dataset (e.g. partial dependence plots).

    For a large dataset (many examples and / or features), computing the
    analysis can take significant time.

    While some information might be valid, it is generatly not recommended to
    analyze a model on its training dataset.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").train(train_ds)

    test_ds = pd.read_csv("test.csv")
    analysis = model.analyze(test_ds)

    # Display the analysis in a notebook.
    analysis
    ```

    Args:
      data: Dataset. Can be a dictionary of list or numpy array of values,
        Pandas DataFrame, or a VerticalDataset.
      sampling: Ratio of examples to use for the analysis. The analysis can be
        expensive to compute. On large datasets, use a small sampling value e.g.
        0.01.
      num_bins: Number of bins used to accumulate statistics. A large value
        increase the resolution of the plots but takes more time to compute.
      partial_depepence_plot: Compute partial dependency plots a.k.a PDPs.
        Expensive to compute.
      conditional_expectation_plot: Compute the conditional expectation plots
        a.k.a. CEP. Cheap to compute.
      permutation_variable_importance_rounds: If >1, computes permutation
        variable importances using "permutation_variable_importance_rounds"
        rounds. The most rounds the more accurate the results. Using a single
        round is often acceptable i.e. permutation_variable_importance_rounds=1.
        If permutation_variable_importance_rounds=0, disables the computation of
        permutation variable importances.
      num_threads: Number of threads to use to compute the analysis.

    Returns:
      Model analysis.
    """

    with log.cc_log_context():
      ds = dataset.create_vertical_dataset(
          data, data_spec=self._model.data_spec()
      )

      options_proto = model_analysis_pb2.Options(
          num_threads=num_threads,
          pdp=model_analysis_pb2.Options.PlotConfig(
              enabled=partial_depepence_plot,
              example_sampling=sampling,
              num_numerical_bins=num_bins,
          ),
          cep=model_analysis_pb2.Options.PlotConfig(
              enabled=conditional_expectation_plot,
              example_sampling=sampling,
              num_numerical_bins=num_bins,
          ),
          permuted_variable_importance=model_analysis_pb2.Options.PermutedVariableImportance(
              enabled=permutation_variable_importance_rounds > 0,
              num_rounds=permutation_variable_importance_rounds,
          ),
          include_model_structural_variable_importances=True,
      )

      analysis_proto = self._model.Analyze(ds._dataset, options_proto)  # pylint: disable=protected-access
      return analysis.Analysis(analysis_proto, options_proto)

  def to_cpp(self, key: str = "my_model") -> str:
    """Generates the code of a .h file to run the model in C++.

    How to use this function:

    1. Copy the output of this function in a new .h file.
      open("model.h", "w").write(model.to_cpp())
    2. If you use Bazel/Blaze, create a rule with the dependencies:
      //third_party/absl/status:statusor
      //third_party/absl/strings
      //external/ydf_cc/yggdrasil_decision_forests/api:serving
    3. In your C++ code, include the .h file and call the model with:
      // Load the model (to do only once).
      namespace ydf = yggdrasil_decision_forests;
      const auto model = ydf::exported_model_123::Load(<path to model>);
      // Run the model
      predictions = model.Predict();
    4. The generated "Predict" function takes no inputs. Instead, it fills the
      input features with placeholder values. Therefore, you will want to add
      your input as arguments to the "Predict" function, and use it to populate
      the "examples->Set..." section accordingly.
    5. (Bonus) You can further optimize the inference speed by pre-allocating
      and re-using the examples and predictions for each thread running the
      model.

    This documentation is also available in the header of the generated content
    for more details.

    Args:
      key: Name of the model. Used to define the c++ namespace of the model.

    Returns:
      String containing an example header for running the model in C++.
    """
    return template_cpp_export.template(
        key, self._model.data_spec(), self._model.input_features()
    )

  # TODO: Change default value of "mode" before 1.0 release.
  def to_tensorflow_saved_model(  # pylint: disable=dangerous-default-value
      self,
      path: str,
      input_model_signature_fn: Any = None,
      *,
      mode: Literal["keras", "tf"] = "keras",
      feature_dtypes: Dict[str, "export_tf.TFDType"] = {},  # pytype: disable=name-error
      servo_api: bool = False,
      feed_example_proto: bool = False,
      pre_processing: Optional[Callable] = None,  # pylint: disable=g-bare-generic
      post_processing: Optional[Callable] = None,  # pylint: disable=g-bare-generic
      temp_dir: Optional[str] = None,
      tensor_specs: Optional[Dict[str, Any]] = None,
      feature_specs: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Exports the model as a TensorFlow Saved model.

    This function requires TensorFlow and TensorFlow Decision Forests to be
    installed. Install them by running the command `pip install
    tensorflow_decision_forests`. The generated SavedModel model relies on the
    TensorFlow Decision Forests Custom Inference Op. This Op is available by
    default in various platforms such as Servomatic, TensorFlow Serving, Vertex
    AI, and TensorFlow.js.

    Usage example:

    ```python
    !pip install tensorflow_decision_forests

    import ydf
    import numpy as np
    import tensorflow as tf

    # Train a model.
    model = ydf.RandomForestLearner(label="l").train({
        "f1": np.random.random(size=100),
        "f2": np.random.random(size=100).astype(dtype=np.float32),
        "l": np.random.randint(2, size=100),
    })

    # Export the model to the TensorFlow SavedModel format.
    # The model can be executed with Servomatic, TensorFlow Serving and
    # Vertex AI.
    model.to_tensorflow_saved_model(path="/tmp/my_model", mode="tf")

    # The model can also be loaded in TensorFlow and executed locally.

    # Load the TensorFlow Saved model.
    tf_model = tf.saved_model.load("/tmp/my_model")

    # Make predictions
    tf_predictions = tf_model({
        "f1": tf.constant(np.random.random(size=10)),
        "f2": tf.constant(np.random.random(size=10), dtype=tf.float32),
    })
    ```

    TensorFlow SavedModel do not cast automatically feature values. For
    instance, a model trained with a dtype=float32 semantic=numerical feature,
    will require for this feature to be fed as float32 numbers during inference.
    You can override the dtype of a feature with the `feature_dtypes` argument:

    ```python
    model.to_tensorflow_saved_model(
        path="/tmp/my_model",
        mode="tf",
        # "f1" is fed as an tf.int64 instead of tf.float64
        feature_dtypes={"f1": tf.int64},
    )
    ```

    Some TensorFlow Serving or Servomatic pipelines rely on feed examples as
    serialized TensorFlow Example proto (instead of raw tensor values) and/or
    wrap the model raw output (e.g. probability predictions) into a special
    structure (called the Serving API). You can create models compatible with
    those two convensions with `feed_example_proto=True` and `servo_api=True`
    respectively:

    ```python
    model.to_tensorflow_saved_model(
        path="/tmp/my_model",
        mode="tf",
        feed_example_proto=True,
        servo_api=True
    )
    ```

    If your model requires some data preprocessing or post-processing, you can
    express them as a @tf.function or a tf module and pass them to the
    `pre_processing` and `post_processing` arguments respectively.

    Warning: When exporting a SavedModel, YDF infers the model signature using
    the dtype of the features observed during training. If the signature of the
    pre_processing function is different than the signature of the model (e.g.,
    the processing creates a new feature), you need to specify the tensor specs
    (`tensor_specs`; if `feed_example_proto=False`) or feature spec
    (`feature_specs`; if `feed_example_proto=True`) argument:

    ```python
    # Define a pre-processing function
    @tf.function
    def pre_processing(raw_features):
      features = {**raw_features}
      # Create a new feature.
      features["sin_f1"] = tf.sin(features["f1"])
      # Remove a feature
      del features["f1"]
      return features

    # Create Numpy dataset
    raw_dataset = {
        "f1": np.random.random(size=100),
        "f2": np.random.random(size=100),
        "l": np.random.randint(2, size=100),
    }

    # Apply the preprocessing on the training dataset.
    processed_dataset = (
        tf.data.Dataset.from_tensor_slices(raw_dataset)
        .batch(128)  # The batch size has no impact on the model.
        .map(preprocessing)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Train a model on the pre-processed dataset.
    ydf_model = specialized_learners.RandomForestLearner(
        label="l",
        task=generic_learner.Task.CLASSIFICATION,
    ).train(processed_dataset)

    # Export the model to a raw SavedModel model with the pre-processing
    model.to_tensorflow_saved_model(
        path="/tmp/my_model",
        mode="tf",
        feed_example_proto=False,
        pre_processing=pre_processing,
        tensor_specs{
            "f1": tf.TensorSpec(shape=[None], name="f1", dtype=tf.float64),
            "f2": tf.TensorSpec(shape=[None], name="f2", dtype=tf.float64),
        }
    )

    # Export the model to a SavedModel consuming serialized tf examples with the
    # pre-processing
    model.to_tensorflow_saved_model(
        path="/tmp/my_model",
        mode="tf",
        feed_example_proto=True,
        pre_processing=pre_processing,
        feature_specs={
            "f1": tf.io.FixedLenFeature(
                shape=[], dtype=tf.float32, default_value=math.nan
            ),
            "f2": tf.io.FixedLenFeature(
                shape=[], dtype=tf.float32, default_value=math.nan
            ),
        }
    )
    ```

    For more flexibility, use the method `to_tensorflow_function` instead of
    `to_tensorflow_saved_model`.

    Args:
      path: Path to store the Tensorflow Decision Forests model.
      input_model_signature_fn: A lambda that returns the
        (Dense,Sparse,Ragged)TensorSpec (or structure of TensorSpec e.g.
        dictionary, list) corresponding to input signature of the model. If not
        specified, the input model signature is created by
        `tfdf.keras.build_default_input_model_signature`. For example, specify
        `input_model_signature_fn` if an numerical input feature (which is
        consumed as DenseTensorSpec(float32) by default) will be feed
        differently (e.g. RaggedTensor(int64)). Only compatible with
        mode="keras".
      mode: How is the YDF converted into a TensorFlow SavedModel. 1) mode =
        "keras" (default): Turn the model into a Keras 2 model using TensorFlow
        Decision Forests, and then save it with `tf_keras.models.save_model`. 2)
        mode = "tf" (recommended; will become default): Turn the model into a
        TensorFlow Module, and save it with `tf.saved_model.save`.
      feature_dtypes: Mapping from feature name to TensorFlow dtype. Use this
        mapping to feature dtype. For instance, numerical features are encoded
        with tf.float32 by default. If you plan on feeding tf.float64 or
        tf.int32, use `feature_dtype` to specify it. `feature_dtypes` is ignored
        if `tensor_specs` is set. If set, disables the automatic signature
        extraction on `pre_processing` (if `pre_processing` is also set). Only
        compatible with mode="tf".
      servo_api: If true, adds a SavedModel signature to make the model
        compatible with the `Classify` or `Regress` servo APIs. Only compatible
        with mode="tf". If false, outputs the raw model predictions.
      feed_example_proto: If false, the model expects for the input features to
        be provided as TensorFlow values. This is most efficient way to make
        predictions. If true, the model expects for the input featurs to be
        provided as a binary serialized TensorFlow Example proto. This is the
        format expected by VertexAI and most TensorFlow Serving pipelines.
      pre_processing: Optional TensorFlow function or module to apply on the
        input features before applying the model. If the `pre_processing`
        function has been traced (i.e., the function has been called once with
        actual data and contains a concrete instance in its cache), this
        signature is extracted and used as signature of the SavedModel. Only
        compatible with mode="tf".
      post_processing: Optional TensorFlow function or module to apply on the
        model predictions. Only compatible with mode="tf".
      temp_dir: Temporary directory used during the conversion. If None
        (default), uses `tempfile.mkdtemp` default temporary directory.
      tensor_specs: Optional dictionary of `tf.TensorSpec` that define the input
        features of the model to export. If not provided, the TensorSpecs are
        automatically generated based on the model features seen during
        training. This means that "tensor_specs" is only necessary when using a
        "pre_processing" argument that expects different features than what the
        model was trained with. This argument is ignored when exporting model
        with `feed_example_proto=True` as in this case, the TensorSpecs are
        defined by the `tf.io.parse_example` parsing feature specs. Only
        compatible with mode="tf".
      feature_specs: Optional dictionary of `tf.io.parse_example` parsing
        feature specs e.g. `tf.io.FixedLenFeature` or `tf.io.RaggedFeature`. If
        not provided, the praising feature specs are automatically generated
        based on the model features seen during training. This means that
        "feature_specs" is only necessary when using a "pre_processing" argument
        that expects different features than what the model was trained with.
        This argument is ignored when exporting model with
        `feed_example_proto=False`. Only compatible with mode="tf".
    """

    if mode == "keras":
      log.warning(
          "Calling `to_tensorflow_saved_model(mode='keras', ...)`. Use"
          " `to_tensorflow_saved_model(mode='tf', ...)` instead. mode='tf' is"
          " more efficient, has better compatibility, and offers more options."
          " Starting June 2024, `mode='tf'` will become the default value.",
          message_id=log.WarningMessage.TO_TF_SAVED_MODEL_KERAS_MODE,
      )

    _get_export_tf().ydf_model_to_tensorflow_saved_model(
        ydf_model=self,
        path=path,
        input_model_signature_fn=input_model_signature_fn,
        mode=mode,
        feature_dtypes=feature_dtypes,
        servo_api=servo_api,
        feed_example_proto=feed_example_proto,
        pre_processing=pre_processing,
        post_processing=post_processing,
        temp_dir=temp_dir,
        tensor_specs=tensor_specs,
        feature_specs=feature_specs,
    )

  def to_tensorflow_function(  # pytype: disable=name-error
      self,
      temp_dir: Optional[str] = None,
      can_be_saved: bool = True,
      squeeze_binary_classification: bool = True,
  ) -> "tensorflow.Module":
    """Converts the YDF model into a @tf.function callable TensorFlow Module.

    The output module can be composed with other TensorFlow operations,
    including other models serialized with `to_tensorflow_function`.

    This function requires TensorFlow and TensorFlow Decision Forests to be
    installed. You can install them using the command `pip install
    tensorflow_decision_forests`. The generated SavedModel model relies on the
    TensorFlow Decision Forests Custom Inference Op. This Op is available by
    default in various platforms such as Servomatic, TensorFlow Serving, Vertex
    AI, and TensorFlow.js.

    Usage example:

    ```python
    !pip install tensorflow_decision_forests

    import ydf
    import numpy as np
    import tensorflow as tf

    # Train a model.
    model = ydf.RandomForestLearner(label="l").train({
        "f1": np.random.random(size=100),
        "f2": np.random.random(size=100),
        "l": np.random.randint(2, size=100),
    })

    # Convert model to a TF module.
    tf_model = model.to_tensorflow_function()

    # Make predictions with the TF module.
    tf_predictions = tf_model({
        "f1": tf.constant([0, 0.5, 1]),
        "f2": tf.constant([1, 0, 0.5]),
    })
    ```

    Args:
      temp_dir: Temporary directory used during the conversion. If None
        (default), uses `tempfile.mkdtemp` default temporary directory.
      can_be_saved: If can_be_saved = True (default), the returned module can be
        saved using `tf.saved_model.save`. In this case, files created in
        temporary directory during the conversion are not removed when
        `to_tensorflow_function` exit, and those files should still be present
        when calling `tf.saved_model.save`. If can_be_saved = False, the files
        created in the temporary directory during conversion are immediately
        removed, and the returned object cannot be serialized with
        `tf.saved_model.save`.
      squeeze_binary_classification: If true (default), in case of binary
        classification, outputs a tensor of shape [num examples] containing the
        probability of the positive class. If false, in case of binary
        classification, outputs a tensorflow of shape [num examples, 2]
        containing the probability of both the negative and positive classes.
        Has no effect on non-binary classification models.

    Returns:
      A TensorFlow @tf.function.
    """

    return _get_export_tf().ydf_model_to_tf_function(
        ydf_model=self,
        temp_dir=temp_dir,
        can_be_saved=can_be_saved,
        squeeze_binary_classification=squeeze_binary_classification,
    )

  def to_jax_function(  # pytype: disable=name-error
      self,
      jit: bool = True,
      apply_activation: bool = True,
      leaves_as_params: bool = False,
  ) -> "export_jax.JaxModel":
    """Converts the YDF model into a JAX function.

    Usage example:

    ```python
    import ydf
    import numpy as np
    import jax.numpy as jnp

    # Train a model.
    model = ydf.GradientBoostedTreesLearner(label="l").train({
        "f1": np.random.random(size=100),
        "f2": np.random.random(size=100),
        "l": np.random.randint(2, size=100),
    })

    # Convert model to a JAX function.
    jax_model = model.to_jax_function()

    # Make predictions with the JAX function.
    jax_predictions = jax_model.predict({
        "f1": jnp.array([0, 0.5, 1]),
        "f2": jnp.array([1, 0, 0.5]),
    })
    ```

    TODO: Document the encoder and jax params.

    Args:
      jit: If true, compiles the function with @jax.jit.
      apply_activation: Should the activation function, if any, be applied on
        the model output.
      leaves_as_params: If true, exports the leaf values as learnable
        parameters. In this case, `params` is set in the returned value, and it
        should be passed to `predict(feature_values, params)`.

    Returns:
      A dataclass containing the JAX prediction function (`predict`) and
      optionnaly the model parameteres (`params`) and feature encoder
      (`encoder`).
    """

    return _get_export_jax().to_jax_function(
        model=self,
        jit=jit,
        apply_activation=apply_activation,
        leaves_as_params=leaves_as_params,
    )

  def update_with_jax_params(self, params: Dict[str, Any]):
    """Updates the model with JAX params as created by `to_jax_function`.

    Usage example:

    ```python
    import ydf
    import numpy as np
    import jax.numpy as jnp

    # Train a model with YDF
    dataset = {
        "f1": np.random.random(size=100),
        "f2": np.random.random(size=100),
        "l": np.random.randint(2, size=100),
    }
    model = ydf.GradientBoostedTreesLearner(label="l").train(dataset)

    # Convert model to a JAX function with leave values as parameters.
    jax_model = model.to_jax_function(
        leaves_as_params=True,
        apply_activation=True)
    # Note: The learnable model parameter are in `jax_model.params`.

    # Finetune the model parameters with your own logic.
    jax_model.params = fine_tune_model(jax_model.params, ...)

    # Update the YDF model with the finetuned parameters
    model.update_with_jax_params(jax_model.params)

    # Make predictions with the finetuned YDF model
    predictions = model.predict(dataset)

    # Save the YDF model
    model.save("/tmp/my_ydf_model")
    ```

    Args:
      params: Learnable parameter of the model generated with `to_jax_function`.
    """
    _get_export_jax().update_with_jax_params(model=self, params=params)

  def hyperparameter_optimizer_logs(
      self,
  ) -> Optional[optimizer_logs.OptimizerLogs]:
    """Returns the logs of the hyper-parameter tuning.

    If the model is not trained with hyper-parameter tuning, returns None.
    """
    proto_logs = self._model.hyperparameter_optimizer_logs()
    if proto_logs is None:
      return None
    return optimizer_logs.proto_optimizer_logs_to_optimizer_logs(proto_logs)

  def variable_importances(self) -> Dict[str, List[Tuple[float, str]]]:
    """Variable importances to measure the impact of features on the model.

    Variable importances generally indicates how much a variable (feature)
    contributes to the model predictions or quality. Different Variable
    importances have different semantics and are generally not comparable.

    The variable importances returned by `variable_importances()` depends on the
    learning algorithm and its hyper-parameters. For example, the hyperparameter
    `compute_oob_variable_importances=True` of the Random Forest learner enables
    the computation of permutation out-of-bag variable importances.

    # TODO: Add variable importances to documentation.

    Features are sorted by decreasing importance.

    Usage example:

    ```python
    # Train a Random Forest. Enable the computation of OOB (out-of-bag) variable
    # importances.
    model = ydf.RandomForestModel(compute_oob_variable_importances=True,
                                  label=...).train(ds)
    # List the available variable importances.
    print(model.variable_importances().keys())

    # Show a specific variable importance.
    model.variable_importances()["MEAN_DECREASE_IN_ACCURACY"]
    >> [("bill_length_mm", 0.0713061951754389),
        ("island", 0.007298519736842035),
        ("flipper_length_mm", 0.004505893640351366),
    ...
    ```

    Returns:
      Variable importances.
    """
    variable_importances = {}
    # Collect the variable importances stored in the model.
    for (
        name,
        importance_set,
    ) in self._model.VariableImportances().items():
      variable_importances[name] = [
          (src.importance, self.data_spec().columns[src.attribute_idx].name)
          for src in importance_set.variable_importances
      ]
    return variable_importances

  def label_col_idx(self) -> int:
    return self._model.label_col_idx()

  def label(self) -> str:
    """Name of the label column."""
    return self.data_spec().columns[self.label_col_idx()].name

  def label_classes(self) -> List[str]:
    """Returns the label classes for a classification model; fails otherwise."""
    if self.task() != Task.CLASSIFICATION:
      raise ValueError(
          "Label classes are only available for classification models. This"
          f" model has type {self.task().name}"
      )
    label_column = self.data_spec().columns[self._model.label_col_idx()]
    if label_column.type != data_spec_pb2.CATEGORICAL:
      semantic = dataspec.Semantic.from_proto_type(label_column.type)
      raise ValueError(
          "Categorical type expected for classification label."
          f" Got {semantic} instead."
      )

    if label_column.categorical.is_already_integerized:
      log.info(
          "The label column is integerized. This is expected for models trained"
          " with TensorFlow Decision Forests."
      )

    # The first element is the "out-of-vocabulary" that is not used in labels.
    return dataspec.categorical_column_dictionary_to_list(label_column)[1:]

  def input_feature_names(self) -> List[str]:
    """Returns the names of the input features.

    The features are sorted in increasing order of column_idx.
    """

    dataspec_columns = self.data_spec().columns
    return [dataspec_columns[idx].name for idx in self._model.input_features()]

  def input_features(self) -> Sequence[InputFeature]:
    """Returns the input features of the model.

    The features are sorted in increasing order of column_idx.
    """
    dataspec_columns = self.data_spec().columns
    return [
        InputFeature(
            name=dataspec_columns[column_idx].name,
            semantic=dataspec.Semantic.from_proto_type(
                dataspec_columns[column_idx].type
            ),
            column_idx=column_idx,
        )
        for column_idx in self._model.input_features()
    ]

  def self_evaluation(self) -> metric.Evaluation:
    """Returns the model's self-evaluation.

    Different models use different methods for self-evaluation. Notably, Random
    Forests use OOB evaluation and Gradient Boosted Trees use evaluation on the
    validation dataset. Therefore, self-evaluations are not comparable between
    different model types.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    model = ydf.GradientBoostedTreesLearner(label="label").train(train_ds)

    self_evaluation = model.self_evaluation()
    # In an interactive Python environment, print a rich evaluation report.
    self_evaluation
    ```
    """
    raise NotImplementedError(
        "Self-evaluation is not available for this model type."
    )

  def list_compatible_engines(self) -> Sequence[str]:
    """Lists the inference engines compatible with the model.

    The engines are sorted to likely-fastest to  likely-slowest.

    Returns:
      List of compatible engines.
    """
    return self._model.ListCompatibleEngines()

  def force_engine(self, engine_name: Optional[str]) -> None:
    """Forces the engines used by the model.

    If not specified (i.e., None; default value), the fastest compatible engine
    (i.e., the first value returned from "list_compatible_engines") is used for
    all model inferences (e.g., model.predict, model.evaluate).

    If passing a non-existing or non-compatible engine, the next model inference
    (e.g., model.predict, model.evaluate) will fail.

    Args:
      engine_name: Name of a compatible engine or None to automatically select
        the fastest engine.
    """
    self._model.ForceEngine(engine_name)


def from_sklearn(
    sklearn_model: Any,
    label_name: str = "label",
    feature_name: str = "features",
) -> GenericModel:
  """Converts a tree-based scikit-learn model to a YDF model.

  Usage example:

  ```python
  import ydf
  from sklearn import datasets
  from sklearn import tree

  # Train a SKLearn model
  X, y = datasets.make_classification()
  skl_model = tree.DecisionTreeClassifier().fit(X, y)

  # Convert the SKLearn model to a YDF model
  ydf_model = ydf.from_sklearn(skl_model)

  # Make predictions with the YDF model
  ydf_predictions = ydf_model.predict({"features": X})

  # Analyse the YDF model
  ydf_model.analyze({"features": X})
  ```

  Currently supported models are:
  *   sklearn.tree.DecisionTreeClassifier
  *   sklearn.tree.DecisionTreeRegressor
  *   sklearn.tree.ExtraTreeClassifier
  *   sklearn.tree.ExtraTreeRegressor
  *   sklearn.ensemble.RandomForestClassifier
  *   sklearn.ensemble.RandomForestRegressor
  *   sklearn.ensemble.ExtraTreesClassifier
  *   sklearn.ensemble.ExtraTreesRegressor
  *   sklearn.ensemble.GradientBoostingRegressor
  *   sklearn.ensemble.IsolationForest

  Unlike YDF, Scikit-learn does not name features and labels. Use the fields
  `label_name` and `feature_name` to specify the name of the columns in the YDF
  model.

  Additionally, only single-label classification and scalar regression are
  supported (e.g. multivariate regression models will not convert).

  Args:
    sklearn_model: the scikit-learn tree based model to be converted.
    label_name: Name of the multi-dimensional feature in the output YDF model.
    feature_name: Name of the label in the output YDF model.

  Returns:
    a YDF Model that emulates the provided scikit-learn model.
  """
  return _get_export_sklearn().from_sklearn(
      sklearn_model=sklearn_model,
      label_name=label_name,
      feature_name=feature_name,
  )


def _get_export_jax():
  try:
    from ydf.model import export_jax  # pylint: disable=g-import-not-at-top,import-outside-toplevel # pytype: disable=import-error

    return export_jax
  except ImportError as exc:
    raise ValueError(
        '"jax" is needed by this function. Make sure it installed and try'
        " again. See https://jax.readthedocs.io/en/latest/installation.html"
    ) from exc


def _get_export_tf():
  try:
    from ydf.model import export_tf  # pylint: disable=g-import-not-at-top,import-outside-toplevel # pytype: disable=import-error

    return export_tf
  except ImportError as exc:
    raise ValueError(
        '"tensorflow_decision_forests" is needed by this function. Make sure '
        "it installed and try again. If using pip, run `pip install"
        " tensorflow_decision_forests`."
    ) from exc


def _get_export_sklearn():
  try:
    from ydf.model import export_sklearn  # pylint: disable=g-import-not-at-top,import-outside-toplevel # pytype: disable=import-error

    return export_sklearn
  except ImportError as exc:
    raise ValueError(
        '"scikit-learn" is needed by this function. Make sure '
        "it installed and try again. If using pip, run `pip install"
        " scikit-learn`."
    ) from exc


ModelType = TypeVar("ModelType", bound=GenericModel)
