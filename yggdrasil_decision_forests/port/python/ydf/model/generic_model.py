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

import abc
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
from ydf.model import feature_selector_logs
from ydf.model import model_metadata
from ydf.model import optimizer_logs
from ydf.model import template_cpp_export
from ydf.utils import concurrency
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
    RANKING: Rank items by label values. When using default NDCG settings, the
      label is expected to be between 0 and 4 with NDCG semantic (0: completely
      unrelated, 4: perfect match).
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
      outside of edge cases. When loading a model, the prefix, if not specified,
      is auto-detected if possible. When saving a model, the empty string is
      used as file prefix unless it is explicitly specified.
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
  BLOB_SEQUENCE_GZIP = enum.auto()


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


class GenericModel(abc.ABC):
  """Abstract superclass for all YDF models."""

  def __str__(self) -> str:
    return f"""\
Model: {self.name()}
Task: {self.task().name}
Class: ydf.{self.__class__.__name__}
Use `model.describe()` for more details
"""

  @abc.abstractmethod
  def name(self) -> str:
    """Returns the name of the model type."""
    raise NotImplementedError

  @abc.abstractmethod
  def __getstate__(self):
    raise NotImplementedError

  @abc.abstractmethod
  def __setstate__(self, state):
    raise NotImplementedError

  @abc.abstractmethod
  def task(self) -> Task:
    """Task solved by the model."""
    raise NotImplementedError

  @abc.abstractmethod
  def metadata(self) -> model_metadata.ModelMetadata:
    """Metadata associated with the model.

    A model's metadata contains information stored with the model that does not
    influence the model's predictions (e.g. data created). When distributing a
    model for wide release, it may be useful to clear / modify the model
    metadata with `model.set_metadata(ydf.ModelMetadata())`.

    Returns:
      The model's metadata.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def set_metadata(self, metadata: model_metadata.ModelMetadata):
    """Sets the model metadata."""
    raise NotImplementedError

  @abc.abstractmethod
  def set_feature_selection_logs(
      self, value: Optional[feature_selector_logs.FeatureSelectorLogs]
  ) -> None:
    """Records the feature selection logs."""
    raise NotImplementedError

  @abc.abstractmethod
  def feature_selection_logs(
      self,
  ) -> Optional[feature_selector_logs.FeatureSelectorLogs]:
    """Gets the feature selection logs."""

  @abc.abstractmethod
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
    raise NotImplementedError

  @abc.abstractmethod
  def data_spec(self) -> data_spec_pb2.DataSpecification:
    """Returns the data spec used for train the model."""
    raise NotImplementedError

  def set_data_spec(self, data_spec: data_spec_pb2.DataSpecification) -> None:
    """Changes the dataspec of the model.

    This operation is targeted to advanced users.

    Args:
      data_spec: New dataspec.
    """
    raise NotImplementedError(
        "This model does not support for the dataspec to be updated"
    )

  @abc.abstractmethod
  def benchmark(
      self,
      ds: dataset.InputDataset,
      benchmark_duration: float = 3,
      warmup_duration: float = 1,
      batch_size: int = 100,
      num_threads: Optional[int] = None,
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
      num_threads: Number of threads used for the multi-threaded benchmark. If
        not specified, the number of threads is set to the number of cpu cores.

    Returns:
      Benchmark results.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def save(
      self,
      path: str,
      advanced_options: ModelIOOptions = ModelIOOptions(),
      *,
      pure_serving: bool = False,
  ) -> None:
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
      pure_serving: If true, save the model without debug information to save
        disk space. Note that this option might require additional memory during
        saving, even though the resulting model can be significantly smaller on
        disk.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def serialize(self) -> bytes:
    """Serializes a model to a sequence of bytes (i.e. `bytes`).

    A serialized model is equivalent to model saved with `model.save`. It can
    possibly contain meta-data related to model training and interpretation. To
    minimize the size of a serialized model, removes this meta-data by passing
    the argument `pure_serving_model=True` to the `train` method.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Create a model
    dataset = pd.DataFrame({"feature": [0, 1], "label": [0, 1]})
    learner = ydf.RandomForestLearner(label="label")
    model = learner.train(dataset)

    # Serialize model
    # Note: serialized_model is a bytes.
    serialized_model = model.serialize()

    # Deserialize model
    deserialized_model = ydf.deserialize_model(serialized_model)

    # Make predictions
    model.predict(dataset)
    deserialized_model.predict(dataset)
    ```

    Returns:
      The serialized model.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def predict(
      self,
      data: dataset.InputDataset,
      *,
      use_slow_engine: bool = False,
      num_threads: Optional[int] = None,
  ) -> np.ndarray:
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

    The predictions are a NumPy array of float32 values. The structure of this
    array depends on the model's task and, in some cases, the number of classes.

    **Classification (`model.task() == ydf.Task.CLASSIFICATION`)**

    * *Binary Classification:* For models with two classes
    (`len(model.label_classes()) == 2`), the output is an array of shape
    `[num_examples]`. Each value represents the predicted probability of the
    positive class ( `model.label_classes()[1]` ).  To get the probability of
    the negative class, use `1 - model.predict(dataset)`.

    Here is an example of how to get the most probably class:

    ```python
    prediction_proba = model.predict(test_ds)
    predicted_classes = np.take(model.label_classes(), prediction_proba
    >= 0.5)

    # Or simply
    predicted_classes = model.predict_class(test_ds)
    ```

    * *Multi-class Classification:* For models with more than two classes, the
    output is an array of shape `[num_examples, num_classes]`. The value at
    index `[i, j]`  is the probability of class `j` for example `i`.

    Here is an example of how to get the most probably class:

    ```python
    prediction_proba = model.predict(test_ds)
    prediction_class_idx = np.argmax(prediction_proba, axis=1)
    predicted_classes = np.take(model.label_classes(),
    prediction_class_idx)

    # Or simply
    predicted_classes = model.predict_class(test_ds)
    ```

    **Regression (`model.task() == ydf.Task.REGRESSION`)**

    The output is an array of shape `[num_examples]`, where each value is
    the predicted value for the corresponding example.

    **Ranking (`model.task() == ydf.Task.RANKING`)**

    The output is an array of shape `[num_examples]`, where each value
    represents the score of the corresponding example. Higher scores
    indicate higher ranking.

    **Categorical Uplift (`model.task() == ydf.Task.CATEGORICAL_UPLIFT`)**

    The output is an array of shape `[num_examples]`, where each value
    represents the predicted uplift.  Positive values indicate a positive
    effect of the treatment on the outcome, while values close to zero
    indicate little to no effect.

    **Numerical Uplift (`model.task() == ydf.Task.NUMERICAL_UPLIFT`)**

    The output is an array of shape `[num_examples]`, with the
    interpretation being the same as for Categorical Uplift.

    **Anomaly Detection (`model.task() == ydf.Task.ANOMALY_DETECTION`)**

    The output is an array of shape `[num_examples]`, where each value is
    the anomaly score for the corresponding example. Scores range from 0
    (most normal) to 1 (most anomalous).

    Args:
      data: Dataset. Supported formats: VerticalDataset, (typed) path, list of
        (typed) paths, Pandas DataFrame, Xarray Dataset, TensorFlow Dataset,
        PyGrain DataLoader and Dataset (experimental, Linux only), dictionary of
        string to NumPy array or lists. If the dataset contains the label
        column, that column is ignored.
      use_slow_engine: If true, uses the slow engine for making predictions. The
        slow engine of YDF is an order of magnitude slower than the other
        prediction engines. There exist very rare edge cases where predictions
        with the regular engines fail, e.g., models with a very large number of
        categorical conditions. It is only in these cases, that users should use
        the slow engine and report the issue to the YDF developers.
      num_threads: Number of threads used to run the model.

    Returns:
      The predictions of the model on the given dataset.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def evaluate(
      self,
      data: dataset.InputDataset,
      *,
      weighted: Optional[bool] = None,
      task: Optional[Task] = None,
      label: Optional[str] = None,
      group: Optional[str] = None,
      bootstrapping: Union[bool, int] = False,
      ndcg_truncation: int = 5,
      mrr_truncation: int = 5,
      evaluation_task: Optional[Task] = None,
      use_slow_engine: bool = False,
      num_threads: Optional[int] = None,
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
    evaluation = model.evaluate(test_ds)
    ```

    In a notebook, if a cell returns an evaluation object, this evaluation will
    be as a rich html with plots:

    ```
    evaluation = model.evaluate(test_ds)
    # If model is an anomaly detection model:
    # evaluation = model.evaluate(test_ds,
                                  evaluation_task=ydf.Task.CLASSIFICATION)
    evaluation
    ```

    It is possible to evaluate the model differently than it was trained. For
    example, you can change the label, task and group.

    ```python
    ...
    # Train a regression model
    model = ydf.RandomForestLearner(label="label",
    task=ydf.Task.REGRESSION).train(train_ds)

    # Evaluate the model as a regressive model
    regressive_evaluation = model.evaluate(test_ds)

    # Evaluate the model as a ranking model model
    regressive_evaluation = model.evaluate(test_ds,
      task=ydf.Task.RANKING, group="group_column")
    ```

    Args:
      data: Dataset. Supported formats: VerticalDataset, (typed) path, list of
        (typed) paths, Pandas DataFrame, Xarray Dataset, TensorFlow Dataset,
        PyGrain DataLoader and Dataset (experimental, Linux only), dictionary of
        string to NumPy array or lists.
      weighted: If true, the evaluation is weighted according to the training
        weights. If false, the evaluation is non-weighted. b/351279797: Change
        default to weights=True.
      task: Override the task of the model during the evaluation. If None
        (default), the model is evaluated according to its training task.
      label: Override the label used to evaluate the model. If None (default),
        use the model's label.
      group: Override the group used to evaluate the model. If None (default),
        use the model's group. Only used for ranking models.
      bootstrapping: Controls whether bootstrapping is used to evaluate the
        confidence intervals and statistical tests (i.e., all the metrics ending
        with "[B]"). If set to false, bootstrapping is disabled. If set to true,
        bootstrapping is enabled and 2000 bootstrapping samples are used. If set
        to an integer, it specifies the number of bootstrapping samples to use.
        In this case, if the number is less than 100, an error is raised as
        bootstrapping will not yield useful results.
      ndcg_truncation: Controls at which ranking position the NDCG metric should
        be truncated. Default to 5. Ignored for non-ranking models.
      mrr_truncation: Controls at which ranking position the MRR metric loss
        should be truncated. Default to 5. Ignored for non-ranking models.
      evaluation_task: Deprecated. Use `task` instead.
      use_slow_engine: If true, uses the slow engine for making predictions. The
        slow engine of YDF is an order of magnitude slower than the other
        prediction engines. There exist very rare edge cases where predictions
        with the regular engines fail, e.g., models with a very large number of
        categorical conditions. It is only in these cases, that users should use
        the slow engine and report the issue to the YDF developers.
      num_threads: Number of threads used to run the model.

    Returns:
      Model evaluation.
    """
    raise NotImplementedError

  @abc.abstractmethod
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
      single_example: Example to explain. Supported formats: VerticalDataset,
        (typed) path, list of (typed) paths, Pandas DataFrame, Xarray Dataset,
        TensorFlow Dataset, PyGrain DataLoader and Dataset (experimental, Linux
        only), dictionary of string to NumPy array or lists.

    Returns:
      Prediction explanation.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def analyze(
      self,
      data: dataset.InputDataset,
      sampling: float = 1.0,
      num_bins: int = 50,
      partial_dependence_plot: bool = True,
      conditional_expectation_plot: bool = True,
      permutation_variable_importance_rounds: int = 1,
      num_threads: Optional[int] = None,
      maximum_duration: Optional[float] = 20,
  ) -> analysis.Analysis:
    """Analyzes a model on a test dataset.

    An analysis contains structural information about the model (e.g., variable
    importances), and the information about the application of the model on the
    given dataset (e.g. partial dependence plots).

    For a large dataset (many examples and / or features), computing the
    analysis can take significant time.

    While some information might be valid, it is generally not recommended to
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
      data: Dataset. Supported formats: VerticalDataset, (typed) path, list of
        (typed) paths, Pandas DataFrame, Xarray Dataset, TensorFlow Dataset,
        PyGrain DataLoader and Dataset (experimental, Linux only), dictionary of
        string to NumPy array or lists.
      sampling: Ratio of examples to use for the analysis. The analysis can be
        expensive to compute. On large datasets, use a small sampling value e.g.
        0.01.
      num_bins: Number of bins used to accumulate statistics. A large value
        increase the resolution of the plots but takes more time to compute.
      partial_dependence_plot: Compute partial dependency plots a.k.a PDPs.
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
      maximum_duration: Maximum duration of the analysis in seconds. Note that
        the analysis can last a little longer than this value.

    Returns:
      Model analysis.
    """
    raise NotImplementedError

  @abc.abstractmethod
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
    raise NotImplementedError

  @abc.abstractmethod
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
      force: bool = False,
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
    those two conventions with `feed_example_proto=True` and `servo_api=True`
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
        not provided, the parsing feature specs are automatically generated
        based on the model features seen during training. This means that
        "feature_specs" is only necessary when using a "pre_processing" argument
        that expects different features than what the model was trained with.
        This argument is ignored when exporting model with
        `feed_example_proto=False`. Only compatible with mode="tf".
      force: Try to export even in currently unsupported environments. WARNING:
        Setting this to true may crash the Python runtime.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def to_tensorflow_function(  # pytype: disable=name-error
      self,
      temp_dir: Optional[str] = None,
      can_be_saved: bool = True,
      squeeze_binary_classification: bool = True,
      force: bool = False,
  ) -> "tensorflow.Module":  # pylint: disable=undefined-variable
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
      force: Try to export even in currently unsupported environments.

    Returns:
      A TensorFlow @tf.function.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def to_jax_function(  # pytype: disable=name-error
      self,
      jit: bool = True,
      apply_activation: bool = True,
      leaves_as_params: bool = False,
      compatibility: Union[str, "export_jax.Compatibility"] = "XLA",  # pylint: disable=undefined-variable
  ) -> "export_jax.JaxModel":  # pylint: disable=undefined-variable
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
      compatibility: Constraint on the YDF->JAX conversion to runtime
        compatibility. Can be "XLA" (default), and "TFL" (for TensorFlow Lite).

    Returns:
      A dataclass containing the JAX prediction function (`predict`) and
      optionally the model parameters (`params`) and feature encoder
      (`encoder`).
    """
    raise NotImplementedError

  @abc.abstractmethod
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
    raise NotImplementedError

  @abc.abstractmethod
  def hyperparameter_optimizer_logs(
      self,
  ) -> Optional[optimizer_logs.OptimizerLogs]:
    """Returns the logs of the hyper-parameter tuning.

    If the model is not trained with hyper-parameter tuning, returns None.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def variable_importances(self) -> Dict[str, List[Tuple[float, str]]]:
    """Variable importances to measure the impact of features on the model.

    Variable importances generally indicates how much a variable (feature)
    contributes to the model predictions or quality. Different Variable
    importances have different semantics and are generally not comparable.

    The variable importances returned by `variable_importances()` depends on the
    learning algorithm and its hyper-parameters. For example, the hyperparameter
    `compute_oob_variable_importances=True` of the Random Forest learner enables
    the computation of permutation out-of-bag variable importances.

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
    raise NotImplementedError

  @abc.abstractmethod
  def label_col_idx(self) -> int:
    raise NotImplementedError

  @abc.abstractmethod
  def input_features_col_idxs(self) -> Sequence[int]:
    raise NotImplementedError

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

  @abc.abstractmethod
  def list_compatible_engines(self) -> Sequence[str]:
    """Lists the inference engines compatible with the model.

    The engines are sorted to likely-fastest to  likely-slowest.

    Returns:
      List of compatible engines.
    """
    raise NotImplementedError

  @abc.abstractmethod
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
    raise NotImplementedError

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
        for column_idx in self.input_features_col_idxs()
    ]

  def input_feature_names(self) -> List[str]:
    """Returns the names of the input features.

    The features are sorted in increasing order of column_idx.
    """

    return [f.name for f in self.input_features()]

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
    label_column = self.data_spec().columns[self.label_col_idx()]
    if label_column.type != data_spec_pb2.CATEGORICAL:
      semantic = dataspec.Semantic.from_proto_type(label_column.type)
      raise ValueError(
          "CATEGORICAL column expected for classification label. Got"
          f" {semantic} instead. Should the model be a regresion? If so, set"
          " `task=ydf.REGRESSION` in the learner constructor argument."
      )

    if label_column.categorical.is_already_integerized:
      log.info(
          "The label column is integerized. This is expected for models trained"
          " with TensorFlow Decision Forests."
      )

    # The first element is the "out-of-vocabulary" that is not used in labels.
    return dataspec.categorical_column_dictionary_to_list(label_column)[1:]

  def predict_class(
      self,
      data: dataset.InputDataset,
      *,
      use_slow_engine: bool = False,
      num_threads: Optional[int] = None,
  ) -> np.ndarray:
    """Returns the most likely predicted class for a classification model.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").train(train_ds)

    test_ds = pd.read_csv("test.csv")
    predictions = model.predict_class(test_ds)
    ```

    This method returns a numpy array of string of shape `[num_examples]`. Each
    value represents the most likely class for the corresponding example. This
    method can only be used for classification models.

    In case of ties, the first class in`model.label_classes()` is returned.

    See `model.predict` to generate the full prediction probabilities.

    Args:
      data: Dataset. Supported formats: VerticalDataset, (typed) path, list of
        (typed) paths, Pandas DataFrame, Xarray Dataset, TensorFlow Dataset,
        PyGrain DataLoader and Dataset (experimental, Linux only), dictionary of
        string to NumPy array or lists. If the dataset contains the label
        column, that column is ignored.
      use_slow_engine: If true, uses the slow engine for making predictions. The
        slow engine of YDF is an order of magnitude slower than the other
        prediction engines. There exist very rare edge cases where predictions
        with the regular engines fail, e.g., models with a very large number of
        categorical conditions. It is only in these cases that users should use
        the slow engine and report the issue to the YDF developers.
      num_threads: Number of threads used to run the model.

    Returns:
      The most likely predicted class for each example.
    """

    if self.task() != Task.CLASSIFICATION:
      raise ValueError(
          "predict_class is only supported for classification models."
      )

    label_classes = self.label_classes()
    prediction_proba = self.predict(
        data, use_slow_engine=use_slow_engine, num_threads=num_threads
    )

    if len(label_classes) == 2:
      return np.take(label_classes, prediction_proba > 0.5)
    else:
      prediction_class_idx = np.argmax(prediction_proba, axis=1)
      return np.take(label_classes, prediction_class_idx)


class GenericCCModel(GenericModel):
  """Abstract superclass for the YDF models implemented in C++."""

  def __init__(self, raw_model: ydf.GenericCCModel):
    self._model = raw_model

  def name(self) -> str:
    return self._model.name()

  def __getstate__(self):
    log.warning(
        "Model pickling is discouraged. To save a model on disk, use"
        " `model.save(path)` and `... = ydf.load_model(path)` instead. To"
        " serialize a model to bytes, use `data = model.serialize()` and"
        " `... = ydf.deserialize_model(data)` instead.",
        message_id=log.WarningMessage.DONT_USE_PICKLE,
    )
    return self._model.Serialize()

  def __setstate__(self, state):
    self._model = ydf.DeserializeModel(state)

  def task(self) -> Task:
    return Task._from_proto_type(self._model.task())  # pylint: disable=protected-access

  def metadata(self) -> model_metadata.ModelMetadata:
    return model_metadata.ModelMetadata._from_proto_type(self._model.metadata())  # pylint:disable=protected-access

  def set_metadata(self, metadata: model_metadata.ModelMetadata):
    self._model.set_metadata(metadata._to_proto_type())  # pylint:disable=protected-access

  def set_feature_selection_logs(
      self, value: Optional[feature_selector_logs.FeatureSelectorLogs]
  ) -> None:
    if value is None:
      self._model.set_feature_selection_logs(None)
    else:
      self._model.set_feature_selection_logs(
          feature_selector_logs.value_to_proto(value)
      )

  def feature_selection_logs(
      self,
  ) -> Optional[feature_selector_logs.FeatureSelectorLogs]:
    proto = self._model.feature_selection_logs()
    if proto is None:
      return None
    else:
      return feature_selector_logs.proto_to_value(proto)

  def describe(
      self,
      output_format: Literal["auto", "text", "notebook", "html"] = "auto",
      full_details: bool = False,
  ) -> Union[str, html.HtmlNotebookDisplay]:
    if output_format == "auto":
      output_format = "text" if log.is_direct_output() else "notebook"

    with log.cc_log_context():
      description = self._model.Describe(full_details, output_format == "text")
      if output_format == "notebook":
        return html.HtmlNotebookDisplay(description)
      else:
        return description

  def data_spec(self) -> data_spec_pb2.DataSpecification:
    return self._model.data_spec()

  def set_data_spec(self, data_spec: data_spec_pb2.DataSpecification) -> None:
    self._model.set_data_spec(data_spec)

  def benchmark(
      self,
      ds: dataset.InputDataset,
      benchmark_duration: float = 3,
      warmup_duration: float = 1,
      batch_size: int = 100,
      num_threads: Optional[int] = None,
  ) -> ydf.BenchmarkInferenceCCResult:
    if num_threads is None:
      num_threads = concurrency.determine_optimal_num_threads(training=False)

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
          vds._dataset,  # pylint: disable=protected-access
          benchmark_duration,
          warmup_duration,
          batch_size,
          num_threads,
      )
    return result

  def save(
      self, path: str, advanced_options=ModelIOOptions(), *, pure_serving=False
  ) -> None:
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
      self._model.Save(path, advanced_options.file_prefix, pure_serving)

  def serialize(self) -> bytes:
    with log.cc_log_context():
      return self._model.Serialize()

  def predict(
      self,
      data: dataset.InputDataset,
      *,
      use_slow_engine: bool = False,
      num_threads: Optional[int] = None,
  ) -> np.ndarray:
    if num_threads is None:
      num_threads = concurrency.determine_optimal_num_threads(training=False)

    with log.cc_log_context():
      # The data spec contains the label / weights /  ranking group / uplift
      # treatment column, but those are not required for making predictions.
      ds = dataset.create_vertical_dataset(
          data,
          data_spec=self._model.data_spec(),
          required_columns=self.input_feature_names(),
      )
      result = self._model.Predict(
          ds._dataset, use_slow_engine, num_threads=num_threads  # pylint: disable=protected-access
      )
    return result

  def evaluate(
      self,
      data: dataset.InputDataset,
      *,
      weighted: Optional[bool] = None,
      task: Optional[Task] = None,
      label: Optional[str] = None,
      group: Optional[str] = None,
      bootstrapping: Union[bool, int] = False,
      ndcg_truncation: int = 5,
      mrr_truncation: int = 5,
      evaluation_task: Optional[Task] = None,
      use_slow_engine: bool = False,
      num_threads: Optional[int] = None,
  ) -> metric.Evaluation:
    if num_threads is None:
      num_threads = concurrency.determine_optimal_num_threads(training=False)

    # Warning about deprecation of "evaluation_task"
    if evaluation_task is not None:
      log.warning(
          "The `evaluation_task` argument is deprecated. Use `task` instead.",
          message_id=log.WarningMessage.DEPRECATED_EVALUATION_TASK,
      )
      if task is not None:
        raise ValueError("Cannot specify both `task` and `evaluation_task`")
      task = evaluation_task

    # Warning about change default value of "weighted")
    if weighted is None:
      weighted = False
      if self._model.weighted_training():
        # TODO: Change default to true and remove warning.
        log.warning(
            "Non-weighted evaluation of a model trained with weighted training."
            " Are you sure you don't want to do a weighted evaluation? Set"
            " `model.evaluate(weighted=True, ...)` or"
            " `model.evaluate(weighted=False, ...)` accordingly.",
            message_id=log.WarningMessage.WEIGHTED_NOT_SET_IN_EVAL,
        )

    # Warning about unnecessary arguments
    if task is not None and task == self.task():
      log.warning(
          "No need to set the `task` argument in `model.evaluate` if the model"
          " is evaluated the same way it was trained.",
          message_id=log.WarningMessage.UNNECESSARY_TASK_ARGUMENT,
      )
    if label is not None and label == self.label():
      log.warning(
          "No need to set the `task` argument in `model.evaluate` if the model"
          " is evaluated the same way it was trained.",
          message_id=log.WarningMessage.UNNECESSARY_LABEL_ARGUMENT,
      )

    if isinstance(bootstrapping, bool):
      bootstrapping_samples = 2000 if bootstrapping else -1
    elif isinstance(bootstrapping, int) and bootstrapping >= 100:
      bootstrapping_samples = bootstrapping
    else:
      raise ValueError(
          "bootstrapping argument should be boolean or an integer greater than"
          " 100 as bootstrapping will not yield useful results otherwise. Got"
          f" {bootstrapping!r}"
      )
    if task is None:
      task = self.task()

    with log.cc_log_context():

      effective_dataspec, label_col_idx, group_col_idx, required_columns = (
          self._build_evaluation_dataspec(
              override_task=task._to_proto_type(),  # pylint: disable=protected-access
              override_label=label,
              override_group=group,
              weighted=weighted,
          )
      )

      ds = dataset.create_vertical_dataset(
          data, data_spec=effective_dataspec, required_columns=required_columns
      )

      options_proto = metric_pb2.EvaluationOptions(
          bootstrapping_samples=bootstrapping_samples,
          task=task._to_proto_type(),  # pylint: disable=protected-access
          ranking=metric_pb2.EvaluationOptions.Ranking(
              ndcg_truncation=ndcg_truncation, mrr_truncation=mrr_truncation
          )
          if task == Task.RANKING
          else None,
          num_threads=num_threads,
      )

      evaluation_proto = self._model.Evaluate(
          ds._dataset,  # pylint: disable=protected-access
          options_proto,
          weighted=weighted,
          label_col_idx=label_col_idx,
          group_col_idx=group_col_idx,
          use_slow_engine=use_slow_engine,
          num_threads=num_threads,
      )
    return metric.Evaluation(evaluation_proto)

  def analyze_prediction(
      self,
      single_example: dataset.InputDataset,
  ) -> analysis.PredictionAnalysis:
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
      partial_dependence_plot: bool = True,
      conditional_expectation_plot: bool = True,
      permutation_variable_importance_rounds: int = 1,
      num_threads: Optional[int] = None,
      maximum_duration: Optional[float] = 20,
  ) -> analysis.Analysis:
    if num_threads is None:
      num_threads = concurrency.determine_optimal_num_threads(training=False)

    enable_permutation_variable_importances = (
        permutation_variable_importance_rounds > 0
    )
    if (
        enable_permutation_variable_importances
        and self.task() == Task.ANOMALY_DETECTION
        and self._model.label_col_idx() == -1
    ):
      # TODO: Allow AD evaluation and analysis without providing label at training time.
      enable_permutation_variable_importances = False
      log.warning(
          "ANOMALY DETECTION models must be trained with a label for variable"
          " importance computation",
          message_id=log.WarningMessage.AD_PERMUTATION_VARIABLE_IMPORTANCE_NOT_ENABLED,
      )

    with log.cc_log_context():
      ds = dataset.create_vertical_dataset(
          data, data_spec=self._model.data_spec()
      )

      options_proto = model_analysis_pb2.Options(
          num_threads=num_threads,
          maximum_duration_seconds=maximum_duration,
          pdp=model_analysis_pb2.Options.PlotConfig(
              enabled=partial_dependence_plot,
              example_sampling=sampling,
              num_numerical_bins=num_bins,
          ),
          cep=model_analysis_pb2.Options.PlotConfig(
              enabled=conditional_expectation_plot,
              example_sampling=sampling,
              num_numerical_bins=num_bins,
          ),
          permuted_variable_importance=model_analysis_pb2.Options.PermutedVariableImportance(
              enabled=enable_permutation_variable_importances,
              num_rounds=permutation_variable_importance_rounds,
          ),
          include_model_structural_variable_importances=True,
      )

      analysis_proto = self._model.Analyze(ds._dataset, options_proto)  # pylint: disable=protected-access
      return analysis.Analysis(analysis_proto, options_proto)

  def to_cpp(self, key: str = "my_model") -> str:
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
      force: bool = False,
  ) -> None:
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
      force: bool = False,
  ) -> "tensorflow.Module":  # pylint: disable=undefined-variable
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
      compatibility: Union[str, "export_jax.Compatibility"] = "XLA",  # pylint: disable=undefined-variable
  ) -> "export_jax.JaxModel":  # pylint: disable=undefined-variable
    return _get_export_jax().to_jax_function(
        model=self,
        jit=jit,
        apply_activation=apply_activation,
        leaves_as_params=leaves_as_params,
        compatibility=compatibility,
    )

  def update_with_jax_params(self, params: Dict[str, Any]):
    _get_export_jax().update_with_jax_params(model=self, params=params)

  def to_docker(
      self,
      path: str,
      exist_ok: bool = False,
  ) -> None:
    """Exports the model to a Docker endpoint deployable on Cloud.

    This function creates a directory containing a Dockerfile, the model and
    support files.

    Usage example:

    ```python
    import ydf

    # Train a model.
    model = ydf.RandomForestLearner(label="l").train({
        "f1": np.random.random(size=100),
        "f2": np.random.random(size=100),
        "l": np.random.randint(2, size=100),
    })

    # Export the model to a Docker endpoint.
    model.to_docker(path="/tmp/my_model")

    # Print instructions on how to use the model
    !cat /tmp/my_model/readme.md

    # Test the end-point locally
    docker build --platform linux/amd64 -t ydf_predict_image /tmp/my_model
    docker run --rm -p 8080:8080 -d ydf_predict_image

    # Deploy the model on Google Cloud
    gcloud run deploy ydf-predict --source /tmp/my_model

    # Check the automatically created utility scripts "test_locally.sh" and
    # "deploy_in_google_cloud.sh" for more examples.
    ```

    Args:
      path: Directory where to create the Docker endpoint
      exist_ok: If false (default), fails if the directory already exist. If
        true, override the directory content if any.
    """
    _get_export_docker().to_docker(model=self, path=path, exist_ok=exist_ok)

  def hyperparameter_optimizer_logs(
      self,
  ) -> Optional[optimizer_logs.OptimizerLogs]:
    proto_logs = self._model.hyperparameter_optimizer_logs()
    if proto_logs is None:
      return None
    return optimizer_logs.proto_optimizer_logs_to_optimizer_logs(proto_logs)

  def variable_importances(self) -> Dict[str, List[Tuple[float, str]]]:
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

  def input_features_col_idxs(self) -> Sequence[int]:
    return self._model.input_features()

  def list_compatible_engines(self) -> Sequence[str]:
    return self._model.ListCompatibleEngines()

  def force_engine(self, engine_name: Optional[str]) -> None:
    self._model.ForceEngine(engine_name)

  def _build_evaluation_dataspec(
      self,
      override_task: abstract_model_pb2.Task,
      override_label: Optional[str],
      override_group: Optional[str],
      weighted: bool,
  ) -> Tuple[data_spec_pb2.DataSpecification, int, int, List[str]]:
    """Creates a dataspec for evaluation.

    Args:
      override_task: Override task to use for the evaluation.
      override_label: Override name of the label column.
      override_group: Override name of the group column.
      weighted: Whether or not the evaluation is weighted.

    Returns:
      The dataspec, the label column index, the group column index and the
      required columns.
    """

    # Default dataspec of the model
    effective_dataspec = self._model.data_spec()

    def find_existing_or_add_column(
        semantic: Optional[Any],
        name: Optional[str],
        default_col_idx: int,
        usage: str,
    ) -> int:
      """Create a new or retreive an existing column."""
      if name is None:
        if semantic is None:
          return -1
        else:
          existing_col_def = effective_dataspec.columns[default_col_idx]
          if existing_col_def.type == semantic:
            return default_col_idx  # Use the model's default
          else:
            col_idx = default_col_idx
      else:
        if semantic is None:
          raise ValueError(
              f"A {abstract_model_pb2.Task.Name(override_task)} evaluation does"
              f" not expected a {usage} column."
          )
        col_idx = column_names.get(name, None)
        if col_idx is not None:
          # A column with the same name already exists
          existing_col_def = effective_dataspec.columns[col_idx]
          if existing_col_def.type != semantic:
            log.warning(
                f"Add dual semantic to {usage} column {name!r}. Original"
                " semantic:"
                f"{data_spec_pb2.ColumnType.Name(existing_col_def.type)} New"
                f" semantic:{data_spec_pb2.ColumnType.Name(semantic)}"
            )
          else:
            return col_idx

      # Create a new column
      if col_idx is None:
        col_idx = len(effective_dataspec.columns)
        new_col = effective_dataspec.columns.add(name=name, type=semantic)
      else:
        new_col = effective_dataspec.columns[col_idx]
        new_col.type = semantic

      # Populate column content
      if (
          override_task == abstract_model_pb2.Task.CLASSIFICATION
          and self._model.task() == abstract_model_pb2.Task.CLASSIFICATION
      ):
        # Copy the dictionnary of the categorical label
        new_col.categorical.CopyFrom(
            effective_dataspec.columns[default_col_idx].categorical
        )
      elif semantic == data_spec_pb2.ColumnType.CATEGORICAL:
        # Create a binary looking category
        new_col.categorical.most_frequent_value = 1
        new_col.categorical.number_of_unique_values = 3
        new_col.categorical.is_already_integerized = False
        new_col.categorical.items[dataspec.YDF_OOD].index = 0
        new_col.categorical.items["0"].index = 1
        new_col.categorical.items["1"].index = 2

      return col_idx

    # Compute the expected semantic of the evaluation columns
    label_col_semantic = (
        data_spec_pb2.ColumnType.CATEGORICAL
        if override_task
        in [
            abstract_model_pb2.Task.CLASSIFICATION,
            abstract_model_pb2.Task.CATEGORICAL_UPLIFT,
        ]
        else data_spec_pb2.ColumnType.NUMERICAL
    )
    group_col_semantic = (
        data_spec_pb2.ColumnType.HASH
        if override_task == abstract_model_pb2.Task.RANKING
        else None
    )

    # Index the column names
    column_names = {
        col.name: idx for idx, col in enumerate(effective_dataspec.columns)
    }

    effective_label_col_idx = find_existing_or_add_column(
        label_col_semantic, override_label, self._model.label_col_idx(), "label"
    )
    effective_group_col_idx = find_existing_or_add_column(
        group_col_semantic,
        override_group,
        self._model.group_col_idx(),
        "group",
    )

    required_columns = [c.name for c in effective_dataspec.columns]
    if weighted:
      if not self._model.weighted_training():
        raise ValueError(
            "Weighted evaluation is only supported for models trained with"
            " weights."
        )
    else:
      if self._model.weighted_training():
        weight_col_to_remove = self._model.weight_col_idx()
        if weight_col_to_remove is None:
          raise ValueError("No weight column found in the model.")
        # Pick the weight column from the original data spec.
        if weight_col_to_remove >= len(self.data_spec().columns):
          raise ValueError(
              f"Weight column has index {weight_col_to_remove}, but the data"
              f" spec only has {len(self.data_spec().columns)} columns."
          )
        required_columns.remove(
            self.data_spec().columns[weight_col_to_remove].name
        )
    return (
        effective_dataspec,
        effective_label_col_idx,
        effective_group_col_idx,
        required_columns,
    )


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


def _get_export_docker():
  try:
    from ydf.model import export_docker  # pylint: disable=g-import-not-at-top,import-outside-toplevel # pytype: disable=import-error

    return export_docker
  except ImportError as exc:
    raise ValueError("Cannot import the export_docker utility") from exc


ModelType = TypeVar("ModelType", bound=GenericModel)
