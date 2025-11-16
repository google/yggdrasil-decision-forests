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
from yggdrasil_decision_forests.serving.embed import embed_pb2
from yggdrasil_decision_forests.utils import model_analysis_pb2


@enum.unique
class Task(enum.Enum):
  """A task that a model is trained to solve.

  Not all tasks are compatible with all learners or hyperparameters. For more
  information, see the tutorials on individual tasks in the documentation.

  Usage example:

  ```python
  import ydf

  learner = ydf.RandomForestLearner(
      label="income", task=ydf.Task.CLASSIFICATION
  )
  # model = learner.train(...)
  # assert model.task() == ydf.Task.CLASSIFICATION
  ```

  Attributes:
    CLASSIFICATION: Predicts a categorical label.
    REGRESSION: Predicts a numerical label.
    RANKING: Ranks a set of items. The label represents the relevance of an
      item. For example, with the default NDCG metric, the label is a numerical
      value where 0 indicates a completely unrelated item and 4 indicates a
      perfect match.
    CATEGORICAL_UPLIFT: Predicts the incremental impact of a treatment on a
      categorical outcome.
    NUMERICAL_UPLIFT: Predicts the incremental impact of a treatment on a
      numerical outcome.
    ANOMALY_DETECTION: Detects if an instance is an outlier compared to the
      training data. The prediction is a score between 0 and 1, where 0
      represents a normal instance and 1 represents the most anomalous instance.
    SURVIVAL_ANALYSIS: Predicts the survival probability of an individual over
      time.
  """

  CLASSIFICATION = "CLASSIFICATION"
  REGRESSION = "REGRESSION"
  RANKING = "RANKING"
  CATEGORICAL_UPLIFT = "CATEGORICAL_UPLIFT"
  NUMERICAL_UPLIFT = "NUMERICAL_UPLIFT"
  ANOMALY_DETECTION = "ANOMALY_DETECTION"
  SURVIVAL_ANALYSIS = "SURVIVAL_ANALYSIS"

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
    Task.SURVIVAL_ANALYSIS: abstract_model_pb2.SURVIVAL_ANALYSIS,
}
PROTO_TO_TASK = {v: k for k, v in TASK_TO_PROTO.items()}


@dataclasses.dataclass(frozen=True)
class ModelIOOptions:
  """Advanced options for saving and loading YDF models.

  Attributes:
    file_prefix: Optional prefix for model files. Allows multiple models to be
      stored in the same directory, although this is discouraged. If not
      specified during loading, the prefix is auto-detected. If not specified
      during saving, no prefix is used.
  """

  file_prefix: Optional[str] = None


@enum.unique
class NodeFormat(enum.Enum):
  # pyformat: disable
  """Specifies the storage format for the internal nodes of a tree-based model.

  Attributes:
    BLOB_SEQUENCE: Default format for the public version of YDF.
    BLOB_SEQUENCE_GZIP: Efficient compressed version of the BLOB_SEQUENCE
      format. Might not be compatible with pre-2025 builds of YDF and TF-DF.
  """
  # pyformat: enable

  BLOB_SEQUENCE = enum.auto()
  BLOB_SEQUENCE_GZIP = enum.auto()


@dataclasses.dataclass
class InputFeature:
  """An input feature of a model.

  Attributes:
    name: The unique name of the feature.
    semantic: The semantic type of the feature (e.g., numerical, categorical).
    column_idx: The index of the feature's column in the model's data
      specification (`dataspec`).
  """

  name: str
  semantic: dataspec.Semantic
  column_idx: int


@dataclasses.dataclass(frozen=True)
class TrainingLogEntry:
  """A record of evaluation metrics at a specific point during model training.

  This structure is returned by `model.training_logs()`. It contains the
  evaluation metrics of the model at a specific point during training (e.g.,
  after a given number of trees have been trained).

  Attributes:
    iteration: The training iteration when the evaluation was recorded. For many
      models, this is the number of trees.
    evaluation: Evaluation metrics at the given training iteration. For Gradient
      Boosted Trees, this is the evaluation on the validation dataset. For
      Random Forests, this is the out-of-bag evaluation.
    training_evaluation: Optional evaluation metrics computed on the training
      dataset at the given iteration. This is generally less insightful than the
      main `evaluation` but can be useful for debugging.
  """

  iteration: int
  evaluation: metric.Evaluation
  training_evaluation: Optional[metric.Evaluation]


class GenericModel(abc.ABC):
  """Abstract superclass for all YDF models."""

  def __str__(self) -> str:
    return f"""\
Model: {self.name()}
Task: {self.task().name}
Class: ydf.{self.__class__.__name__}
Use `model.describe()` for more details.
"""

  def __repr__(self) -> str:
    return f"<ydf.{self.__class__.__name__}>"

  @abc.abstractmethod
  def name(self) -> str:
    """Returns the name of the model type (e.g., "RANDOM_FOREST")."""
    raise NotImplementedError

  @abc.abstractmethod
  def __getstate__(self):
    """Serializes the model for pickling."""
    raise NotImplementedError

  @abc.abstractmethod
  def __setstate__(self, state):
    """Deserializes the model for unpickling."""
    raise NotImplementedError

  @abc.abstractmethod
  def task(self) -> Task:
    """The task the model is trained to solve.

    Returns:
      The task enum for this model.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def metadata(self) -> model_metadata.ModelMetadata:
    """Metadata associated with the model.

    A model's metadata contains information that does not influence its
    predictions, such as the creation time. When distributing a model for wide
    release, it may be useful to clear or modify the metadata.

    Example:
    ```python
    # Clear the metadata
    model.set_metadata(ydf.ModelMetadata())
    ```

    Returns:
      The model's metadata object.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def set_metadata(self, metadata: model_metadata.ModelMetadata):
    """Updates the model's metadata.

    Args:
      metadata: The new metadata object for the model.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def set_feature_selection_logs(
      self, value: Optional[feature_selector_logs.FeatureSelectorLogs]
  ) -> None:
    """Sets the feature selection logs for the model.

    Args:
      value: The feature selection logs to set, or `None` to clear them.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def feature_selection_logs(
      self,
  ) -> Optional[feature_selector_logs.FeatureSelectorLogs]:
    """Retrieves the feature selection logs, if available.

    Returns:
      The feature selection logs, or `None` if they are not available.
    """

  @abc.abstractmethod
  def describe(
      self,
      output_format: Literal["auto", "text", "notebook", "html"] = "auto",
      full_details: bool = False,
  ) -> Union[str, html.HtmlNotebookDisplay]:
    """Generates a textual or HTML description of the model.

    Args:
      output_format: The format of the output. - "auto": "notebook" in an
        IPython notebook, "text" otherwise. - "text": A plain text description.
        - "html": A standalone HTML description. - "notebook": An HTML
        description for display in a notebook cell.
      full_details: If `True`, the full model structure is included, which can
        be very large.

    Returns:
      The model description as a string or an HTML display object.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def data_spec(self) -> data_spec_pb2.DataSpecification:
    """The data specification of the dataset used to train the model.

    Returns:
      A DataSpecification protobuf object.
    """
    raise NotImplementedError

  def set_data_spec(self, data_spec: data_spec_pb2.DataSpecification) -> None:
    """Updates the data specification of the model.

    This is an advanced feature and should be used with caution, as it can
    easily lead to a broken model.

    Args:
      data_spec: The new DataSpecification protobuf object.
    """
    raise NotImplementedError(
        "This model does not support updating the dataspec."
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
    """Benchmarks the inference speed of the model on a given dataset.

    This method measures the time it takes to run predictions on the dataset
    using the Yggdrasil Decision Forests C++ engine. Note that inference times
    may vary on different machines or with other APIs. A C++ serving template
    can be generated with `model.to_cpp()`.

    Args:
      ds: The dataset to use for benchmarking.
      benchmark_duration: The target duration of the benchmark in seconds. The
        actual duration may be slightly different. Must be > 0.
      warmup_duration: The target duration of the warmup phase in seconds.
        During this phase, predictions are run but not timed, to warm up caches.
        Must be > 0.
      batch_size: The number of examples to process in each batch. The impact of
        this parameter depends on the machine's architecture (e.g., cache
        sizes).
      num_threads: The number of threads to use for the benchmark. If not
        specified, it defaults to the number of available CPU cores.

    Returns:
      An object containing the benchmark results.

    Raises:
      ValueError: If `benchmark_duration`, `warmup_duration`, or `batch_size`
        are not positive.
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
    """Saves the model to a directory.

    YDF uses a proprietary format consisting of multiple files in a single
    directory. This directory should ideally contain only one model.

    YDF models can also be exported to other formats, such as TensorFlow
    SavedModel (`to_tensorflow_saved_model()`) or C++ code (`to_cpp()`).

    The model may contain metadata (see `model.metadata()`). Before distributing
    a model, consider clearing this metadata:
    `model.set_metadata(ydf.ModelMetadata())`.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train a Random Forest model
    df = pd.read_csv("my_dataset.csv")
    model = ydf.RandomForestLearner(label="my_label").train(df)

    # Save the model to disk
    model.save("/models/my_model")
    ```

    Args:
      path: The path to the directory where the model will be saved.
      advanced_options: Advanced options for saving the model.
      pure_serving: If `True`, saves a smaller version of the model suitable for
        serving by removing training-specific metadata and debug information.
        This might require more memory during the saving process, but the
        resulting model on disk will be smaller.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def serialize(self) -> bytes:
    """Serializes the model into a `bytes` object.

    A serialized model is equivalent to a model saved with `model.save()`. It
    may contain metadata related to training and interpretation. To minimize
    its size, you can train with the `pure_serving_model=True` option in the
    learner.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Create and train a model
    dataset = pd.DataFrame({"feature": [0, 1], "label": [0, 1]})
    learner = ydf.RandomForestLearner(label="label")
    model = learner.train(dataset)

    # Serialize the model to a bytes object
    serialized_model = model.serialize()

    # Deserialize the model
    deserialized_model = ydf.deserialize_model(serialized_model)

    # Make predictions with both models
    predictions = model.predict(dataset)
    deserialized_predictions = deserialized_model.predict(dataset)
    ```

    Returns:
      The serialized model as a `bytes` object.
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
    """Runs the model on a dataset and returns its predictions.

    The output is a NumPy array of `float32` values. The structure of this
    array depends on the model's task. See the "Returns" section for details.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train a model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").train(train_ds)

    # Get predictions on a test dataset
    test_ds = pd.read_csv("test.csv")
    predictions = model.predict(test_ds)
    ```

    Args:
      data: The dataset to make predictions on. Can be a pandas DataFrame, a
        dictionary of NumPy arrays, a path to a file, etc. If the dataset
        contains the label column, it will be ignored.
      use_slow_engine: If `True`, uses a slower, more robust inference engine.
        This is a fallback for rare edge cases where the default engines might
        fail (e.g., models with a very large number of categorical conditions).
        If you encounter such a case, please report it to the YDF developers.
      num_threads: The number of threads to use for prediction. If `None`, it
        defaults to the number of available CPU cores.

    Returns:
      A NumPy array containing the predictions. The shape and content vary by
      task:

      - **`Task.CLASSIFICATION`**:
        - **Binary Classification** (2 classes): An array of shape
          `[num_examples]`. Each value is the probability of the positive class
          (at `model.label_classes()[1]`). The probability of the negative class
          is `1 - prediction`.
        - **Multi-class Classification** (>2 classes): An array of shape
          `[num_examples, num_classes]`. Each row contains the probabilities
          for each class, in the order of `model.label_classes()`.

      - **`Task.REGRESSION`**: An array of shape `[num_examples]`, where each
        value is the predicted numerical outcome.

      - **`Task.RANKING`**: An array of shape `[num_examples]`, where each value
        is the predicted score for the item. Higher scores indicate higher
        rank.

      - **`Task.CATEGORICAL_UPLIFT`** and **`Task.NUMERICAL_UPLIFT`**: An array
      of
        shape `[num_examples]`. Each value is the predicted uplift, representing
        the incremental effect of the treatment.

      - **`Task.ANOMALY_DETECTION`**: An array of shape `[num_examples]`, where
        each value is the anomaly score (0 for most normal, 1 for most
        anomalous).
    """
    raise NotImplementedError

  def predict_shap(
      self,
      data: dataset.InputDataset,
      *,
      num_threads: Optional[int] = None,
  ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Computes SHAP values for each example in the given dataset.

    SHAP (SHapley Additive exPlanations) values explain a prediction by
    attributing the outcome to each feature. The sum of an example's SHAP values
    plus the model's initial prediction (`initial_value`) equals the model's raw
    prediction (before any activation function like sigmoid).

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train a model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").train(train_ds)

    # Compute SHAP values on the test dataset
    test_ds = pd.read_csv("test.csv")
    shap_values, initial_value = model.predict_shap(test_ds)
    ```

    Args:
      data: The dataset to compute SHAP values for. If it contains the label
        column, it will be ignored.
      num_threads: The number of threads to use. Defaults to the number of
        available CPU cores.

    Returns:
      A tuple `(shap_values, initial_value)` where:
        - `shap_values`: A dictionary mapping feature names to NumPy arrays.
          Each array has a shape of `[num_examples]` or `[num_examples,
          num_outputs]`, containing the SHAP values for that feature.
        - `initial_value`: A NumPy array of shape `[]` or `[num_outputs]`
          representing the model's initial prediction (i.e., offset).
    """
    raise NotImplementedError("SHAP is not implemented for this model")

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
      map_truncation: int = 5,
      use_slow_engine: bool = False,
      num_threads: Optional[int] = None,
  ) -> metric.Evaluation:
    """Evaluates the quality of a model on a dataset.

    In a notebook environment, the returned `Evaluation` object is displayed as
    a rich HTML report with plots.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train a model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").train(train_ds)

    # Evaluate the model on a test dataset
    test_ds = pd.read_csv("test.csv")
    evaluation = model.evaluate(test_ds)

    # Display the evaluation report in a notebook
    evaluation
    ```

    You can also evaluate the model on a different task than it was trained for,
    by overriding the `task`, `label`, and `group` arguments.

    ```python
    # Train a regression model
    model = ydf.RandomForestLearner(label="price",
    task=ydf.Task.REGRESSION).train(...)

    # Evaluate it as a ranking model
    ranking_evaluation = model.evaluate(
        test_ds, task=ydf.Task.RANKING, group="session_id"
    )
    ```

    Args:
      data: The dataset for evaluation.
      weighted: If `True`, the evaluation is weighted using the training
        weights. If `False`, it is unweighted. If `None` (default), it defaults
        to `False` with a warning if the model was trained with weights. The
        default value will change to `True` in a future version.
      task: Overrides the model's task for this evaluation. Defaults to the
        model's original task.
      label: Overrides the label column for this evaluation. Defaults to the
        model's original label.
      group: Overrides the grouping column for this evaluation, used for ranking
        tasks. Defaults to the model's original group column.
      bootstrapping: If `True`, enables bootstrapping with 2000 samples to
        compute confidence intervals and statistical tests. If an integer (>=
        100) is provided, it specifies the number of samples.
      ndcg_truncation: The truncation level for the NDCG metric.
      mrr_truncation: The truncation level for the MRR metric.
      map_truncation: The truncation level for the MAP metric.
      use_slow_engine: If `True`, uses a slower, more robust inference engine.
        See `predict()` for details.
      num_threads: The number of threads to use. Defaults to the number of
        available CPU cores.

    Returns:
      An `Evaluation` object containing the model's performance metrics.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def analyze_prediction(
      self,
      single_example: dataset.InputDataset,
      features: Optional[List[str]] = None,
  ) -> analysis.PredictionAnalysis:
    """Explains a single prediction of the model.

    This method shows how each feature value contributed to the final
    prediction for a specific example. For a global model analysis, use
    `model.analyze()` instead.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train a model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").train(train_ds)

    # Explain the prediction for the first example in the test set
    test_ds = pd.read_csv("test.csv")
    first_example = test_ds.iloc[:1]
    explanation = model.analyze_prediction(first_example)

    # Display the explanation in a notebook.
    explanation
    ```

    Args:
      single_example: A dataset containing a single example to explain.
      features: If specified, the analysis will be limited to these features,
        and they will be displayed in the specified order.

    Returns:
      A `PredictionAnalysis` object containing the explanation.
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
      permutation_variable_importance: bool = True,
      shap_values: bool = True,
      permutation_variable_importance_rounds: int = 1,
      num_threads: Optional[int] = None,
      maximum_duration: Optional[float] = 20,
      features: Optional[List[str]] = None,
  ) -> analysis.Analysis:
    """Analyzes the model's structure and its behavior on a dataset.

    An analysis includes structural information (e.g., variable importances) and
    performance characteristics on the given dataset (e.g., partial dependence
    plots). Computing the analysis can be time-consuming on large datasets. It
    is generally recommended to run analysis on a test set, not the training
    set.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train a model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").train(train_ds)

    # Analyze the model on a test set
    test_ds = pd.read_csv("test.csv")
    analysis = model.analyze(test_ds)

    # Display the analysis report in a notebook
    analysis
    ```

    Args:
      data: The dataset for analysis.
      sampling: The fraction of examples to use for the analysis (e.g., 0.1 for
        10%). On large datasets, a smaller sample can significantly speed up
        computation.
      num_bins: The number of bins for accumulating statistics in plots. More
        bins provide higher resolution but take longer to compute.
      partial_dependence_plot: If `True`, computes Partial Dependence Plots
        (PDPs), which can be computationally expensive.
      conditional_expectation_plot: If `True`, computes Conditional Expectation
        Plots (CEPs), which are computationally cheap.
      permutation_variable_importance: If `True`, computes permutation variable
        importance.
      shap_values: If `True`, computes SHAP-based metrics.
      permutation_variable_importance_rounds: The number of rounds for
        permutation variable importance. More rounds increase accuracy but take
        longer. A value of 1 is often sufficient. Set to 0 to disable.
      num_threads: The number of threads to use. Defaults to the number of
        available CPU cores.
      maximum_duration: The approximate maximum duration of the analysis in
        seconds. The analysis may run slightly longer.
      features: If specified, PDP and CEP plots will be limited to these
        features and displayed in this order.

    Returns:
      An `Analysis` object containing the results.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def to_cpp(self, key: str = "my_model") -> str:
    """Generates C++ code (.h file) for running the model.

    This method provides a fast and widely compatible way to deploy YDF models
    in C++. For applications where binary size is critical, `to_standalone_cc`
    is an alternative that produces much smaller binaries with zero
    dependencies, but may be slower and less compatible with all model types.

    **How to use:**

    1.  Generate the header file:
        `open("model.h", "w").write(model.to_cpp())`
    2.  In your Bazel/Blaze `BUILD` file, add the necessary dependencies:
        ```
        //third_party/absl/status:statusor
        //third_party/absl/strings
        //external/ydf_cc/yggdrasil_decision_forests/api:serving
        ```
    3.  In your C++ code, include the header and use the model:
        ```cpp
        #include "path/to/model.h"
        #include "yggdrasil_decision_forests/api/serving.h"

        namespace ydf = yggdrasil_decision_forests;
        // Load the model once.
        const auto model = ydf::exported_model_123::LoadModel("<path to model
        dir>");
        // Run predictions.
        predictions = model.Predict(...);
        ...
        ```
    4.  The generated `Predict` function uses placeholder values for features.
        You will need to modify this function to accept your own input data and
        populate the `examples->Set(...)` calls accordingly.
    5.  For optimal performance, pre-allocate and reuse the `examples` and
        `predictions` objects for each thread.

    The generated file contains further documentation.

    Args:
      key: A name for the model, used to create a unique C++ namespace.

    Returns:
      A string containing the C++ header code.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def to_standalone_cc(
      self,
      name: str = "ydf_model",
      algorithm: Literal["IF_ELSE", "ROUTING"] = "ROUTING",
      classification_output: Literal["CLASS", "SCORE", "PROBABILITY"] = "CLASS",
      categorical_from_string: bool = False,
  ) -> Union[str, Dict[str, bytes]]:
    """Generates standalone, dependency-free C++ code for model inference.

    This method is ideal for size-critical applications. See `to_cpp` for an
    alternative with better performance and model compatibility.

    **How to use:**

    1.  Copy the generated C++ code into a `.h` file.
    2.  In your C++ code, include the header and call the prediction function:
        ```cpp
        #include "path/to/generated_model.h"
        using namespace <name>;
        const auto pred = Prediction(Instance{.f1=5.0, .f2=F2::kRed});
        ```
        The function is thread-safe.

    Alternatively, you can use the `cc_ydf_standalone_model` Bazel rule for
    automated code generation (internal to Google).

    1. Save the model with `model.save(...)` in a directory in Google3.
    2. Create a BUILD file with a filegroup in the model directory e.g.:
      ```
      filegroup(
        name = "model",
        srcs = glob(["**"]),
      )
      ```
    3. In your library's BUILD, create a "cc_ydf_standalone_model " build rule.
      ```
      load("//external/ydf_cc/yggdrasil_decision_forests/serving/embed:embed.bzl",
        "cc_ydf_standalone_model ")
      cc_ydf_standalone_model (
        name = "my_model",
        classification_output = "SCORE",
        data = "<path to filegroup>",
      )
      ```
    4. In your cc_binary or cc_library, add ":my_model" as a dependency.
    5. In your C++ code, include:
      ```c++
      #include "<path to BUILD>/my_model.h"
      ```
      Then call:
      ```c++
      using namespace <name>;
      const auto pred = Prediction(Instance{.f1=5, f2=F2:kRed});
      ```

    Args:
      name: A name for the model, used to create the C++ namespace.
      algorithm: The underlying algorithm for prediction. - "ROUTING" (default):
        Faster and produces a smaller binary. - "IF_ELSE": Generates
        human-readable if-else conditions.
      classification_output: The output format for classification models. -
        "CLASS" (default): The predicted class index (fast). - "SCORE": The raw
        scores (e.g., logits) for all classes. - "PROBABILITY": The
        probabilities for all classes (slower, as it requires a softmax).
      categorical_from_string: If `True`, generates helper functions to convert
        strings to categorical feature enum values.

    Returns:
      A string with the C++ source code, or a dictionary of filename to source
      code if multiple files are generated.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def to_standalone_java(
      self,
      name: str = "YdfModel",
      package_name: str = "com.example.ydfmodel",
      classification_output: Literal["CLASS", "SCORE", "PROBABILITY"] = "CLASS",
  ) -> Dict[str, bytes]:
    """Generates standalone, dependency-free Java code for model inference.

    This method is ideal for size-critical applications.

    **How to use:**

    1.  Call this function to get the generated code and data:
        ```python
        model = ydf.load_model(...)
        java_files = model.to_standalone_java(
            name="MyYdfModel",
            package_name="com.mycompany.myproject"
        )
        ```

    2.  The function returns a dictionary containing two items:
        - Key: `{name}.java` (e.g., "MyYdfModel.java"): Value is the Java source
          code as bytes.
        - Key: `{name}Data.bin` (e.g., "MyYdfModelData.bin"): Value is the
          binary model data as bytes.

    3.  Save these files to your Java project:
        ```python
        with open(f"{name}.java", "wb") as f:
            f.write(java_files[f"{name}.java"])
        with open(f"{name}Data.bin", "wb") as f:
            f.write(java_files[f"{name}Data.bin"])
        ```
        Place the `{name}Data.bin` file in the Java classpath, typically in the
        resources directory.

    4.  In your Java code, import the generated class and use the static
        `predict` method:
        ```java
        import com.mycompany.myproject.MyYdfModel;

        // Create an Instance with feature values.
        // Categorical features are represented by enums in the generated class.
        MyYdfModel.Instance instance = new MyYdfModel.Instance(
            5.0f, // Numerical feature
            MyYdfModel.FeatureF2.kRed // Categorical feature
        );

        // Get the prediction.
        float prediction = MyYdfModel.predict(instance);
        ```
        The `predict` function is thread-safe. The generated class also
        contains enums for all categorical features.

    Args:
      name: A name for the model, used to create the Java class name.
      package_name: The Java package name for the generated class.
      classification_output: The output format for classification models. -
        "CLASS" (default): The predicted class enum. - "SCORE": The raw scores
        (e.g., logits) for all classes. - "PROBABILITY": The probabilities for
        all classes.

    Returns:
      A dictionary of filename to source code. This includes the Java source
      file and a binary resource file containing the model data.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def to_tensorflow_saved_model(  # pylint: disable=dangerous-default-value
      self,
      path: str,
      input_model_signature_fn: Any = None,
      *,
      mode: Literal["keras", "tf"] = "tf",
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
    """Exports the model as a TensorFlow SavedModel.

    This function requires TensorFlow and TensorFlow Decision Forests to be
    installed. Install them by running the command `pip install
    tensorflow_decision_forests`. The generated SavedModel relies on the
    TensorFlow Decision Forests Custom Inference Op. This op is available by
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

    TensorFlow SavedModels do not automatically cast feature values. For
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
    ydf_model = ydf.RandomForestLearner(
        label="l",
        task=ydf.Task.CLASSIFICATION,
    ).train(processed_dataset)

    # Export the model to a raw SavedModel model with the pre-processing
    model.to_tensorflow_saved_model(
        path="/tmp/my_model",
        mode="tf",
        feed_example_proto=False,
        pre_processing=pre_processing,
        tensor_specs={
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

    Note that export to Tensorflow is not yet available for Isolation Forest
    models.

    Args:
      path: Path to store the TensorFlow Decision Forests model.
      input_model_signature_fn: A lambda that returns the
        (Dense,Sparse,Ragged)TensorSpec (or structure of TensorSpec e.g.
        dictionary, list) corresponding to input signature of the model. If not
        specified, the input model signature is created by
        `tfdf.keras.build_default_input_model_signature`. For example, specify
        `input_model_signature_fn` if a numerical input feature (which is
        consumed as DenseTensorSpec(float32) by default) will be fed differently
        (e.g. RaggedTensor(int64)). Only compatible with mode="keras".
      mode: How the YDF model is converted into a TensorFlow SavedModel. 1) mode
        = "keras" (default): Turn the model into a Keras 2 model using
        TensorFlow Decision Forests, and then save it with
        `tf_keras.models.save_model`. 2) mode = "tf" (recommended; will become
        default): Turn the model into a TensorFlow Module, and save it with
        `tf.saved_model.save`.
      feature_dtypes: Mapping from feature name to TensorFlow dtype. Use this
        mapping to override feature dtypes. For instance, numerical features are
        encoded with tf.float32 by default. If you plan on feeding tf.float64 or
        tf.int32, use `feature_dtype` to specify it. `feature_dtypes` is ignored
        if `tensor_specs` is set. If set, disables the automatic signature
        extraction on `pre_processing` (if `pre_processing` is also set). Only
        compatible with mode="tf".
      servo_api: If true, adds a SavedModel signature to make the model
        compatible with the `Classify` or `Regress` servo APIs. Only compatible
        with mode="tf". If false, outputs the raw model predictions.
      feed_example_proto: If false, the model expects for the input features to
        be provided as TensorFlow values. This is the most efficient way to make
        predictions. If true, the model expects for the input features to be
        provided as a binary serialized TensorFlow Example proto. This is the
        format expected by VertexAI and most TensorFlow Serving pipelines.
      pre_processing: Optional TensorFlow function or module to apply on the
        input features before applying the model. If the `pre_processing`
        function has been traced (i.e., the function has been called once with
        actual data and contains a concrete instance in its cache), this
        signature is extracted and used as the signature of the SavedModel. Only
        compatible with mode="tf".
      post_processing: Optional TensorFlow function or module to apply on the
        model predictions. Only compatible with mode="tf".
      temp_dir: Temporary directory used during the conversion. If None
        (default), uses `tempfile.mkdtemp` default temporary directory.
      tensor_specs: Optional dictionary of `tf.TensorSpec` that define the input
        features of the model to export. If not provided, the `TensorSpec`s are
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
      force: Tries to export even in currently unsupported environments.
        WARNING: Setting this to true may crash the Python runtime.
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
    """Converts the model into a callable TensorFlow Module (`@tf.function`).

    This allows the YDF model to be integrated into larger TensorFlow graphs.
    Requires `tensorflow_decision_forests` (`pip install
    tensorflow_decision_forests`).

    Note: Export to TensorFlow is not yet available for Anomaly Detection
    models.

    Usage example:

    ```python
    import ydf
    import numpy as np
    import tensorflow as tf

    # Train a model
    model = ydf.RandomForestLearner(label="l").train({
        "f1": np.random.random(100),
        "l": np.random.randint(2, size=100),
    })

    # Convert to a TF Module
    tf_model_fn = model.to_tensorflow_function()

    # Make predictions
    predictions = tf_model_fn({"f1": tf.constant([0.1, 0.5, 0.9])})
    ```

    Args:
      temp_dir: A temporary directory for the conversion process.
      can_be_saved: If `True` (default), the returned module can be saved with
        `tf.saved_model.save`, and temporary files are preserved. If `False`,
        temporary files are deleted, and the module cannot be saved.
      squeeze_binary_classification: If `True` (default), binary classification
        models will output a tensor of shape `[num_examples]` with the
        probability of the positive class. If `False`, the output is shape
        `[num_examples, 2]`.
      force: If `True`, attempts to export even in unsupported environments.

    Returns:
      A `tf.Module` containing the model.
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
    """Converts the model into a JAX function for use in JAX ecosystems.

    Usage example:

    ```python
    import ydf
    import numpy as np
    import jax.numpy as jnp

    # Train a model
    model = ydf.GradientBoostedTreesLearner(label="l").train({
        "f1": np.random.random(100),
        "l": np.random.randint(2, 100),
    })

    # Convert to a JAX function
    jax_model = model.to_jax_function()

    # Make predictions
    predictions = jax_model.predict({
        "f1": jnp.array([0.1, 0.5, 0.9]),
    })
    ```

    Args:
      jit: If `True`, the returned function will be just-in-time compiled with
        `@jax.jit`.
      apply_activation: If `True`, the model's activation function (e.g.,
        sigmoid) will be applied to the output.
      leaves_as_params: If `True`, the model's leaf values are exported as
        learnable parameters. The returned object will contain a `params`
        attribute, which must be passed to the `predict` function. This is
        useful for fine-tuning.
      compatibility: The JAX runtime compatibility. Can be "XLA" (default) or
        "TFL" (for TensorFlow Lite).

    Returns:
      A dataclass containing the JAX prediction function (`predict`), and
      optionally the model parameters (`params`) and a feature encoder
      (`encoder`).
    """
    raise NotImplementedError

  @abc.abstractmethod
  def update_with_jax_params(self, params: Dict[str, Any]):
    """Updates the model's parameters with values from a JAX fine-tuning process.

    This function allows you to take a model fine-tuned in JAX (after being
    exported with `to_jax_function(leaves_as_params=True)`) and update the
    original YDF model object with the new parameters.

    Usage example:

    ```python
    import ydf
    import jax

    # Train a model with YDF
    # dataset = ...
    model = ydf.GradientBoostedTreesLearner(label="l").train(dataset)

    # Convert to a JAX function with learnable parameters
    jax_model = model.to_jax_function(leaves_as_params=True)

    # Fine-tune the parameters in JAX
    # jax_model.params = my_fine_tuning_logic(jax_model.params, ...)

    # Update the YDF model with the new parameters
    model.update_with_jax_params(jax_model.params)

    # The YDF model now reflects the fine-tuning
    # model.save("/path/to/finetuned_model")
    ```

    Args:
      params: A dictionary of model parameters, as produced by
        `to_jax_function`.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def hyperparameter_optimizer_logs(
      self,
  ) -> Optional[optimizer_logs.OptimizerLogs]:
    """Returns the logs of the hyperparameter tuning process, if any.

    Returns:
      An `OptimizerLogs` object containing the tuning trials, or `None` if the
      model was not trained with hyperparameter tuning.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def variable_importances(self) -> Dict[str, List[Tuple[float, str]]]:
    """Returns the variable importances (VIs) of the model.

    Variable importances indicate how much each feature contributes to the
    model's predictions. Different VI metrics have different semantics and are
    generally not comparable.

    The available VIs depend on the learning algorithm and its hyperparameters.
    For example, for Random Forest, setting
    `compute_oob_variable_importances=True`
    enables the computation of permutation out-of-bag VIs.

    Usage example:

    ```python
    # Train a Random Forest and enable OOB VI computation.
    learner = ydf.RandomForestLearner(
        label="species", compute_oob_variable_importances=True
    )
    model = learner.train(dataset)

    # List available VI metrics.
    print(model.variable_importances().keys())
    # dict_keys(['NUM_AS_ROOT', 'SUM_SCORE', 'MEAN_DECREASE_IN_ACCURACY'])

    # Get a specific VI, sorted by importance.
    vi = model.variable_importances()["MEAN_DECREASE_IN_ACCURACY"]
    # [('bill_length_mm', 0.0713), ('island', 0.0072), ...]
    ```

    Returns:
      A dictionary where keys are the names of the VI metrics and values are
      lists of `(importance_value, feature_name)` tuples, sorted in descending
      order of importance.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def label_col_idx(self) -> int:
    """Returns the index of the label column in the dataspec.

    Returns:
       The column index, or -1 if the model has no label.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def input_features_col_idxs(self) -> Sequence[int]:
    """Returns the column indices of the input features in the dataspec."""
    raise NotImplementedError

  def self_evaluation(self) -> metric.Evaluation:
    """Returns the model's self-evaluation, computed during training.

    The method of self-evaluation depends on the model type. For example,
    Random Forests use out-of-bag (OOB) evaluation, while Gradient Boosted
    Trees use evaluation on a validation dataset. Because of this, self-
    evaluations are not directly comparable between different model types.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train a model
    train_ds = pd.read_csv("train.csv")
    model = ydf.GradientBoostedTreesLearner(label="label").train(train_ds)

    # Get the self-evaluation
    self_evaluation = model.self_evaluation()

    # In a notebook, this will print a rich report.
    self_evaluation
    ```

    Returns:
      An `Evaluation` object with the metrics.
    """
    raise NotImplementedError(
        "Self-evaluation is not available for this model type."
    )

  @abc.abstractmethod
  def list_compatible_engines(self) -> Sequence[str]:
    """Lists the inference engines compatible with the model.

    The engines are sorted from likely-fastest to likely-slowest.

    Returns:
      A list of names of compatible inference engines.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def force_engine(self, engine_name: Optional[str]) -> None:
    """Forces the model to use a specific inference engine.

    By default (`engine_name=None`), the model automatically uses the fastest
    compatible engine. This method allows you to override that behavior.

    If an invalid or incompatible engine name is provided, subsequent calls to
    `predict()`, `evaluate()`, etc., will fail.

    Args:
      engine_name: The name of a compatible engine, or `None` to restore
        automatic selection.
    """
    raise NotImplementedError

  def training_logs(self) -> List[TrainingLogEntry]:
    """Returns the model's training logs.

    The training logs contain performance metrics calculated periodically
    during model training. The content and evaluation method depend on the
    model type (e.g., out-of-bag for Random Forest, validation set for
    Gradient Boosted Trees).

    Usage example:

    ```python
    import pandas as pd
    import ydf
    import matplotlib.pyplot as plt

    # Train a model
    train_ds = pd.read_csv("train.csv")
    model = ydf.GradientBoostedTreesLearner(label="label").train(train_ds)

    # Get the training logs
    logs = model.training_logs()

    # Plot the accuracy over training iterations
    plt.plot(
        [log.iteration for log in logs],
        [log.evaluation.accuracy for log in logs]
    )
    plt.xlabel("Iteration (Number of Trees)")
    plt.ylabel("Validation Accuracy")
    plt.show()
    ```

    Returns:
      A list of `TrainingLogEntry` objects.
    """
    raise NotImplementedError(
        "Training logs are not available for this model type."
    )

  def input_features(self) -> Sequence[InputFeature]:
    """Returns the input features of the model.

    The features are sorted by their column index in the data specification.

    Returns:
        A list of `InputFeature` objects.
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

    The feature names are sorted by their column index in the data
    specification.

    Returns:
        A list of feature name strings.
    """

    return [f.name for f in self.input_features()]

  def label(self) -> Optional[str]:
    """Returns the name of the label column.

    Returns:
      The label column name as a string, or `None` if the model has no label.
    """
    label_col_idx = self.label_col_idx()
    if label_col_idx < -1:
      raise ValueError(
          f"Invalid label column index {label_col_idx}. This model might be"
          " corrupted."
      )
    if label_col_idx == -1:
      return None
    return self.data_spec().columns[self.label_col_idx()].name

  def label_classes(self) -> List[str]:
    """Returns the list of possible label values for a classification model.

    The order of the classes in the returned list corresponds to the order of
    probabilities in the output of `model.predict()`.

    Returns:
      A list of class name strings.

    Raises:
      ValueError: If the model is not a classification model.
    """
    if self.task() != Task.CLASSIFICATION:
      raise ValueError(
          "Label classes are only available for classification models. This"
          f" model has task {self.task().name}"
      )
    label_column = self.data_spec().columns[self.label_col_idx()]
    if label_column.type != data_spec_pb2.CATEGORICAL:
      semantic = dataspec.Semantic.from_proto_type(label_column.type)
      raise ValueError(
          "A CATEGORICAL column is expected for a classification label, but"
          f" got {semantic} instead. Should the task be REGRESSION? If so, set"
          " `task=ydf.Task.REGRESSION` in the learner constructor."
      )

    if label_column.categorical.is_already_integerized:
      log.info(
          "The label column is integerized. This is expected for models trained"
          " with TensorFlow Decision Forests."
      )

    # The first element is the "out-of-vocabulary" value, which is not used.
    return dataspec.categorical_column_dictionary_to_list(label_column)[1:]

  def predict_class(
      self,
      data: dataset.InputDataset,
      *,
      use_slow_engine: bool = False,
      num_threads: Optional[int] = None,
  ) -> np.ndarray:
    """Returns the most likely predicted class for a classification model.

    This is a convenience method for classification tasks. It returns a NumPy
    array of strings representing the predicted class for each example. In case
    of a tie in probabilities, the class that appears first in
    `model.label_classes()` is chosen.

    For the full class probabilities, use `model.predict()`.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train a classification model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="category").train(train_ds)

    # Get the predicted class for each example
    test_ds = pd.read_csv("test.csv")
    predicted_classes = model.predict_class(test_ds)
    ```

    Args:
      data: The dataset to make predictions on.
      use_slow_engine: If `True`, uses a slower, more robust inference engine.
        See `predict()` for details.
      num_threads: The number of threads to use. Defaults to the number of
        available CPU cores.

    Returns:
      A NumPy array of strings of shape `[num_examples]`, containing the most
      likely predicted class for each example.

    Raises:
      ValueError: If the model is not a classification model.
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
      # For binary classification, predict() returns the probability of the
      # positive class.
      return np.take(label_classes, (prediction_proba >= 0.5).astype(int))
    else:
      # For multi-class, find the index of the highest probability
      prediction_class_idx = np.argmax(prediction_proba, axis=1)
      return np.take(label_classes, prediction_class_idx)


class GenericCCModel(GenericModel):
  """Abstract superclass for YDF models implemented in C++."""

  def __init__(self, raw_model: ydf.GenericCCModel):
    self._model = raw_model

  def name(self) -> str:
    return self._model.name()

  def __getstate__(self):
    """Serializes the model for pickling.

    Warning: Pickling is discouraged. For saving a model to disk, use
    `model.save()`. For serializing to a byte string, use `model.serialize()`.

    Returns:
      The serialized state of the model.
    """
    log.warning(
        "Model pickling is discouraged. To save a model on disk, use"
        " `model.save(path)` and `ydf.load_model(path)`. To"
        " serialize a model to bytes, use `data = model.serialize()` and"
        " `ydf.deserialize_model(data)` instead.",
        message_id=log.WarningMessage.DONT_USE_PICKLE,
    )
    return self._model.Serialize()

  def __setstate__(self, state):
    """Deserializes the model from a pickled state."""
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
          "The benchmark duration must be positive, but got"
          f" {benchmark_duration}."
      )
    if warmup_duration <= 0:
      raise ValueError(
          f"The warmup duration must be positive, but got {warmup_duration}."
      )
    if batch_size <= 0:
      raise ValueError(
          f"The batch size must be positive, but got {batch_size}."
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
    if advanced_options.file_prefix is None:
      if os.path.exists(path) and os.path.isdir(path):
        with os.scandir(path) as it:
          if any(it):
            logging.warning(
                "The directory %r to save the model to is not empty. This"
                " can lead to model corruption. To avoid this, specify an"
                " empty or non-existing directory, or use `advanced_options`"
                " to set a file prefix for the model.",
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
      # The data spec contains columns like label, weights, etc., which are
      # not required for prediction.
      ds = dataset.create_vertical_dataset(
          data,
          data_spec=self._model.data_spec(),
          required_columns=self.input_feature_names(),
      )
      result = self._model.Predict(
          ds._dataset, use_slow_engine, num_threads=num_threads  # pylint: disable=protected-access
      )
    return result

  def predict_shap(
      self,
      data: dataset.InputDataset,
      *,
      num_threads: Optional[int] = None,
  ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    if num_threads is None:
      num_threads = concurrency.determine_optimal_num_threads(training=False)

    with log.cc_log_context():
      # The data spec contains columns like label, weights, etc., which are
      # not required for prediction.
      ds = dataset.create_vertical_dataset(
          data,
          data_spec=self._model.data_spec(),
          required_columns=self.input_feature_names(),
      )
      return self._model.PredictShap(
          ds._dataset, num_threads=num_threads  # pylint: disable=protected-access
      )

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
      map_truncation: int = 5,
      use_slow_engine: bool = False,
      num_threads: Optional[int] = None,
  ) -> metric.Evaluation:
    if num_threads is None:
      num_threads = concurrency.determine_optimal_num_threads(training=False)

    # Handle the default value for "weighted".
    if weighted is None:
      weighted = False
      if self._model.weighted_training():
        # TODO: Change default to True and remove warning.
        log.warning(
            "The model was trained with weights, but `weighted` was not"
            " specified in `evaluate()`. Using unweighted evaluation. To use"
            " weighted evaluation, set `weighted=True`. This warning will be"
            " removed and the default will change to `True` in a future"
            " version.",
            message_id=log.WarningMessage.WEIGHTED_NOT_SET_IN_EVAL,
        )

    # Check for unnecessary arguments.
    if task is not None and task == self.task():
      log.warning(
          "The `task` argument in `evaluate()` is the same as the model's"
          " training task. This argument can be omitted.",
          message_id=log.WarningMessage.UNNECESSARY_TASK_ARGUMENT,
      )
    if label is not None and label == self.label():
      log.warning(
          "The `label` argument in `evaluate()` is the same as the model's"
          " training label. This argument can be omitted.",
          message_id=log.WarningMessage.UNNECESSARY_LABEL_ARGUMENT,
      )

    if self.label() is None:
      if self.task() == Task.ANOMALY_DETECTION:
        raise ValueError(
            "This Anomaly Detection model was trained without a label. To"
            " enable evaluation, provide a `label` during training."
            " Alternatively, use `ydf.evaluate_predictions()` to evaluate"
            " predictions from an unlabeled dataset."
        )
      else:
        raise ValueError(
            "This model does not have a label and cannot be evaluated."
        )

    if isinstance(bootstrapping, bool):
      bootstrapping_samples = 2000 if bootstrapping else -1
    elif isinstance(bootstrapping, int) and bootstrapping >= 100:
      bootstrapping_samples = bootstrapping
    else:
      raise ValueError(
          "`bootstrapping` must be a boolean or an integer >= 100, but got"
          f" {bootstrapping!r}."
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
              ndcg_truncation=ndcg_truncation,
              mrr_truncation=mrr_truncation,
              map_truncation=map_truncation,
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
      features: Optional[List[str]] = None,
  ) -> analysis.PredictionAnalysis:
    with log.cc_log_context():
      ds = dataset.create_vertical_dataset(
          single_example, data_spec=self._model.data_spec()
      )

      options_proto = model_analysis_pb2.PredictionAnalysisOptions()
      if features is not None:
        options_proto.features.extend(features)
      analysis_proto = self._model.AnalyzePrediction(ds._dataset, options_proto)  # pylint: disable=protected-access
      return analysis.PredictionAnalysis(analysis_proto, options_proto)

  def analyze(
      self,
      data: dataset.InputDataset,
      sampling: float = 1.0,
      num_bins: int = 50,
      partial_dependence_plot: bool = True,
      conditional_expectation_plot: bool = True,
      permutation_variable_importance: bool = True,
      shap_values: bool = True,
      permutation_variable_importance_rounds: int = 1,
      num_threads: Optional[int] = None,
      maximum_duration: Optional[float] = 20,
      features: Optional[List[str]] = None,
  ) -> analysis.Analysis:
    if num_threads is None:
      num_threads = concurrency.determine_optimal_num_threads(training=False)

    enable_permutation_variable_importances = (
        permutation_variable_importance_rounds > 0
    ) and permutation_variable_importance
    if (
        enable_permutation_variable_importances
        and self.task() == Task.ANOMALY_DETECTION
        and self._model.label_col_idx() == -1
    ):
      # TODO: Allow AD evaluation and analysis without label at training.
      enable_permutation_variable_importances = False
      log.warning(
          "Permutation variable importance cannot be computed for Anomaly"
          " Detection models trained without a label. To enable this, provide"
          " a label during training.",
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
          shap_variable_importance=model_analysis_pb2.Options.ShapVariableImportance(
              enabled=shap_values,
              example_sampling=sampling,
          ),
          include_model_structural_variable_importances=True,
      )
      # TODO: Use "shap_values" to enable other SHAP based analyses.
      if features is not None:
        options_proto.features.extend(features)

      analysis_proto = self._model.Analyze(ds._dataset, options_proto)  # pylint: disable=protected-access
      return analysis.Analysis(analysis_proto, options_proto)

  def to_cpp(self, key: str = "my_model") -> str:
    return template_cpp_export.template(
        key, self._model.data_spec(), self._model.input_features()
    )

  def to_standalone_cc(
      self,
      name: str = "ydf_model",
      algorithm: Literal["IF_ELSE", "ROUTING"] = "ROUTING",
      classification_output: Literal["CLASS", "SCORE", "PROBABILITY"] = "CLASS",
      categorical_from_string: bool = False,
  ) -> Union[str, Dict[str, bytes]]:
    options = embed_pb2.Options(
        name=name,
        classification_output=embed_pb2.ClassificationOutput.Enum.Value(
            classification_output
        ),
        algorithm=embed_pb2.Algorithm.Enum.Value(algorithm),
        categorical_from_string=categorical_from_string,
        cc=embed_pb2.CC(),
    )
    results = self._model.EmbedModel(options)
    if len(results) == 1:
      return list(results.values())[0].decode()
    else:
      return results

  def to_standalone_java(
      self,
      name: str = "YdfModel",
      package_name: str = "com.example.ydfmodel",
      classification_output: Literal["CLASS", "SCORE", "PROBABILITY"] = "CLASS",
  ) -> Dict[str, bytes]:
    options = embed_pb2.Options(
        name=name,
        classification_output=embed_pb2.ClassificationOutput.Enum.Value(
            classification_output
        ),
        java=embed_pb2.Java(package_name=package_name),
    )
    return self._model.EmbedModel(options)

  # TODO: Change default value of "mode" before 1.0 release.
  def to_tensorflow_saved_model(  # pylint: disable=dangerous-default-value
      self,
      path: str,
      input_model_signature_fn: Any = None,
      *,
      mode: Literal["keras", "tf"] = "tf",
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
    # TODO: Add tensorflow support for anomaly detection.
    if self.task() == Task.ANOMALY_DETECTION:
      raise ValueError(
          "Export to TensorFlow is not yet supported for Anomaly Detection"
          " models."
      )
    if mode == "keras":
      log.warning(
          "Calling `to_tensorflow_saved_model(mode='keras', ...)`. The 'keras'"
          " mode is deprecated. Use `to_tensorflow_saved_model(mode='tf', ...)`"
          " instead, which is more efficient, has better compatibility, and"
          " offers more options.",
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
    # TODO: Add tensorflow support for anomaly detection.
    if self.task() == Task.ANOMALY_DETECTION:
      raise ValueError(
          "Export to TensorFlow is not yet supported for Anomaly Detection"
          " models."
      )
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
    """Exports the model as a self-contained Docker endpoint for deployment.

    This function creates a directory with a Dockerfile, the model, and all
    necessary support files to serve the model over an HTTP endpoint.

    Usage example:

    ```python
    import ydf
    import numpy as np

    # Train a model
    model = ydf.RandomForestLearner(label="l").train({
        "f1": np.random.random(size=100),
        "f2": np.random.random(size=100),
        "l": np.random.randint(2, size=100),
    })

    # Export the model to a Docker endpoint directory
    model.to_docker(path="/tmp/my_docker_model")

    # See the generated README for instructions
    !cat /tmp/my_docker_model/readme.md

    # Test the end-point locally
    docker build --platform linux/amd64 -t ydf_predict_image /tmp/my_model
    docker run --rm -p 8080:8080 -d ydf_predict_image

    # Deploy the model on Google Cloud
    gcloud run deploy ydf-predict --source /tmp/my_model

    # Check the automatically created utility scripts "test_locally.sh" and
    # "deploy_in_google_cloud.sh" for more examples.
    ```

    Args:
      path: The directory where the Docker endpoint files will be created.
      exist_ok: If `False` (default), raises an error if the `path` directory
        already exists. If `True`, overwrites the content of the directory if it
        exists.
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
      """Create a new or retrieve an existing column."""
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
              f" not expect a {usage} column."
          )
        col_idx = column_names.get(name, None)
        if col_idx is not None:
          # A column with the same name already exists
          existing_col_def = effective_dataspec.columns[col_idx]
          if existing_col_def.type != semantic:
            log.warning(
                "Adding dual semantic to %s column %r. Original"
                " semantic: %s New semantic: %s",
                usage,
                name,
                data_spec_pb2.ColumnType.Name(existing_col_def.type),
                data_spec_pb2.ColumnType.Name(semantic),
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
        # Copy the dictionary of the categorical label
        new_col.categorical.CopyFrom(
            effective_dataspec.columns[default_col_idx].categorical
        )
      elif semantic == data_spec_pb2.ColumnType.CATEGORICAL:
        # Create a binary-looking category
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

  Currently supported models:
    - `sklearn.tree.DecisionTreeClassifier`
    - `sklearn.tree.DecisionTreeRegressor`
    - `sklearn.tree.ExtraTreeClassifier`
    - `sklearn.tree.ExtraTreeRegressor`
    - `sklearn.ensemble.RandomForestClassifier`
    - `sklearn.ensemble.RandomForestRegressor`
    - `sklearn.ensemble.ExtraTreesClassifier`
    - `sklearn.ensemble.ExtraTreesRegressor`
    - `sklearn.ensemble.GradientBoostingRegressor`
    - `sklearn.ensemble.IsolationForest`

  Scikit-learn models do not have named features, so the input features are
  combined into a single multi-dimensional feature. You can specify its name
  with the `feature_name` argument.

  Usage example:

  ```python
  import ydf
  from sklearn import datasets
  from sklearn import tree
  import numpy as np

  # Train a scikit-learn model
  X, y = datasets.make_classification(n_features=4, n_classes=2)
  skl_model = tree.DecisionTreeClassifier().fit(X, y)

  # Convert the model to YDF
  ydf_model = ydf.from_sklearn(skl_model)

  # Make predictions with the YDF model
  # The input must be a dictionary with the specified feature name.
  ydf_predictions = ydf_model.predict({"features": X})

  # Analyze the YDF model
  # analysis_ds = {"features": X, "label": y}
  # ydf_model.analyze(analysis_ds)
  ```

  Args:
    sklearn_model: The scikit-learn tree-based model to convert.
    label_name: The name to assign to the label column in the YDF model.
    feature_name: The name to assign to the multi-dimensional feature column in
      the YDF model.

  Returns:
    A YDF model that emulates the provided scikit-learn model.
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
        '"jax" is needed by this function. Make sure it is installed and try'
        " again. See https://jax.readthedocs.io/en/latest/installation.html"
    ) from exc


def _get_export_tf():
  try:
    from ydf.model import export_tf  # pylint: disable=g-import-not-at-top,import-outside-toplevel # pytype: disable=import-error

    return export_tf
  except ImportError as exc:
    raise ValueError(
        '"tensorflow_decision_forests" is needed by this function. Make sure'
        " it is installed and try again. If using pip, run `pip install"
        " tensorflow_decision_forests`."
    ) from exc


def _get_export_sklearn():
  try:
    from ydf.model import export_sklearn  # pylint: disable=g-import-not-at-top,import-outside-toplevel # pytype: disable=import-error

    return export_sklearn
  except ImportError as exc:
    raise ValueError(
        '"scikit-learn" is needed by this function. Make sure '
        "it is installed and try again. If using pip, run `pip install"
        " scikit-learn`."
    ) from exc


def _get_export_docker():
  try:
    from ydf.model import export_docker  # pylint: disable=g-import-not-at-top,import-outside-toplevel # pytype: disable=import-error

    return export_docker
  except ImportError as exc:
    raise ValueError("Cannot import the export_docker utility") from exc


ModelType = TypeVar("ModelType", bound=GenericModel)
