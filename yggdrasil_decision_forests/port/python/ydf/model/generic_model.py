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
import tempfile
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union

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
  """

  CLASSIFICATION = "CLASSIFICATION"
  REGRESSION = "REGRESSION"
  RANKING = "RANKING"
  CATEGORICAL_UPLIFT = "CATEGORICAL_UPLIFT"
  NUMERICAL_UPLIFT = "NUMERICAL_UPLIFT"

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
}
PROTO_TO_TASK = {v: k for k, v in TASK_TO_PROTO.items()}


@dataclasses.dataclass(frozen=True)
class ModelIOOptions:
  """Advanced options for saving and loading YDF models.

  Attributes:
      file_prefix: Optional prefix for the model. File prefixes allow multiple
        models to exist in the same folder. Doing so is heavily DISCOURAGED
        outside of edge cases. When loading a model, the prefix, if not
        specified, auto-detected if possible. When saving a model, the empty
        string is used as file prefix unless it is explicitly specified.
  """

  file_prefix: Optional[str] = None


@enum.unique
class NodeFormat(enum.Enum):
  """Serialization format for a model.

  Determines the storage format for nodes.

  Attributes:
    BLOB_SEQUENCE: Default format for the public version of YDF.
  """

  BLOB_SEQUENCE = enum.auto()


class GenericModel:
  """Abstract superclass for all YDF models."""

  def __init__(self, raw_model: ydf.GenericCCModel):
    self._model = raw_model

  def name(self) -> str:
    """Returns the name of the model type."""
    return self._model.name()

  def task(self) -> Task:
    """Task solved by the model."""
    return Task._from_proto_type(self._model.task())

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
      format: Literal["auto", "text", "notebook", "html"] = "auto",
      full_details: bool = False,
  ) -> Union[str, html.HtmlNotebookDisplay]:
    """Description of the model.

    Args:
      format: Format of the display: - auto: Use the "notebook" format if
        executed in an IPython notebook. Otherwise, use the "text" format. -
        text: Text description of the model. - html: Html description of the
          model. -
        notebook: Html description of the model displayed in a notebook cell.
      full_details: Should the full model be printed. This can be large.

    Returns:
      The model description.
    """

    if format == "auto":
      format = "text" if log.is_direct_output() else "notebook"

    with log.cc_log_context():
      description = self._model.Describe(full_details, format == "text")
      if format == "notebook":
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
    C++ API can be generated with `to_cpp()`.

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
          required_columns=self._input_feature_names(),
      )
      result = self._model.Benchmark(
          vds._dataset, benchmark_duration, warmup_duration, batch_size
      )
    return result

  def save(self, path, advanced_options=ModelIOOptions()) -> None:
    """Save the model to disk.

    YDF uses a proprietary model format for saving models. A model consists of
    multiple files located in the same directory.
    A directory should only contain a single YDF model. See `advanced_options`
    for more information. See
    https://ydf.readthedocs.io/en/latest/convert_model.html for more information
    about the YDF model format.

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
    with log.cc_log_context():
      # The data spec contains the label / weights /  ranking group / uplift
      # treatment column, but those are not required for making predictions.
      ds = dataset.create_vertical_dataset(
          data,
          data_spec=self._model.data_spec(),
          required_columns=self._input_feature_names(),
      )
      result = self._model.Predict(ds._dataset)  # pylint: disable=protected-access
    return result

  def evaluate(
      self,
      data: dataset.InputDataset,
      bootstrapping: Union[bool, int] = False,
  ) -> metric.Evaluation:
    """Evaluates the quality of a model on a dataset.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").Train(train_ds)

    test_ds = pd.read_csv("train.csv")
    evaluation = model.evaluates(test_ds)
    ```

    In a notebook, if a cell returns an evaluation object, this evaluation will
    be as a rich html and plots:

    ```
    evaluation = model.evaluates(test_ds)
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
          task=self.task()._to_proto_type(),
      )

      evaluation_proto = self._model.Evaluate(ds._dataset, options_proto)  # pylint: disable=protected-access
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
    model = ydf.RandomForestLearner(label="label").Train(train_ds)

    test_ds = pd.read_csv("test.csv")

    # We want to explain the model prediction on the first test example.
    selected_example = test_ds.iloc[:1]

    analysis = model.analyze_prediction(selected_example, test_ds)

    # Display the analysis in a notebook.
    analysis
    ```

    Args:
      single_example: Example to explain. Can be a dictionary of list or numpy
        array of values, Pandas DataFrame, or a VerticalDataset.

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
    importance, training logs), the dataset (e.g., column statistics), and the
    application of the model on the dataset (e.g. partial dependence plots).

    While some information might be valid, it is generatly not recommended to
    analyze a model on its training dataset.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").Train(train_ds)

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
    4. The generated "Predict" function takes no inputs. Instead, it fill the
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

  def to_tensorflow_saved_model(
      self, path: str, input_model_signature_fn: Any = None
  ) -> None:
    """Exports the model as a TensorFlow Saved model.

    The generated model is a Tensorflow SavedModel that requires the TensorFlow
    Decision Forests custom ops. It is compatible with TF Serving and
    Tensorflow.js, but it may not be compatible with other TensorFlow libraries.
    The signature of the resulting model is determined automatically based on
    the dataspec of the model unless given by `input_model_signature_fn`.

    Requires Tensorflow Decision Forests to be installed.

    Usage example:

    ```python
    !pip install tensorflow_decision_forests
    import tensorflow as tf
    import tensorflow_decision_forests as tfdf
    import ydf
    import pandas as pd

    path_to_tfdf_model = '/path/to/tfdfmodel'
    # `model` is a model trained with ydf.
    model.to_tensorflow_saved_model(path=path_to_tfdf_model)

    # Load the dataset and convert to TensorFlow
    test_df = pd.read_csv(test_dataset_path)
    tf_test = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, "Rings")

    # Load the model with TF-DF.
    tf_model = tf.keras.models.load_model(path_to_tfdf_model)
    # Run predictions with TF-DF
    tfdf_predictions = tf_model.predict(tf_test)
    ```

    Args:
      path: Path to store the Tensorflow Decision Forests model.
      input_model_signature_fn: A lambda that returns the
        (Dense,Sparse,Ragged)TensorSpec (or structure of TensorSpec e.g.
        dictionary, list) corresponding to input signature of the model. If not
        specified, the input model signature is created by
        `tfdf.keras.build_default_input_model_signature`. For example, specify
        `input_model_signature_fn` if an numerical input feature (which is
        consumed as DenseTensorSpec(float32) by default) will be feed
        differently (e.g. RaggedTensor(int64)).
    """
    try:
      import tensorflow_decision_forests as tfdf  # type:ignore
    except ImportError as exc:
      raise ValueError(
          "Exporting to tensorflow requires the tensorflow_decision_forests"
          " package to be installed. When using pip, run `pip install"
          " tensorflow_decision_forests`"
      ) from exc
    # Do not pass input_model_signature_fn if it is None.
    not_none_params = {}
    if input_model_signature_fn is not None:
      not_none_params["input_model_signature_fn"] = input_model_signature_fn
    with tempfile.TemporaryDirectory() as tmpdirname:
      self.save(tmpdirname)
      tfdf.keras.yggdrasil_model_to_keras_model(
          src_path=tmpdirname,
          dst_path=path,
          verbose=log.current_log_level(),
          **not_none_params,
      )

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

    Values are sorted by decreasing value/importance.

    Usage example:

    ```python
    # Train a Random Forest. Enable the computation of OOB (out-of-bag) variable
    # importances.
    model = ydf.RandomForestModel(compute_oob_variable_importances=True,
                                  label=...).train(ds)
    # List the available variable importances
    print(model.variable_importances().keys())

    # Show a specific variable importance
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

  def label_classes(self) -> List[str]:
    """Returns the label classes for classification tasks, None otherwise."""
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

  def _input_feature_names(self) -> List[str]:
    """Returns the names of the input features"""
    dataspec_columns = self.data_spec().columns
    return [dataspec_columns[idx].name for idx in self._model.input_features()]


ModelType = TypeVar("ModelType", bound=GenericModel)
