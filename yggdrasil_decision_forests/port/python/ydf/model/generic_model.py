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
import os
from typing import Optional, TypeVar, Union

from absl import logging
import numpy as np

from yggdrasil_decision_forests.metric import metric_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.cc import ydf
from ydf.dataset import dataset
from ydf.metric import metric
from ydf.model import analysis
from ydf.model import template_cpp_export
from yggdrasil_decision_forests.utils import model_analysis_pb2

# TODO: Allow a simpler input type (e.g. string)
Task = abstract_model_pb2.Task


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


class GenericModel:
  """Abstract superclass for all YDF models."""

  def __init__(self, raw_model: ydf.GenericCCModel):
    self._model = raw_model

  def name(self) -> str:
    """Returns the name of the model type."""
    return self._model.name()

  def task(self) -> Task:
    """Task solved by the model."""

    return self._model.task()

  def describe(self, full_details: bool = False) -> str:
    """Description of the model."""

    return self._model.Describe(full_details)

  def __str__(self) -> str:
    return f"""\
Model: {self.name()}
Task: {Task.Name(self.task())}
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
    vds = dataset.create_vertical_dataset(ds, data_spec=self._model.data_spec())
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

    YDF models can also be exported to other formats, see the methods under
    `export` for details.
    # TODO: Implement model exports and update this description.

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

    self._model.Save(path, advanced_options.file_prefix)

  def predict(self, data: dataset.InputDataset) -> np.ndarray:
    ds = dataset.create_vertical_dataset(
        data, data_spec=self._model.data_spec()
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
        Pandas DataFrame, or a VerticalDatset.
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

    ds = dataset.create_vertical_dataset(
        data, data_spec=self._model.data_spec()
    )

    if isinstance(bootstrapping, bool):
      bootstrapping_samples = 2000 if bootstrapping else -1
    elif isinstance(bootstrapping, int) and bootstrapping >= 100:
      bootstrapping_samples = bootstrapping
    else:
      raise ValueError(
          "bootstrapping argument should be boolean or an integer greater than"
          " 100 as bootstrapping will not yield useful results. Got"
          f" {bootstrapping!r} instead"
      )

    options_proto = metric_pb2.EvaluationOptions(
        bootstrapping_samples=bootstrapping_samples,
        task=self._model.task(),
    )
    evaluation_proto = self._model.Evaluate(ds._dataset, options_proto)  # pylint: disable=protected-access
    return metric.Evaluation(evaluation_proto)

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

    While some information might be valid, it is generatly not recommanded to
    analyze a model on its training dataset.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").Train(train_ds)

    test_ds = pd.read_csv("train.csv")
    analysis = model.analyze(test_ds)

    # Display the analysis in a notebook.
    analysis
    ```

    Args:
      data: Dataset. Can be a dictionary of list or numpy array of values,
        Pandas DataFrame, or a VerticalDatset.
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
    5. (Bonus) To further speed-up inference speed, you can pre-allocate and
      reuse the "examples" and "predictions" for each model running threads.

    This documentation is also available in the header of the generated content
    for more details.

    Args:
      key: Name of the model. Used to define the c++ namespace of the model.
    """
    return template_cpp_export.template(
        key, self._model.data_spec(), self._model.input_features()
    )

  def hyperparameter_optimizer_logs(
      self,
  ) -> Optional[abstract_model_pb2.HyperparametersOptimizerLogs]:
    """Returns the logs of the hyper-parameter tuning.

    If the model is not trained with hyper-parameter tuning, returns None.
    """

    return self._model.hyperparameter_optimizer_logs()


ModelType = TypeVar("ModelType", bound=GenericModel)
