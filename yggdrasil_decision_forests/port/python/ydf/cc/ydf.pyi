from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt

from google3.third_party.yggdrasil_decision_forests.dataset import data_spec_pb2
from google3.third_party.yggdrasil_decision_forests.learner import abstract_learner_pb2
from google3.third_party.yggdrasil_decision_forests.metric import metric_pb2
from google3.third_party.yggdrasil_decision_forests.model import abstract_model_pb2
from google3.third_party.yggdrasil_decision_forests.model import hyperparameter_pb2
from google3.third_party.yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2
from google3.third_party.yggdrasil_decision_forests.model.gradient_boosted_trees import gradient_boosted_trees_pb2
from google3.third_party.yggdrasil_decision_forests.model.isolation_forest import isolation_forest_pb2
from google3.third_party.yggdrasil_decision_forests.model.random_forest import random_forest_pb2
from google3.third_party.yggdrasil_decision_forests.serving.embed import embed_pb2
from google3.third_party.yggdrasil_decision_forests.utils import fold_generator_pb2
from google3.third_party.yggdrasil_decision_forests.utils import fold_generator_pb2
from google3.third_party.yggdrasil_decision_forests.utils import model_analysis_pb2

# pylint: disable=g-wrong-blank-lines

# Dataset bindings
# ================

class VerticalDataset:
  def data_spec(self) -> data_spec_pb2.DataSpecification: ...
  def nrow(self) -> int: ...
  def MemoryUsage(self) -> int: ...
  def DebugString(self) -> str: ...
  def CreateColumnsFromDataSpec(
      self, data_spec: data_spec_pb2.DataSpecification
  ) -> None: ...
  def SetAndCheckNumRowsAndFillMissing(self, set_data_spec: bool) -> None: ...
  def PopulateColumnCategoricalNPBytes(
      self,
      name: str,
      data: npt.NDArray[np.bytes_],
      ydf_dtype: Optional[data_spec_pb2.DType],
      max_vocab_count: int = -1,
      min_vocab_frequency: int = -1,
      column_idx: Optional[int] = None,
      dictionary: Optional[npt.NDArray[np.bytes_]] = None,
  ) -> None: ...
  def PopulateColumnCategoricalSetNPBytes(
      self,
      name: str,
      data: npt.NDArray[np.object_],
      ydf_dtype: Optional[data_spec_pb2.DType],
      max_vocab_count: int = -1,
      min_vocab_frequency: int = -1,
      column_idx: Optional[int] = None,
      dictionary: Optional[npt.NDArray[np.bytes_]] = None,
  ) -> None: ...
  def PopulateColumnNumericalNPFloat32(
      self,
      name: str,
      data: npt.NDArray[np.float32],
      ydf_dtype: Optional[data_spec_pb2.DType],
      column_idx: Optional[int],
  ) -> None: ...
  def PopulateColumnDiscretizedNumericalNPFloat32(
      self,
      name: str,
      data: npt.NDArray[np.float32],
      ydf_dtype: Optional[data_spec_pb2.DType],
      maximum_num_bins: Optional[int],
      column_idx: Optional[int],
  ) -> None: ...
  def PopulateColumnBooleanNPBool(
      self,
      name: str,
      data: npt.NDArray[np.bool_],
      ydf_dtype: Optional[data_spec_pb2.DType],
      column_idx: Optional[int],
  ) -> None: ...
  def PopulateColumnHashNPBytes(
      self,
      name: str,
      data: npt.NDArray[np.bytes_],
      ydf_dtype: Optional[data_spec_pb2.DType],
      column_idx: Optional[int],
  ) -> None: ...
  def PopulateColumnNumericalVectorSequence(
      self,
      name: str,
      data: List[npt.NDArray],
      ydf_dtype: Optional[data_spec_pb2.DType],
      column_idx: Optional[int],
  ) -> None: ...
  def CreateFromPathWithDataSpec(
      self,
      path: str,
      data_spec: data_spec_pb2.DataSpecification,
      required_columns: Optional[Sequence[str]] = None,
  ) -> None: ...
  def CreateFromPathWithDataSpecGuide(
      self,
      path: str,
      data_spec_guide: data_spec_pb2.DataSpecificationGuide,
      required_columns: Optional[Sequence[str]] = None,
  ) -> None: ...
  def SetMultiDimDataspec(
      self,
      unrolling: Dict[str, List[str]],
  ) -> None: ...

def DataspecToTextReport(dataset: data_spec_pb2.DataSpecification) -> str: ...

# Model bindings
# ================

class BenchmarkInferenceCCResult:
  """Results of the inference benchmark.

  Attributes:
      duration_per_example: Average duration per example in seconds.
      benchmark_duration: Total duration of the benchmark run without warmup
        runs in seconds.
      num_runs: Number of times the benchmark fully ran over all the examples of
        the dataset. Warmup runs are not included.
      batch_size: Number of examples per batch used when benchmarking.
  """

  duration_per_example: float
  duration_per_example_multithread: float
  benchmark_duration: float
  num_runs: int
  batch_size: int
  num_threads: int

class GBTCCTrainingLogEntry:
  """Training logs for a GBT model.

  Attributes:
      iteration: The iteration number for this log entry.
      validation_evaluation: Evaluation proto for the validation dataset.
      training_evaluation: Evaluation proto for the training dataset.
  """

  iteration: int
  validation_evaluation: metric_pb2.EvaluationResults
  training_evaluation: metric_pb2.EvaluationResults

class GenericCCModel:
  def Predict(
      self,
      dataset: VerticalDataset,
      use_slow_engine: bool,
      num_threads: int,
  ) -> npt.NDArray[np.float32]: ...
  def PredictShap(
      self,
      dataset: VerticalDataset,
      num_threads: int,
  ) -> Tuple[Dict[str, npt.NDArray[np.float32]], npt.NDArray[np.float32]]: ...
  def Evaluate(
      self,
      dataset: VerticalDataset,
      options: metric_pb2.EvaluationOptions,
      weighted: bool,
      label_col_idx: int,
      group_col_idx: int,
      use_slow_engine: bool,
      num_threads: int,
  ) -> metric_pb2.EvaluationResults: ...
  def Analyze(
      self,
      dataset: VerticalDataset,
      options: model_analysis_pb2.Options,
  ) -> model_analysis_pb2.StandaloneAnalysisResult: ...
  def AnalyzePrediction(
      self,
      example: VerticalDataset,
      options: model_analysis_pb2.PredictionAnalysisOptions,
  ) -> model_analysis_pb2.PredictionAnalysisResult: ...
  def name(self) -> str: ...
  def task(self) -> abstract_model_pb2.Task: ...
  def data_spec(self) -> data_spec_pb2.DataSpecification: ...
  def set_data_spec(
      self, data_spec: data_spec_pb2.DataSpecification
  ) -> None: ...
  def metadata(self) -> abstract_model_pb2.Metadata: ...
  def set_metadata(self, metadata: abstract_model_pb2.Metadata) -> None: ...
  def label_col_idx(self) -> int: ...
  def group_col_idx(self) -> int: ...
  def weight_col_idx(self) -> Optional[int]: ...
  def Save(
      self, directory: str, file_prefix: Optional[str], pure_serving: bool
  ): ...
  def Serialize(self) -> bytes: ...
  def Describe(self, full_details: bool, text_format: bool) -> str: ...
  def input_features(self) -> List[int]: ...
  def hyperparameter_optimizer_logs(
      self,
  ) -> Optional[abstract_model_pb2.HyperparametersOptimizerLogs]: ...
  def Benchmark(
      self,
      dataset: VerticalDataset,
      benchmark_duration: float,
      warmup_duration: float,
      batch_size: int,
      num_threads: int,
  ) -> BenchmarkInferenceCCResult: ...
  def VariableImportances(
      self,
  ) -> Dict[str, abstract_model_pb2.VariableImportanceSet]: ...
  def ForceEngine(self, engine_name: Optional[str]) -> None: ...
  def ListCompatibleEngines(self) -> Sequence[str]: ...
  # TODO: Remove when solved.
  def weighted_training(self) -> bool: ...
  def set_feature_selection_logs(
      self, value: abstract_model_pb2.FeatureSelectionLogs
  ) -> None: ...
  def feature_selection_logs(
      self,
  ) -> abstract_model_pb2.FeatureSelectionLogs: ...
  def EmbedModel(self, options: embed_pb2.Options) -> Dict[str, str]: ...

class DecisionForestCCModel(GenericCCModel):
  def num_trees(self) -> int: ...
  def num_nodes(self) -> int: ...
  def PredictLeaves(
      self,
      dataset: VerticalDataset,
  ) -> npt.NDArray[np.int32]: ...
  def Distance(
      self,
      dataset1: VerticalDataset,
      dataset2: VerticalDataset,
  ) -> npt.NDArray[np.float32]: ...
  def set_node_format(self, node_format: str) -> None: ...
  def GetTree(
      self,
      tree_idx: int,
  ) -> List[decision_tree_pb2.Node]: ...
  def SetTree(
      self,
      tree_idx: int,
      nodes: Sequence[decision_tree_pb2.Node],
  ) -> None: ...
  def AddTree(
      self,
      nodes: Sequence[decision_tree_pb2.Node],
  ) -> None: ...
  def RemoveTree(
      self,
      tree_idx: int,
  ) -> None: ...

class RandomForestCCModel(DecisionForestCCModel):
  @property
  def kRegisteredName(self): ...
  def out_of_bag_evaluations(
      self,
  ) -> List[random_forest_pb2.OutOfBagTrainingEvaluations]: ...
  def winner_takes_all(self) -> bool: ...

class IsolationForestCCModel(DecisionForestCCModel):
  def set_num_examples_per_tree(self, num_examples: int): ...
  @property
  def kRegisteredName(self): ...
  def num_examples_per_tree(self) -> int: ...

class GradientBoostedTreesCCModel(DecisionForestCCModel):
  @property
  def kRegisteredName(self): ...
  def validation_loss(self) -> float: ...
  def validation_evaluation(self) -> metric_pb2.EvaluationResults: ...
  def initial_predictions(self) -> npt.NDArray[float]: ...
  def set_initial_predictions(self, values: npt.NDArray[float]): ...
  def num_trees_per_iter(self) -> int: ...
  def loss(self) -> gradient_boosted_trees_pb2.Loss: ...
  def training_logs(self) -> List[GBTCCTrainingLogEntry]: ...

ModelCCType = TypeVar('ModelCCType', bound=GenericCCModel)

def LoadModel(directory: str, file_prefix: Optional[str]) -> ModelCCType: ...
def DeserializeModel(data: bytes) -> ModelCCType: ...
def ModelAnalysisCreateHtmlReport(
    analysis: model_analysis_pb2.StandaloneAnalysisResult,
    options: model_analysis_pb2.Options,
) -> str: ...
def PredictionAnalysisCreateHtmlReport(
    analysis: model_analysis_pb2.PredictionAnalysisResult,
    options: model_analysis_pb2.PredictionAnalysisOptions,
) -> str: ...

# Learner bindings
# ================

class CCRegressionLoss:
  def __init__(
      self,
      initial_predictions: Callable[
          [npt.NDArray[np.float32], npt.NDArray[np.float32]],
          np.float32,
      ],
      loss: Callable[
          [
              npt.NDArray[np.float32],
              npt.NDArray[np.float32],
              npt.NDArray[np.float32],
          ],
          np.float32,
      ],
      gradient_and_hessian: Callable[
          [npt.NDArray[np.float32], npt.NDArray[np.float32]],
          Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
      ],
      may_trigger_gc: bool,
  ): ...

class CCBinaryClassificationLoss:
  def __init__(
      self,
      initial_predictions: Callable[
          [npt.NDArray[np.int32], npt.NDArray[np.float32]],
          np.float32,
      ],
      loss: Callable[
          [
              npt.NDArray[np.int32],
              npt.NDArray[np.float32],
              npt.NDArray[np.float32],
          ],
          np.float32,
      ],
      gradient_and_hessian: Callable[
          [npt.NDArray[np.int32], npt.NDArray[np.float32]],
          Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
      ],
      may_trigger_gc: bool,
  ): ...

class CCMultiClassificationLoss:
  def __init__(
      self,
      initial_predictions: Callable[
          [npt.NDArray[np.int32], npt.NDArray[np.float32]],
          npt.NDArray[np.float32],
      ],
      loss: Callable[
          [
              npt.NDArray[np.int32],
              npt.NDArray[np.float32],
              npt.NDArray[np.float32],
          ],
          np.float32,
      ],
      gradient_and_hessian: Callable[
          [npt.NDArray[np.int32], npt.NDArray[np.float32]],
          Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
      ],
      may_trigger_gc: bool,
  ): ...

class GenericCCLearner:
  def Train(
      self,
      dataset: VerticalDataset,
      validation_dataset: Optional[VerticalDataset] = None,
  ) -> ModelCCType: ...
  def TrainFromPathWithDataSpec(
      self,
      dataset_path: str,
      data_spec: data_spec_pb2.DataSpecification,
      validation_dataset_path: str = None,
  ) -> ModelCCType: ...
  def TrainFromPathWithGuide(
      self,
      dataset_path: str,
      data_spec_guide: data_spec_pb2.DataSpecificationGuide,
      validation_dataset_path: str = None,
  ) -> ModelCCType: ...
  def Evaluate(
      self,
      dataset: Union[VerticalDataset, str],
      fold_generator: fold_generator_pb2.FoldGenerator,
      evaluation_options: metric_pb2.EvaluationOptions,
      deployment_evaluation: abstract_learner_pb2.DeploymentConfig,
  ) -> metric_pb2.EvaluationResults: ...

def GetLearner(
    train_config: abstract_learner_pb2.TrainingConfig,
    extra_training_config: Optional[abstract_learner_pb2.TrainingConfig],
    hyperparameters: hyperparameter_pb2.GenericHyperParameters,
    deployment_config: abstract_learner_pb2.DeploymentConfig,
    custom_loss: Optional[CCRegressionLoss],
) -> GenericCCLearner: ...
def GetInvalidHyperparameters(
    hp_names: Set[str],
    explicit_hps: Set[str],
    train_config: abstract_learner_pb2.TrainingConfig,
    deployment_config: abstract_learner_pb2.DeploymentConfig,
) -> Set[str]: ...
def ValidateHyperparameters(
    hyperparameters: Set[str],
    train_config: abstract_learner_pb2.TrainingConfig,
    deployment_config: abstract_learner_pb2.DeploymentConfig,
) -> None: ...

# Metric bindings
# ================

def EvaluationToStr(evaluation: metric_pb2.EvaluationResults) -> str: ...
def EvaluationPlotToHtml(evaluation: metric_pb2.EvaluationResults) -> str: ...
def EvaluatePredictions(
    predictions: npt.NDArray[np.float32],
    labels: Union[npt.NDArray[np.float32], npt.NDArray[np.int32]],
    options: metric_pb2.EvaluationOptions,
    weights: npt.NDArray[np.float32],
    label_classes: Optional[list[str]],
    ranking_groups: npt.NDArray[np.uint64],
    random_seed: int,
) -> metric_pb2.EvaluationResults: ...

# Log bindings
# ================

def SetLoggingLevel(level: int, print_file: bool) -> None: ...

# Worker bindings
# ================

def StartWorkerBlocking(port: int) -> None: ...
def StartWorkerNonBlocking(port: int) -> int: ...
def StopWorkerNonBlocking(uid: int) -> None: ...
