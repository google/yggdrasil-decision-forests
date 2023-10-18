from typing import Optional, TypeVar, List

# pylint: disable=g-wrong-blank-lines

import numpy as np
import numpy.typing as npt

from google3.third_party.yggdrasil_decision_forests.dataset import data_spec_pb2
from google3.third_party.yggdrasil_decision_forests.learner import abstract_learner_pb2
from google3.third_party.yggdrasil_decision_forests.metric import metric_pb2
from google3.third_party.yggdrasil_decision_forests.model import hyperparameter_pb2
from google3.third_party.yggdrasil_decision_forests.model import abstract_model_pb2
from google3.third_party.yggdrasil_decision_forests.utils import model_analysis_pb2
from google3.third_party.yggdrasil_decision_forests.utils import fold_generator_pb2

# Dataset bindings
# ================

class VerticalDataset:
  def data_spec(self) -> data_spec_pb2.DataSpecification: ...
  def MemoryUsage(self) -> int: ...
  def DebugString(self) -> str: ...
  def CreateColumnsFromDataSpec(
      self, data_spec: data_spec_pb2.DataSpecification
  ) -> None: ...
  def SetAndCheckNumRows(self, set_data_spec: bool) -> None: ...
  def PopulateColumnCategoricalNPBytes(
      self,
      name: str,
      data: npt.NDArray[np.bytes_],
      max_vocab_count: int = -1,
      min_vocab_frequency: int = -1,
      column_idx: Optional[int] = None,
  ) -> None: ...
  def PopulateColumnNumericalNPFloat32(
      self,
      name: str,
      data: npt.NDArray[np.float32],
      column_idx: Optional[int],
  ) -> None: ...
  def PopulateColumnBooleanNPBool(
      self,
      name: str,
      data: npt.NDArray[np.bool_],
      column_idx: Optional[int],
  ) -> None: ...


# Model bindings
# ================

class GenericCCModel:
  def Predict(
      self,
      dataset: VerticalDataset,
  ) -> npt.NDArray[np.float32]: ...
  def Evaluate(
      self,
      dataset: VerticalDataset,
      options: metric_pb2.EvaluationOptions,
  ) -> metric_pb2.EvaluationResults: ...
  def Analyze(
      self,
      dataset: VerticalDataset,
      options: model_analysis_pb2.Options,
  ) -> model_analysis_pb2.StandaloneAnalysisResult: ...
  def name(self) -> str: ...
  def task(self) -> abstract_model_pb2.Task: ...
  def data_spec(self) -> data_spec_pb2.DataSpecification: ...
  def Save(self, directory: str, file_prefix: Optional[str]): ...
  def Describe(self, full_details: bool) -> str: ...
  def input_features(self) -> List[int]: ...

class DecisionForestCCModel(GenericCCModel):
  def num_trees(self) -> int: ...

class RandomForestCCModel(DecisionForestCCModel):
  @property
  def kRegisteredName(self): ...

class GradientBoostedTreesCCModel(DecisionForestCCModel):
  @property
  def kRegisteredName(self): ...
  def validation_loss(self) -> float: ...

ModelCCType = TypeVar('ModelCCType', bound=GenericCCModel)

def LoadModel(directory: str, file_prefix: Optional[str]) -> ModelCCType: ...
def ModelAnalysisCreateHtmlReport(
    analysis: model_analysis_pb2.StandaloneAnalysisResult,
    options: model_analysis_pb2.Options,
) -> str: ...


# Learner bindings
# ================

class GenericCCLearner:
  def Train(self, dataset: VerticalDataset) -> ModelCCType: ...
  def Evaluate(
      self,
      dataset: VerticalDataset,
      fold_generator: fold_generator_pb2.FoldGenerator,
      evaluation_options: metric_pb2.EvaluationOptions,
      deployment_evaluation: abstract_learner_pb2.DeploymentConfig,
  ) -> metric_pb2.EvaluationResults: ...

def GetLearner(
    train_config: abstract_learner_pb2.TrainingConfig,
    hyperparameters: hyperparameter_pb2.GenericHyperParameters,
    deployment_config: abstract_learner_pb2.DeploymentConfig,
) -> GenericCCLearner: ...


# Metric bindings
# ================

def EvaluationToStr(evaluation: metric_pb2.EvaluationResults) -> str: ...
def EvaluationPlotToHtml(evaluation: metric_pb2.EvaluationResults) -> str: ...

