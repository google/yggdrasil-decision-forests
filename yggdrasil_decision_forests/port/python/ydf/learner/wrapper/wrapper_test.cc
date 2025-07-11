/*
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/hyperparameter.pb.h"
#include "ydf/learner/wrapper/wrapper_generator.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace {

using ::testing::HasSubstr;

class FakeLearner1 : public model::AbstractLearner {
 public:
  explicit FakeLearner1(const model::proto::TrainingConfig& training_config)
      : AbstractLearner(training_config) {}

  absl::StatusOr<std::unique_ptr<model::AbstractModel>> TrainWithStatusImpl(
      const dataset::VerticalDataset& train_dataset,
      std::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset = {}) const override {
    return std::unique_ptr<model::AbstractModel>();
  }

  absl::StatusOr<model::proto::GenericHyperParameterSpecification>
  GetGenericHyperParameterSpecification() const override {
    model::proto::GenericHyperParameterSpecification spec;
    auto& a = (*spec.mutable_fields())["a"];
    a.mutable_real()->set_minimum(1);
    a.mutable_real()->set_minimum(2);
    a.mutable_real()->set_default_value(1);
    a.mutable_documentation()->set_description("Documentation for a");
    a.mutable_documentation()->set_proto_field("a_proto");

    auto& b = (*spec.mutable_fields())["b"];
    b.mutable_real()->set_minimum(3);
    b.mutable_real()->set_minimum(5);
    b.mutable_real()->set_default_value(4);
    b.mutable_documentation()->set_description("Documentation for b");
    b.mutable_documentation()->set_proto_field("b_proto");
    b.mutable_mutual_exclusive()->mutable_other_parameters()->Add("c");
    b.mutable_mutual_exclusive()->set_is_default(true);

    auto& c = (*spec.mutable_fields())["c"];
    c.mutable_real()->set_minimum(6);
    c.mutable_real()->set_minimum(8);
    c.mutable_real()->set_default_value(7);
    c.mutable_documentation()->set_description("Documentation for c");
    c.mutable_documentation()->set_proto_field("c_proto");
    c.mutable_mutual_exclusive()->mutable_other_parameters()->Add("b");
    return spec;
  }

  std::vector<model::proto::PredefinedHyperParameterTemplate>
  PredefinedHyperParameters() const override {
    model::proto::PredefinedHyperParameterTemplate hptemplate;
    hptemplate.set_name("fake_template_1");
    hptemplate.set_version(4);
    hptemplate.set_description("This is a fake template.");
    auto* field = hptemplate.mutable_parameters()->add_fields();
    field->set_name("a");
    field->mutable_value()->set_real(2);

    model::proto::PredefinedHyperParameterTemplate hptemplate2;
    hptemplate2.set_name("fake_template_2");
    hptemplate2.set_version(1);
    hptemplate2.set_description("This is another fake template.");
    field = hptemplate2.mutable_parameters()->add_fields();
    field->set_name("a");
    field->mutable_value()->set_real(3);
    return {hptemplate, hptemplate2};
  };

  model::proto::LearnerCapabilities Capabilities() const override {
    model::proto::LearnerCapabilities capabilities;
    capabilities.set_require_label(false);
    capabilities.set_resume_training(true);
    return capabilities;
  }
};

TEST(LearnerWrappers, LearnerKeyToClassName) {
  EXPECT_EQ(internal::LearnerKeyToClassName("RANDOM_FOREST"),
            "RandomForestLearner");
}

TEST(LearnerWrappers, Base) {
  auto learner_config = internal::LearnerConfigs()["FAKE_ALGORITHM"];
  ASSERT_OK_AND_ASSIGN(
      const auto content,
      internal::GenSingleLearnerWrapper("FAKE_ALGORITHM", learner_config));
  LOG(INFO) << "content:\n" << content;

  EXPECT_EQ(content, R"(
class FakeAlgorithmLearner(generic_learner.GenericCCLearner):
  r"""Fake Algorithm learning algorithm.

  

  Usage example:

  ```python
  import ydf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")

  model = ydf.FakeAlgorithmLearner().train(dataset)

  print(model.describe())
  ```

  Hyperparameters are configured to give reasonable results for typical
  datasets. Hyperparameters can also be modified manually (see descriptions)
  below or by applying the hyperparameter templates available with
  `FakeAlgorithmLearner.hyperparameter_templates()` (see this function's documentation for
  details).

  Attributes:
    label: Label of the dataset. The label column
      should not be identified as a feature in the `features` parameter.
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING, Task.CATEGORICAL_UPLIFT, Task.NUMERICAL_UPLIFT).
    weights: Name of a feature that identifies the weight of each example. If
      weights are not specified, unit weights are assumed. The weight column
      should not be identified as a feature in the `features` parameter.
    class_weights: Dictionary of class weights in the form
      `{class_label: weight}`. The dictionary must specify a weight for each
      label value in the training data. All weights must be non-negative 
      floating-point numbers. It is not possible to specify both `weights`
      (sample weights) and `class_weights`.
      If None (default), all classes have weight 1.
    ranking_group: Only for `task=Task.RANKING`. Name of a feature
      that identifies queries in a query/document ranking task. The ranking
      group should not be identified as a feature in the `features` parameter.
    uplift_treatment: Only for `task=Task.CATEGORICAL_UPLIFT` and `task=Task`.
      NUMERICAL_UPLIFT. Name of a numerical feature that identifies the
      treatment in an uplift problem. The value 0 is reserved for the control
      treatment. Currently, only 0/1 binary treatments are supported.
    features: If None, all columns are used as features. The semantic of the
      features is determined automatically. Otherwise, if
      include_all_columns=False (default) only the column listed in `features`
      are imported. If include_all_columns=True, all the columns are imported as
      features and only the semantic of the columns NOT in `columns` is 
      determined automatically. If specified,  defines the order of the features
      - any non-listed features are appended in-order after the specified
      features (if include_all_columns=True).
      The label, weights, uplift treatment and ranking_group columns should not
      be specified as features.
    include_all_columns: See `features`.
    max_vocab_count: Maximum size of the vocabulary of CATEGORICAL and
      CATEGORICAL_SET columns stored as strings. If more unique values exist,
      only the most frequent values are kept, and the remaining values are
      considered as out-of-vocabulary.
    min_vocab_frequency: Minimum number of occurrence of a value for CATEGORICAL
      and CATEGORICAL_SET columns. Value observed less than
      `min_vocab_frequency` are considered as out-of-vocabulary.
    discretize_numerical_columns: If true, discretize all the numerical columns
      before training. Discretized numerical columns are faster to train with,
      but they can have a negative impact on the model quality. Using
      `discretize_numerical_columns=True` is equivalent as setting the column
      semantic DISCRETIZED_NUMERICAL in the `column` argument. See the
      definition of DISCRETIZED_NUMERICAL for more details.
    num_discretized_numerical_bins: Number of bins used when disretizing
      numerical columns.
    max_num_scanned_rows_to_infer_semantic: Number of rows to scan when
      inferring the column's semantic if it is not explicitly specified. Only
      used when reading from file, in-memory datasets are always read in full.
      Setting this to a lower number will speed up dataset reading, but might
      result in incorrect column semantics. Set to -1 to scan the entire
      dataset.
    max_num_scanned_rows_to_compute_statistics: Number of rows to scan when
      computing a column's statistics. Only used when reading from file,
      in-memory datasets are always read in full. A column's statistics include
      the dictionary for categorical features and the mean / min / max for
      numerical features. Setting this to a lower number will speed up dataset
      reading, but skew statistics in the dataspec, which can hurt model quality
      (e.g. if an important category of a categorical feature is considered
      OOV). Set to -1 to scan the entire dataset.
    data_spec: Dataspec to be used (advanced). If a data spec is given,
      `columns`, `include_all_columns`, `max_vocab_count`,
      `min_vocab_frequency`, `discretize_numerical_columns` and 
      `num_discretized_numerical_bins` will be ignored.
    extra_training_config: Training configuration proto (advanced). If set, this
      training configuration proto is merged with the one implicitely defined
      by the learner. Can be used to set internal or advanced parameters that
      are not exposed as constructor arguments. Parameters in
      extra_training_config have higher priority as the constructor arguments.
    a: Documentation for a Default: 1.0.
    b: Documentation for b Default: 4.0.
    c: Documentation for c Default: None.

    resume_training: If true, the model training resumes from the checkpoint
      stored in the `working_dir` directory. If `working_dir` does not
      contain any model checkpoint, the training starts from the beginning.
      Resuming training is useful in the following situations: (1) The training
      was interrupted by the user (e.g. ctrl+c or "stop" button in a notebook)
      or rescheduled, or (2) the hyper-parameter of the learner was changed e.g.
      increasing the number of trees.
    resume_training_snapshot_interval_seconds: Indicative number of seconds in 
      between snapshots when `resume_training=True`. Might be ignored by
      some learners.
    working_dir: Path to a directory available for the learning algorithm to
      store intermediate computation results. Depending on the learning
      algorithm and parameters, the working_dir might be optional, required, or
      ignored. For instance, distributed training algorithm always need a
      "working_dir", and the gradient boosted tree and hyper-parameter tuners
      will export artefacts to the "working_dir" if provided.
    num_threads: Number of threads used to train the model. Different learning
      algorithms use multi-threading differently and with different degree of
      efficiency. If `None`, `num_threads` will be automatically set to the
      number of processors (up to a maximum of 256; or set to 6 if the number of
      processors cannot be determined). Making `num_threads` significantly
      larger than the number of processors can slow-down the training speed. The
      default value logic might change in the future.
    tuner: If set, automatically select the best hyperparameters using the
      provided tuner. When using distributed training, the tuning is
      distributed.
    feature_selector: If set, automatically select the input features of the
      model using automated feature selection using the specified feature
      selector.
    explicit_args: Helper argument for internal use. Throws if supplied
      explicitly by the user.
  """

  @func_helpers.list_explicit_arguments
  def __init__(self,

      label: Optional[str] = None,
      task: generic_learner.Task = generic_learner.Task.CLASSIFICATION,
      *,
      weights: Optional[str] = None,
      class_weights: Optional[Dict[str, float]] = None,
      ranking_group: Optional[str] = None,
      uplift_treatment: Optional[str] = None,
      features: Optional[dataspec.ColumnDefs] = None,
      include_all_columns: bool = False,
      max_vocab_count: int = 2000,
      min_vocab_frequency: int = 5,
      discretize_numerical_columns: bool = False,
      num_discretized_numerical_bins: int = 255,
      max_num_scanned_rows_to_infer_semantic: int = 100_000,
      max_num_scanned_rows_to_compute_statistics: int = 100_000,
      data_spec: Optional[data_spec_pb2.DataSpecification] = None,
      extra_training_config: Optional[abstract_learner_pb2.TrainingConfig] = None,
      a: float = 1.0,
      b: Optional[float] = 4.0,
      c: Optional[float] = None,
      resume_training: bool = False,
      resume_training_snapshot_interval_seconds: int = 1800,
      working_dir: Optional[str] = None,
      num_threads: Optional[int] = None,
      tuner: Optional[tuner_lib.AbstractTuner] = None,
      feature_selector: Optional[
          abstract_feature_selector_lib.AbstractFeatureSelector
      ] = None,
      explicit_args: Optional[Set[str]] = None,
      ):

    hyper_parameters = {
                      "a" : a,
                      "b" : b,
                      "c" : c,

      }
    if explicit_args is None:
      raise ValueError("`explicit_args` must not be set by the user")

    data_spec_args = dataspec.DataSpecInferenceArgs(
        columns=dataspec.normalize_column_defs(features),
        include_all_columns=include_all_columns,
        max_vocab_count=max_vocab_count,
        min_vocab_frequency=min_vocab_frequency,
        discretize_numerical_columns=discretize_numerical_columns,
        num_discretized_numerical_bins=num_discretized_numerical_bins,
        max_num_scanned_rows_to_infer_semantic=max_num_scanned_rows_to_infer_semantic,
        max_num_scanned_rows_to_compute_statistics=max_num_scanned_rows_to_compute_statistics,
    )

    deployment_config = self._build_deployment_config(
        num_threads=num_threads,
        working_dir=working_dir,
        resume_training=resume_training,
        resume_training_snapshot_interval_seconds=resume_training_snapshot_interval_seconds,

    )

    super().__init__(learner_name="FAKE_ALGORITHM",
      task=task,
      label=label,
      weights=weights,
      class_weights=class_weights,
      ranking_group=ranking_group,
      uplift_treatment=uplift_treatment,
      data_spec_args=data_spec_args,
      data_spec=data_spec,
      hyper_parameters=hyper_parameters,
      explicit_learner_arguments=explicit_args,
      deployment_config=deployment_config,
      tuner=tuner,
      feature_selector=feature_selector,
      extra_training_config=extra_training_config,
    )

  def train(
      self,
      ds: dataset.InputDataset,
      valid: Optional[dataset.InputDataset] = None,
      verbose: Optional[Union[int, bool]] = None,
  ) -> generic_model.GenericModel:
    """Trains a model on the given dataset.

    Options for dataset reading are given on the learner. Consult the
    documentation of the learner or ydf.create_vertical_dataset() for additional
    information on dataset reading in YDF.

    Usage example:

    ```
    import ydf
    import pandas as pd

    train_ds = pd.read_csv(...)

    learner = ydf.FakeAlgorithmLearner(label="label")
    model = learner.train(train_ds)
    print(model.describe())
    ```

    If training is interrupted (for example, by interrupting the cell execution
    in Colab), the model will be returned to the state it was in at the moment
    of interruption.

    Args:
      ds: Training dataset.
      valid: Optional validation dataset. Some learners, such as Random Forest,
        do not need validation dataset. Some learners, such as
        GradientBoostedTrees, automatically extract a validation dataset from
        the training dataset if the validation dataset is not provided.
      verbose: Verbose level during training. If None, uses the global verbose
        level of `ydf.verbose`. Levels are: 0 of False: No logs, 1 or True:
        Print a few logs in a notebook; prints all the logs in a terminal. 2:
        Prints all the logs on all surfaces.

    Returns:
      A trained model.
    """
    return super().train(ds=ds, valid=valid, verbose=verbose)

  @classmethod
  def _capabilities(cls) -> abstract_learner_pb2.LearnerCapabilities:
    return abstract_learner_pb2.LearnerCapabilities(
      support_max_training_duration=False,
      resume_training=True,
      support_validation_dataset=False,
      support_partial_cache_dataset_format=False,
      support_max_model_size_in_memory=False,
      support_monotonic_constraints=False,
      require_label=False,
      support_custom_loss=False,
    )

  @classmethod
  def hyperparameter_templates(cls) -> Dict[str, hyperparameters.HyperparameterTemplate]:
    r"""Hyperparameter templates for this Learner.
    
    Hyperparameter templates are sets of pre-defined hyperparameters for easy
    access to different variants of the learner. Each template is a mapping to a
    set of hyperparameters and can be applied directly on the learner.
    
    Usage example:
    ```python
    templates = ydf.FakeAlgorithmLearner.hyperparameter_templates()
    fake_template_1v4 = templates["fake_template_1v4"]
    # Print a description of the template
    print(fake_template_1v4.description)
    # Apply the template's settings on the learner.
    learner = ydf.FakeAlgorithmLearner(label, **fake_template_1v4)
    ```
    
    Returns:
      Dictionary of the available templates
    """
    return {"fake_template_1v4": hyperparameters.HyperparameterTemplate(name="fake_template_1", version=4, description="This is a fake template.", parameters={"a" :2.0}), "fake_template_2v1": hyperparameters.HyperparameterTemplate(name="fake_template_2", version=1, description="This is another fake template.", parameters={"a" :3.0}), }
)");
}

TEST(LearnerWrappers, FormatDocumentation) {
  const auto formatted =
      internal::FormatDocumentation(R"(AAA AAA AAA AAA AAA.
AAA AAA AAA AAA.
- AAA AAA AAA AAA.
- AAA AAA AAA AAA.
AAA AAA AAA AAA.
  AAA AAA AAA AAA.)",
                                    /*leading_spaces_first_line=*/4,
                                    /*leading_spaces_next_lines=*/6,
                                    /*max_char_per_lines=*/20);
  EXPECT_EQ(formatted, R"(    AAA AAA AAA AAA
      AAA.
      AAA AAA AAA
      AAA.
      - AAA AAA AAA
        AAA.
      - AAA AAA AAA
        AAA.
      AAA AAA AAA
      AAA.
          AAA AAA
        AAA AAA.
)");
}

TEST(LearnerWrappers, NumLeadingSpaces) {
  EXPECT_EQ(internal::NumLeadingSpaces(""), 0);
  EXPECT_EQ(internal::NumLeadingSpaces(" "), 1);
  EXPECT_EQ(internal::NumLeadingSpaces("  "), 2);
  EXPECT_EQ(internal::NumLeadingSpaces("  HELLO "), 2);
}

}  // namespace

namespace model {
namespace {
REGISTER_AbstractLearner(FakeLearner1, "FAKE_ALGORITHM");
}
}  // namespace model
}  // namespace yggdrasil_decision_forests
