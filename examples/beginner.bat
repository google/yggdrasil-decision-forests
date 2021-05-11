:: Copyright 2021 Google LLC.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     https://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.

:: Example showing the training, evaluation and inference benchmark of a model.
:: It will create a directory %userprofile%\yggdrasil_decision_forests_beginner
:: with the trained models.

IF EXIST infer_dataspec.exe (
    ECHO "Using Yggdrasil Decision Forest from the PATH"
    SET CLI=.
    SET DATASET_DIR=yggdrasil_decision_forests\test_data\dataset
) ELSE (
    IF exist bazel-bin\yggdrasil_decision_forests\cli\infer_dataspec.exe (
        ECHO "Using Yggdrasil Decision Forest from the root directory."
        SET CLI=bazel-bin\yggdrasil_decision_forests\cli
        SET DATASET_DIR=yggdrasil_decision_forests\test_data\dataset
    ) ELSE (
        ECHO "Yggdrasil Decision Forest was not found. Compile it and add the"
        ECHO "bazel-bin\yggdrasil_decision_forests\cli directory to the PATH, or run this"
        ECHO "command from the Yggdrasil's example directory (i.e. the directory containing"
        ECHO "this file)."
        GOTO :eof
    )
)

:: Datasets
SET TRAIN_DS=csv:%DATASET_DIR%\adult_train.csv
SET TEST_DS=csv:%DATASET_DIR%\adult_test.csv

:: Directory containing the artifacts of the project.
SET PROJECT=%userprofile%\yggdrasil_decision_forests_beginner
mkdir %PROJECT%

:: Generate the dataspec for the training dataspec.
SET DATASPEC=%PROJECT%\dataspec.pbtxt
%CLI%\infer_dataspec --dataset=%TRAIN_DS% --output=%DATASPEC%


:: Human description of the dataspec.
SET DATASPEC_INFO=%PROJECT%\dataspec.txt
%CLI%\show_dataspec --dataspec=%DATASPEC% --alsologtostderr > %DATASPEC_INFO%
type %DATASPEC_INFO%

:: Create a training configuration i.e. the hyper-parameters.
SET TRAINING_CONFIG=%PROJECT%\train_config.pbtxt
(
echo task: CLASSIFICATION
echo label: "income"
echo learner: "GRADIENT_BOOSTED_TREES"
echo # Change learner specific hyper-parameters.
echo [yggdrasil_decision_forests.model.gradient_boosted_trees.proto.gradient_boosted_trees_config] {
echo   num_trees: 200
echo }
) > %TRAINING_CONFIG%

:: Train the model.
:: Note that there are not validation dataset. Some learners might extract a
:: validation data from the training dataset.
SET MODEL=%PROJECT%\model
%CLI%\train ^
  --dataset=%TRAIN_DS% ^
  --dataspec=%DATASPEC% ^
  --config=%TRAINING_CONFIG% ^
  --output=%MODEL% ^
  --alsologtostderr

:: Display information about the model.
SET MODEL_INFO=%PROJECT%\model\description.txt
%CLI%\show_model --model=%MODEL% --engines --alsologtostderr > %MODEL_INFO%
type %MODEL_INFO%

:: Evaluate the model on the test dataset.
SET EVALUATION=%PROJECT%\evaluation.txt
%CLI%\evaluate --dataset=%TEST_DS% --model=%MODEL% --alsologtostderr > %EVALUATION%
type %EVALUATION%

:: Export the predictions of the model.
SET PREDICTIONS=csv:%PROJECT%\prediction_test.csv
%CLI%\predict --dataset=%TEST_DS% --model=%MODEL% --output=%PREDICTIONS% --alsologtostderr

:: Benchmark the inference speed of the model.
SET BENCHMARK=%PROJECT%\benchmark.txt
%CLI%\benchmark_inference --dataset=%TEST_DS% --model=%MODEL% --alsologtostderr > %BENCHMARK%
type %BENCHMARK%

:: Show the content of the working directory.
dir %PROJECT%
