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

cd %KOKORO_ARTIFACTS_DIR%\git\yggdrasil_decision_forests

:: Output artefact directory
mkdir output

:: Install gcc
choco install -y mingw --no-progress
choco install -y wget --no-progress

:: Install Python 3.8
choco install -y python --version=3.8.3 --no-progress
SET BAZEL_PYTHON=C:\Python38
SET PYTHON_BIN_PATH=C:\Python38\python.exe
SET PATH=C:\Python38;%PATH%
SET PIP_EXE=C:\Python38\Scripts\pip.exe
%PIP_EXE% install numpy --upgrade

:: Setup Bazel
wget https://github.com/bazelbuild/bazel/releases/download/4.0.0/bazel-4.0.0-windows-x86_64.exe

set BAZEL=bazel-4.0.0-windows-x86_64.exe
set FLAGS=--test_output=streamed
set FLAGS_WO_TF=--config=windows_cpp17
:: We actually only use cpp14 functionalities.
set FLAGS_W_TF=--config=windows_cpp17 --config=use_tensorflow_io


:: %BAZEL% build %FLAGS_WO_TF% //yggdrasil_decision_forests/cli/...:all || goto :error

%BAZEL% build %FLAGS_W_TF% //yggdrasil_decision_forests/cli/...:all || goto :error

%BAZEL% version

:: Build
setlocal enabledelayedexpansion
for %%x in (
cli
dataset
learner
metric
model
serving
  ) do (
  %BAZEL% build %FLAGS% %FLAGS_W_TF% //yggdrasil_decision_forests/%%x/...:all || goto :error
  )

:: Test
setlocal enabledelayedexpansion
for %%x in (
cli
dataset
learner
metric
model
serving
utils
  ) do (
  %BAZEL% test %FLAGS% %FLAGS_W_TF% --test_output=all //yggdrasil_decision_forests/%%x/...:all || goto :error
  )

SET errorflag=0
goto :finalize

:error
echo Failed with error #%errorlevel%.
SET errorflag=1

:finalize

xcopy /s %KOKORO_ARTIFACTS_DIR%\git\yggdrasil_decision_forests\bazel-testlogs\*.log %KOKORO_ARTIFACTS_DIR%\git\yggdrasil_decision_forests\output

:: Windows build is broken. VS studio version in Kokoro is too old.
:: exit /b %errorflag%
