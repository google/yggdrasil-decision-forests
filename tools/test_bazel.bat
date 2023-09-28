:: Copyright 2022 Google LLC.
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

:: Compile and runs the unit tests.

:: It is recommended to use Bazelisk: https://github.com/bazelbuild/bazelisk/releases
set BAZEL=bazel.exe
::-4.0.0-windows-x86_64.exe
%BAZEL% version

set TF_SUPPORT=0

IF %TF_SUPPORT%==0 (

  copy /Y WORKSPACE_NO_TF WORKSPACE

  set FLAGS=--config=windows_cpp17
  %BAZEL% build %FLAGS% //yggdrasil_decision_forests/cli:all || goto :error

) ELSE (

  copy /Y WORKSPACE_WITH_TF WORKSPACE

  # Without TensorFlow IO.
  set FLAGS=--config=windows_cpp17
  %BAZEL% build %FLAGS% //yggdrasil_decision_forests/cli/...:all || goto :error
  %BAZEL% test %FLAGS% //yggdrasil_decision_forests/{cli,metric,model,serving,utils}/...:all //examples:beginner_cc || goto :error

  # With TensorFlow IO.
  set FLAGS=--config=windows_cpp14 --config=use_tensorflow_io
  %BAZEL% build %FLAGS% //yggdrasil_decision_forests/cli/...:all || goto :error
  %BAZEL% test %FLAGS% //yggdrasil_decision_forests/...: //examples:beginner_cc || goto :error

)

goto :EOF
:error
echo Failed with error #%errorlevel%.
exit /b %ERRORLEVEL%
