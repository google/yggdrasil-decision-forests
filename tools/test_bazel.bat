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
%BAZEL% version

:: TF_SUPPORT=1 add support for TF.Record of TF.Example datasets and relies on
:: the TensorFlow filesystem. If you don't need this dataset format,
:: TF_SUPPORT=0 is recommended.
set TF_SUPPORT=0

:: Bazel is compatible with Visual Studio 2017 and 2019.
:: https://bazel.build/configure/windows#using
:: If you have multiple or other version of VS (e.g., VS2022), set "BAZEL_VC"
:: accordingly. For example:
:: set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC

IF %TF_SUPPORT%==0 (

  copy /Y WORKSPACE_NO_TF WORKSPACE

  :: Compile without any TensorFlow support.
  set FLAGS=--config=windows_cpp17
  %BAZEL% build %FLAGS% //yggdrasil_decision_forests/cli:all || goto :error

) ELSE (

  copy /Y WORKSPACE_WITH_TF WORKSPACE

  :: Support TensorFlow datasets but does not use TensorFlow filesystem.
  set FLAGS=--config=windows_cpp17
  %BAZEL% build %FLAGS% //yggdrasil_decision_forests/cli/...:all || goto :error
  %BAZEL% test %FLAGS% //yggdrasil_decision_forests/{cli,metric,model,serving,utils}/...:all //examples:beginner_cc || goto :error

  :: Support TensorFlow datasets and use TensorFlow filesystem.
  set FLAGS=--config=windows_cpp14 --config=use_tensorflow_io
  %BAZEL% build %FLAGS% //yggdrasil_decision_forests/cli/...:all || goto :error
  %BAZEL% test %FLAGS% //yggdrasil_decision_forests/...: //examples:beginner_cc || goto :error

)

goto :EOF
:error
echo Failed with error #%errorlevel%.
exit /b %ERRORLEVEL%
