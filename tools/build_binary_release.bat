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

:: Pack the CLI binary in an archive.

:: It is recommanded to use Bazelisk: https://github.com/bazelbuild/bazelisk/releases
set BAZEL=bazel.exe
%BAZEL% version

:: Bazel is compatible with Visual Studio 2017 and 2019.
:: https://bazel.build/configure/windows#using
:: If you have multiple or other version of VS (e.g., VS2022), set "BAZEL_VC"
:: accordingly. For example:
:: set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC

:: Disable support for TensorFlow datasets and IO.
copy /Y WORKSPACE_NO_TF WORKSPACE

:: Compile CLI binaries
set C=//yggdrasil_decision_forests/cli

%BAZEL% build --config=windows_cpp17 --config=windows_avx2^
 %C%:train^
 %C%:show_model^
 %C%:show_dataspec^
 %C%:predict^
 %C%:infer_dataspec^
 %C%:evaluate^
 %C%:edit_model^
 %C%:convert_dataset^
 %C%:benchmark_inference^
 %C%:analyze_model_and_dataset^
 %C%:compute_variable_importances^
 //yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_worker_main^
 || goto :error

:: Pack binaries in a zip
set CLI="bazel-bin\yggdrasil_decision_forests\cli"
mkdir dist
copy configure\cli_readme.txt dist\README

7z a -tzip dist\cli_windows.zip^
 .\dist\README^
 LICENSE^
 .\%CLI%\train.exe^
 .\%CLI%\show_model.exe^
 .\%CLI%\show_dataspec.exe^
 .\%CLI%\predict.exe^
 .\%CLI%\infer_dataspec.exe^
 .\%CLI%\evaluate.exe^
 .\%CLI%\edit_model.exe^
 .\%CLI%\convert_dataset.exe^
 .\%CLI%\benchmark_inference.exe^
 .\%CLI%\analyze_model_and_dataset.exe^
 .\%CLI%\compute_variable_importances.exe^
 .\%CLI%\..\utils\distribute\implementations\grpc\grpc_worker_main.exe

goto :EOF
:error
echo Failed with error #%errorlevel%.
exit /b %ERRORLEVEL%

