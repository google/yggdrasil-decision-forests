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

:: Build the Windows version of PYDF
::
::: For each Python version
::   - Build PYDF
::   - Run PYDF tests (non blocking)
::   - Build a PYDF pip package (exported to the "dist" directory).
::   - Run a simple example with the pip package
::
:: Usage example:
::   :: run tools\release_windows.bat
::   :: The output pip packages are put in "dist".
::
:: Requirements:
::   - MSYS2
::   - Python versions installed in "C:\Python<version>" e.g. C:\Python310.
::     Only used to build and test the pip package: Bazel compiles YDF with
::     its hermetic Python toolchain of the same version (see MODULE.bazel).
::   - Bazel
::   - Visual Studio (tested with VS2022).
::
cls
setlocal


set BAZEL=bazel.exe
set BAZEL_SH=C:\msys64\usr\bin\bash.exe
set BAZEL_FLAGS=--config=windows_cpp20 --config=windows_avx2
set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC
%BAZEL% version

CALL :End2End 310 || goto :error
CALL :End2End 311 || goto :error
CALL :End2End 312 || goto :error
CALL :End2End 313 || goto :error
CALL :End2End 314 || goto :error

:: In case of error
goto :EOF
:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%

:: Runs the full build+test+pip for a specific version of python.
:End2End
set PYTHON_VERSION=%~1
set PYTHON_DIR=C:/Python%PYTHON_VERSION%
set PYTHON=%PYTHON_DIR%/python.exe
set PYTHON3_BIN_PATH=%PYTHON%
set PYTHON3_LIB_PATH=%PYTHON_DIR%/Lib
:: e.g. 310 -> 3.10
set PYTHON_DOT_VERSION=%PYTHON_VERSION:~0,1%.%PYTHON_VERSION:~1%
CALL :Compile %PYTHON_DOT_VERSION% || goto :error
%PYTHON% tools/collect_pip_files.py || goto :error
CALL :BuildPipPackage %PYTHON% || goto :error
mkdir dist
:: Resolve the wheel by glob so the version never needs to be hard-coded here.
set WHEEL=
for %%F in (tmp_package\dist\ydf-*-cp%PYTHON_VERSION%-cp%PYTHON_VERSION%-win_amd64.whl) do set WHEEL=%%~nxF
if "%WHEEL%"=="" (
  echo Could not find a built wheel for cp%PYTHON_VERSION% in tmp_package\dist
  goto :error
)
copy tmp_package\dist\%WHEEL% dist || goto :error
CALL :TestPipPackage dist\%WHEEL% %PYTHON% || goto :error
EXIT /B 0

:: Compiles and runs the tests with the hermetic Python toolchain of the given
:: version (e.g. 3.10).
:Compile
set BAZEL_PY_FLAGS=--@rules_python//python/config_settings:python_version=%~1
%BAZEL% build %BAZEL_FLAGS% %BAZEL_PY_FLAGS% -- //ydf/...:all || goto :error
:: Non blocking tests
:: TODO: Figure how to get pybind11 + bazel + window to work with the ".pyd" trick.
%BAZEL% test %BAZEL_FLAGS% %BAZEL_PY_FLAGS% --test_output=errors -- //ydf/...:all
EXIT /B 0

:: Builds a pip package
:BuildPipPackage
set PYTHON=%~1
%PYTHON% -m ensurepip -U || goto :error
%PYTHON% -m pip install pip -U || goto :error
%PYTHON% -m pip install setuptools -U || goto :error
%PYTHON% -m pip install build -U || goto :error
%PYTHON% -m pip install virtualenv -U || goto :error
cd tmp_package
%PYTHON% -m build || goto :error
cd ..
EXIT /B 0

:: Tests a pip package.
:TestPipPackage
set PACKAGE=%~1
set PYTHON=%~2
%PYTHON% -m pip install -r requirements.txt || goto :error
%PYTHON% -m pip uninstall ydf -y || goto :error
%PYTHON% -m pip install %PACKAGE% || goto :error
%PYTHON% tools/simple_test.py || goto :error
%PYTHON% -m pip uninstall ydf -y || goto :error
EXIT /B 0
