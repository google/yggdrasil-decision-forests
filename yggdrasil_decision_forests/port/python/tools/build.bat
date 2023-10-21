:: Build the Windows version of PYDF
:: 
:: For each python versions
::   - Build PYDF
::   - Run PYDF tests (non blocking)
::   - Build a PYDF pip package (exported to the "dist" directory).
::   - Run a simple example with the pip package
::
:: Usage example:
::   tools\build.bat
::   :: The output pip packages are put in "dist".
::
:: Requirements:
::   - MSYS2
::   - Python versions installed in "C:\Python<version>" e.g. C:\Python310.
::   - Bazel
::   - Visual Studio (tested with VS2022).
::
cls
setlocal

set YDF_VERSION=0.0.3
set BAZEL=bazel.exe
set BAZEL_SH=C:\msys64\usr\bin\bash.exe
set BAZEL_FLAGS=--config=windows_cpp20 --config=windows_avx2
%BAZEL% version

CALL :End2End 39 || goto :error
CALL :End2End 310 || goto :error
CALL :End2End 311 || goto :error

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
CALL :Compile %PYTHON% || goto :error
%PYTHON% tools/assembly_pip_files.py || goto :error
CALL :BuildPipPackage %PYTHON% || goto :error
mkdir dist
copy tmp_package\dist\ydf-%YDF_VERSION%-cp%PYTHON_VERSION%-cp%PYTHON_VERSION%-win_amd64.whl dist || goto :error
CALL :TestPipPackage dist\ydf-%YDF_VERSION%-cp%PYTHON_VERSION%-cp%PYTHON_VERSION%-win_amd64.whl %PYTHON% || goto :error
EXIT /B 0

:: Compiles and runs the tests.
:Compile
set PYTHON=%~1
%PYTHON% -m pip install -r requirements.txt || goto :error
%PYTHON% -m pip install -r dev_requirements.txt || goto :error
%BAZEL% build %BAZEL_FLAGS% -- //ydf/...:all || goto :error
:: Non blocking tests
:: TODO: Figure how to get pybind11 + bazel + window to work with the ".pyd" trick.
%BAZEL% test %BAZEL_FLAGS% --test_output=errors -- //ydf/...:all
EXIT /B 0

:: Builds a pip package
:BuildPipPackage
set PYTHON=%~1
%PYTHON% -m ensurepip -U || goto :error
%PYTHON% -m pip install pip -U || goto :error
%PYTHON% -m pip install setuptools -U || goto :error
%PYTHON% -m pip install build -U || goto :error
%PYTHON% -m pip install virtualenv -U || goto :error
%PYTHON% -m pip install auditwheel==5.2.0 || goto :error
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
