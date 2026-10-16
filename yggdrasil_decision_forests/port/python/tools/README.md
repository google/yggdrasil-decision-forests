## Naming convention

-   build: Compiling ydf.
-   test: Running unit tests.
-   package: Packing build artefacts for distribution e.g. create a pip package.
-   release: build + package all supported pip versions. Does not run tests.
-   *_in_docker: Execute something in a Docker.

## Building files

-   release_linux.sh: Create the pips for linux.
-   release_macos.sh: Create the pips for macos.
-   release_windows.bat: Create the pips for windows.
-   collect_pip_files.py: Collect pip files. Used by release_windows.bat (with
    config/setup.py).
-   package_linux.sh: Build and test the pip package with Bazel (`//:wheel`
    and `//tools:wheel_test`), and copy it to dist/.
-   wheel.bzl, wheel_config.bzl: Bazel rules for the pip package.
-   wheel_test.py: Test of the pip package built by Bazel.

## Support files

-   local_export_and_test.sh: Export YDF to the home folder with Kokoro and run
    release_linux.sh.
-   change_version.sh: Change the YDF version in all the source / config files.
-   simple_test.py: Minimal standalone test of pydf.
-   build_test_linux.sh: ompile and runs the unit tests for the Python port of YDF
