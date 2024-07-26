## Naming convention

-   build: Compiling ydf.
-   test: Running unit tests.
-   package: Packing build artefacts for distribution e.g. create a pip package.
-   release: build + package all supported pip versions. Does not run tests.
-   *_in_docker: Execute something in a Docker.

## Building files

-   release_linux.sh: Create the pips for linux (using dockers).
-   release_macos.sh: Create the pips for macos.
-   release_windows.bat: Create the pips for windows.
-   collect_pip_files.py: Colleche pip files. Used by other scripts, including
    package_linux.sh.
-   package_linux.sh: Packaged already compiled (i.e. build) YDF into a pip
    package.

## Support files

-   local_export_and_test.sh: Export YDF to the home folder with Kokoro and run
    release_linux.sh.
-   change_version.sh: Change the YDF version in all the source / config files.
-   simple_test.py: Minimal standalone test of pydf.
-   build_test_linux.sh:
