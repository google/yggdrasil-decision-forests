## Naming convention

-   build: Compiling ydf.
-   test: Running unit tests.
-   package: Packing build artefacts for distribution e.g. create a pip package.
-   release: build + package all supported pip versions. Only runs the unit
    tests if requested (RUN_TESTS=1), but always tests the pip packages.

## Building files

-   release.sh: Create the pips for the platform of the host (Linux or macOS).

## Support files

-   local_copybara_export.sh: Export YDF to the home folder with Copybara and
    run release.sh.
-   change_version.sh: Change the YDF version in all the source / config files.
-   build_test_linux.sh: Compile and runs PYDF unit tests.
