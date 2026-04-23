# YDF-TF source

This folder contains the source code of the YDF-TF package.

> **YDF-TF** provides the TensorFlow ops required for interoperability between
> **YDF** and **TensorFlow**. It is necessary for exporting YDF models to
> TensorFlow and loading them. Users should install `ydf-tf` instead of
> `tensorflow-decision-forests` whenever it is available for their environment.

YDF-TF hosts the TensorFlow Custom Op for inference of YDF models converted to
TensorFlow. This op has previously been a part of TensorFlow Decision Forests
(TF-DF) and is now distributed as a standalone package.

The version of YDF-TF must always exactly match the version of TensorFlow to
avoid issues caused by binary incompatibility.

## Installation

Install from [PyPI](https://pypi.org/project/ydf-tf) with

```bash
pip install ydf-tf
```

## Building

Run `./build_pip_pkg.sh` (Linux) or `./build_pip_pkg_macos.sh` (macOS) to build
the package locally. See the content of these scripts for customization options.

## Versioning

YDF-TF does not follow semantic versioning. Instead, the YDF-TF version is
exactly pinned to the corresponding TensorFlow version. In rare cases,
post-releases might be used for critical bug / compilation fixes.

### Changelog

#### YDF-TF 2.21.0.post1

Restores support for all filesystems supported by TensorFlow (notably GCP).