"""Utilities to build the YDF pip package (wheel) with Bazel."""

def _path_inside_wheel(f):
    """Path of a file inside the wheel.

    Same logic as `py_package` and `py_wheel` of rules_python: Files of external
    repositories (short path "../<repo>/<path>") are placed at "<path>".

    Args:
      f: A File.

    Returns:
      The path of the file in the wheel.
    """
    short_path = f.short_path
    if short_path.startswith("../"):
        short_path = short_path[3:]
        short_path = short_path[short_path.find("/") + 1:]
    return short_path

def _in_packages(path, packages):
    for package in packages:
        if path == package or path.startswith(package + "/"):
            return True
    return False

def _py_package_init_files_impl(ctx):
    packages = [p.replace(".", "/") for p in ctx.attr.packages]
    files = depset(transitive = [dep[DefaultInfo].files for dep in ctx.attr.deps])

    paths = {}
    for f in files.to_list():
        paths[_path_inside_wheel(f)] = True

    directories = {}
    for path in paths:
        parts = path.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            directories["/".join(parts[:i])] = True

    init_files = []
    for directory in sorted(directories):
        if not _in_packages(directory, packages):
            continue
        init_path = directory + "/__init__.py"
        if init_path in paths:
            continue
        init_file = ctx.actions.declare_file(ctx.label.name + "/" + init_path)
        ctx.actions.write(init_file, "")
        init_files.append(init_file)

    return [DefaultInfo(files = depset(init_files))]

py_package_init_files = rule(
    implementation = _py_package_init_files_impl,
    doc = """Creates the missing `__init__.py` files of a `py_package`.

Bazel (with `--incompatible_default_to_explicit_init_py`) and `py_package` only
package the `__init__.py` files listed in the `srcs` of a `py_library`. This
rule creates an empty `__init__.py` file in each directory of `packages` that
does not already contain one.

The files are created in the "<name>/" sub-directory of the package of the
rule. Strip this prefix in `py_wheel` with `strip_path_prefixes`.
""",
    attrs = {
        "deps": attr.label_list(
            doc = "The `py_package` targets.",
            mandatory = True,
        ),
        "packages": attr.string_list(
            doc = "Python packages in which `__init__.py` files are created.",
            mandatory = True,
        ),
    },
)
