"""Wrapper rule generation utilities."""

load("@ydf_cc//yggdrasil_decision_forests/utils:compile.bzl", "cc_binary_ydf")

def py_wrap_yggdrasil_learners(
        name = None,
        learner_deps = []):
    """Creates Python wrappers around Yggdrasil Decision Forest (YDF) learners.

    Creates a Python library called "{name}" and containing the file "{name}.py".
    This library creates a PYDF class wrapping each YDF learner defined in
    "learner_deps". The constructor of these classes contains an argument for
    this learner's generic hyper-parameters.

    For example, if "learner_deps" contains a C++ dependency that registers a
    learner with a key equal to "RANDOM_FOREST", the wrapper will create a
    Python class called "RandomForestLearner" deriving from the generic PYDF
    learner class.

    Args:
        name: Name of the rule.
        learner_deps: List of dependencies linking Yggdrasil Decision Forest
          learners.
    """

    # Absolute path to the wrapper generator directory.
    wrapper_package = "//ydf/learner/wrapper"

    # Filename of the wrapper generator source code in the user package.
    local_cc_main = name + "_generate_wrapper.cc"

    # Target name of the wrapper generator binary.
    wrapper_name = name + "_generator"

    # Target name of the command running the wrapper generator.
    run_wrapper_name = name + "_run_wrapper"

    # Copy the wrapper main source code to the user package.
    native.genrule(
        name = name + "_copy_cc_main",
        outs = [local_cc_main],
        srcs = [wrapper_package + ":generate_wrapper.cc"],
        cmd = "cp $< $@",
    )

    # Compiles the wrapper binary.
    cc_binary_ydf(
        name = wrapper_name,
        srcs = [":" + local_cc_main],
        deps = [
            wrapper_package + ":wrapper_generator",
            "@ydf_cc//yggdrasil_decision_forests/utils:logging",
        ] + learner_deps,
        linkstatic = 1,
    )

    # Runs the wrapper binary and generate the wrapper .py source code.
    native.genrule(
        name = run_wrapper_name,
        srcs = [],
        outs = [name + ".py"],
        cmd = "$(location " + wrapper_name + ") > \"$@\"",
        tools = [":" + wrapper_name],
    )

    # Python library around the generated .py source code.
    native.py_library(
        name = name,
        srcs = [name + ".py"],
        deps = [
            "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
            "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
            "//ydf/dataset:dataspec",
            "//ydf/learner:generic_learner",
            "//ydf/learner:hyperparameters",
            "//ydf/learner:tuner",
        ],
        data = [":" + run_wrapper_name, ":" + wrapper_name],
    )
