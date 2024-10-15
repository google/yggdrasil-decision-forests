load("@emsdk//emscripten_toolchain:wasm_rules.bzl", "wasm_cc_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# exports_files(["training.js"])

# See https://github.com/emscripten-core/emscripten/blob/main/src/settings.js for the description
# of these options.
WASM_LINKOPTS = [
    "--bind",
    "-s USE_PTHREADS=0",
    "-s EXPORTED_RUNTIME_METHODS=FS",
    "-s ALLOW_MEMORY_GROWTH=1",
    "-s EXIT_RUNTIME=0",
    "-s MALLOC=emmalloc",
    "-s MODULARIZE=1",
    "-s DYNAMIC_EXECUTION=0",
    "-s EXPORT_NAME=YDFTraining",
    "-s FORCE_FILESYSTEM=1",
    "-s INCOMING_MODULE_JS_API=onRuntimeInitialized,fetchSettings,print,printErr,locateFile",
]

cc_binary(
    name = "training",
    srcs = ["training.cc"],
    additional_linker_inputs = [
    ],
    defines = [],
    linkopts = WASM_LINKOPTS + [
        "--post-js yggdrasil_decision_forests/port/javascript/training/training.bundle.js",
    ],
    tags = [
        "manual",
    ],
    deps = [
        "//yggdrasil_decision_forests/dataset:csv_example_reader",
        "//yggdrasil_decision_forests/dataset:csv_example_writer",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/learner:abstract_learner",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/learner/cart",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/learner/random_forest",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/port/javascript/training/dataset",
        "//yggdrasil_decision_forests/port/javascript/training/learner",
        "//yggdrasil_decision_forests/port/javascript/training/model",
        "//yggdrasil_decision_forests/port/javascript/training/util:status_casters",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

wasm_cc_binary(
    name = "training_wasm",
    cc_target = ":training",
    tags = ["manual"],
)

# Extract the emscriptten wasm file.
genrule(
    name = "extract_wasm_file",
    srcs = [":training_wasm"],
    outs = ["training.wasm"],
    cmd = "cp $(BINDIR)/yggdrasil_decision_forests/port/javascript/training/training_wasm/training.wasm $(@D)/",
    tags = ["manual"],
)

# Extract the merged emscriptten js + wrapper file.
genrule(
    name = "extract_js_file",
    srcs = [":training_wasm"],
    outs = ["training.js"],
    cmd = "cp $(BINDIR)/yggdrasil_decision_forests/port/javascript/training/training_wasm/training.js $(@D)/",
    tags = ["manual"],
)

# Zip the library.
genrule(
    name = "create_release",
    srcs = [
        ":extract_wasm_file",
        ":extract_js_file",
    ],
    outs = ["ydf.zip"],
    cmd = "zip -j $@ $(locations :extract_wasm_file) $(locations :extract_js_file) && " +
          "echo Zipfile information: && zipinfo $@ && " +
          "echo Zipfile ls: && ls -lh $@",
    tags = ["manual"],
)
