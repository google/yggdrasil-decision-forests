load("@emsdk//emscripten_toolchain:wasm_rules.bzl", "wasm_cc_binary")

package(
    default_visibility = ["//yggdrasil_decision_forests/port/javascript:users"],
    licenses = ["notice"],
)

package_group(
    name = "users",
    packages = [
        "//learning/lib/ami/simple_ml/ports/javascript/...",
        "//yggdrasil_decision_forests/port/javascript/...",
    ],
)

exports_files(["wrapper.js"])

# Change the extension of the wrapper js file. This is necessary for Emscripten.
genrule(
    name = "wrapper",
    srcs = ["wrapper.js"],
    outs = ["wrapper.lds"],
    cmd = "cat $(SRCS) > $@",
    tags = ["manual"],
    visibility = ["//visibility:private"],
)

# Web assembly logic (part 1).
#
# See https://github.com/emscripten-core/emscripten/blob/main/src/settings.js for the description
# of the linkops.
cc_binary(
    name = "inference",
    srcs = ["inference.cc"],
    defines = [],
    linkopts = [
        "--bind",
        "-s USE_PTHREADS=0",
        "-s EXPORTED_RUNTIME_METHODS=FS",  # To access YDF output file from JS.
        "-s ALLOW_MEMORY_GROWTH=1",
        "-s EXIT_RUNTIME=0",
        "-s MALLOC=emmalloc",
        "-s MODULARIZE=1",
        "-s DYNAMIC_EXECUTION=0",
        "-s EXPORT_NAME=YggdrasilDecisionForests",
        "-s FILESYSTEM=1",  # Link filesystem (should be automatic in some cases).
        # "-s -g",  # Function names in stack trace.
        # "-s ASSERTIONS=2",  # Runtime checks for common memory allocation errors.
        "-s DEMANGLE_SUPPORT=1",  # Better function name in stack stace.
        # fetchSettings is included to bypass CORS issues during development
        "-s INCOMING_MODULE_JS_API=onRuntimeInitialized,fetchSettings,print,printErr,locateFile",
        "--post-js yggdrasil_decision_forests/port/javascript/wrapper.js",
    ],
    tags = [
        "manual",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/learner/cart",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/learner/random_forest",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)

# Web assembly logic (part 2).
wasm_cc_binary(
    name = "inference_wasm",
    cc_target = ":inference",
    tags = [
        "manual",
    ],
)

# Extract the emscriptten wasm file.
genrule(
    name = "extract_wasm_file",
    srcs = [":inference_wasm"],
    outs = ["inference.wasm"],
    cmd = "cp $(BINDIR)/yggdrasil_decision_forests/port/javascript/inference_wasm/inference.wasm $(@D)/",
    tags = ["manual"],
)

# Extract the merged emscriptten js + wrapper file.
genrule(
    name = "extract_js_file",
    srcs = [":inference_wasm"],
    outs = ["inference.js"],
    cmd = "cp $(BINDIR)/yggdrasil_decision_forests/port/javascript/inference_wasm/inference.js $(@D)/",
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
