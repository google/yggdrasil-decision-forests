"""Blaze / Bazel rule to embed YDF models in a binary."""

def cc_ydf_standalone_model(
        name,
        data,
        path = None,
        classification_output = "CLASS",
        algorithm = "ROUTING",
        monitor_usage = False,
        categorical_from_string = False,
        **attrs):
    """Embed a YDF model into a CC library.

    Args:
        name: Name of the model. The library can be injected with the ":<name>" bazel/bazel rule,
            and with the "<name>.h" header.
        data: "filegroup" containing the model files.
        path: If not specified, "data" should be a directory containing the YDF model directly i.e.
            a directory containing a data_spec.pb file. If path is specified, it is the path to the
            model directory (a directory containing a data_spec.pb file) inside of the "data"
            filegroup.
        classification_output: What is the Predict function is returning in case of a classification
            model. Can be one of: CLASS, SCORE or PROBABILITY. See "embed.proto" for details
            about those values. Has no impact on non-classification models.
        algorithm: How the predictions are computed. One of the values of
            "yggdrasil_decision_forests.serving.embed.proto.Algorithm".
        monitor_usage: If set, monitor the model usage. This creates a dependency to
            yggdrasil_decision_forests/utils/usage.h. If you can, leave this flag to true as it
            helps the YDF team to understand our library usage.
        categorical_from_string: If true, generates functions to create categorical feature values
            from strings. For example, for a categorical feature "X" with an associated "FeatureX"
            enum class, the method "FeatureXFromString(absl::string_view name) -> FeatureX" is
            created.
        **attrs: Classical cc_library attributes.
    """

    if classification_output not in ["CLASS", "SCORE", "PROBABILITY"]:
        fail("Invalid classification_output value: %s. Possible values are: CLASS, SCORE, or PROBABILITY." % classification_output)

    if path == None:
        # Determine the path from the data.
        # Informally this genrule does:
        #   - List the model files.
        #   - Search for a "data_spec.pb" file path.
        #   - Extract the dirname from the "data_spec.pb" file.
        #   - Save tghe results in a <name>_path.txt file.
        native.genrule(
            name = name + "_create_path",
            outs = [name + "_path.txt"],
            cmd = "echo \"$(locations " + data + ")\" | tr ' ' '\n' | grep data_spec.pb$$ | sed 's|data_spec.pb||' > \"$@\"",
            tools = [data],
        )
    else:
        native.genrule(
            name = name + "_create_path",
            outs = [name + "_path.txt"],
            cmd = "echo \"" + path + "\" > \"$@\"",
        )

    # Convert the model into source files.

    options = "classification_output: " + classification_output + " algorithm: " + algorithm + " monitor_usage: " + str(monitor_usage) + " categorical_from_string: " + str(categorical_from_string) + " cc: {}"

    native.genrule(
        name = name + "_write_embed",
        outs = [name + ".h"],
        cmd = "$(location //yggdrasil_decision_forests/serving/embed:write_embed) --input=$$(cat $(location " + name + "_create_path ) ) --options='" + options + "' --output='$@' --remove_output_filename --name=" + name + " --alsologtostderr",
        tools = ["//yggdrasil_decision_forests/serving/embed:write_embed", name + "_create_path", data],
    )

    # Creates a cc library with the model.
    native.cc_library(
        name = name,
        srcs = [],
        hdrs = [name + ".h"],
        **attrs
    )
