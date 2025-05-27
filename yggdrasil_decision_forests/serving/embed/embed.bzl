"""Blaze / Bazel rule to embed YDF models in a binary."""

def cc_ydf_embedded_model(name, data, path = None, **attrs):
    """Embed a YDF model into a CC library.

    Args:
        name: Name of the model. The library can be injected with the ":<name>" bazel/bazel rule,
            and with the "<name>.h" header.
        data: "filegroup" containing the model files.
        path: If not specified, "data" should be a directory containing the YDF model directly i.e.
            a directory containing a data_spec.pb file. If path is specified, it is the path to the
            model directory (a directory containing a data_spec.pb file) inside of the "data"
            filegroup.
        **attrs: Classical cc_library attributes.
    """

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
    native.genrule(
        name = name + "_write_embed",
        outs = [name + ".h"],
        cmd = "$(location write_embed) --input=$$(cat $(location " + name + "_create_path ) ) --output='$@' --remove_output_filename --name=" + name + " --alsologtostderr",
        tools = [":write_embed", name + "_create_path", data],
    )

    # Creates a cc library with the model.
    native.cc_library(
        name = name,
        srcs = [],
        hdrs = [name + ".h"],
        **attrs
    )
