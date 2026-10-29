/* Linker version script of the YDF Python extension (ydf.so) on Linux.

   Only the module init function is exported. YDF statically links its
   dependencies (e.g. Abseil, Protobuf). Exporting their symbols would let them
   interpose (or be interposed by) other copies of these libraries in the
   process, e.g. the ones of TensorFlow, if ydf.so is ever loaded with
   RTLD_GLOBAL. */
{
  global:
    PyInit_ydf;
  local:
    *;
};
