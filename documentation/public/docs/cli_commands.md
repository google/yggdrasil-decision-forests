# CLI Commands

This page list the commands for the CLI (command line interface) of Yggdrasil
Decision Forests.

```
train: Train a ML model and export it to disk.

  Flags from cli/train.cc:
    --config (Path to the training configuration i.e. a
      model::proto::TrainingConfig text proto.); default: "";
    --dataset (Typed path to training dataset i.e. [type]:[path] format. Support
      glob, shard and comma. Example: csv:/my/dataset.csv); default: "";
    --dataspec (Path to the dataset specification (dataspec). Note: The dataspec
      is often created with :infer_dataspec and inspected with :show_dataspec.);
      default: "";
    --deployment (Path to the deployment configuration for the training i.e.
      what computing resources to use to train the model. Text proto buffer of
      type model::proto::DeploymentConfig. If not specified, the training is
      done locally with a number of threads chosen by the training algorithm.);
      default: "";
    --output (Output model directory.); default: "";
    --valid_dataset (Optional validation dataset specified with [type]:[path]
      format. If not specified and if the learning algorithm uses a validation
      dataset, the effective validation dataset is extracted from the training
      dataset.); default: "";

Try --helpfull to get a list of all flags or --help=substring shows help for
flags which include specified substring in either in the name, or description or
path.


show_model: Display the statistics and structure of a model.

  Flags from cli/show_model.cc:
    --dataspec (Show the dataspec contained in the model. This is similar as
      running :show_dataspec on the data_spec.pb file in the model directory.);
      default: false;
    --engines (List and test the fast engines compatible with the model. Note:
      Engines needs to be linked to the binary. Some engines depend on the
      platform e.g. if you don't have AVX2, AVX2 engines won't be listed.);
      default: false;
    --explain_engine_incompatibility (If true, and if --engines=true, print an
      explanation of why each of the available serving engine is not compatible
      with the model.); default: false;
    --full_definition (Show the full details of the model. For decision forest
      models, show the tree structure.); default: false;
    --model (Model directory.); default: "";

Try --helpfull to get a list of all flags or --help=substring shows help for
flags which include specified substring in either in the name, or description or
path.


show_dataspec: Print a human readable representation of a dataspec.

  Flags from cli/show_dataspec.cc:
    --dataspec (Path to dataset specification (dataspec).); default: "";
    --is_text_proto (If true, the dataset is read as a text proto. If false, the
      dataspec is read as a binary proto.); default: true;
    --sort_by_column_names (If true, sort the columns by names. If false, sort
      the columns by column index.); default: true;

Try --helpfull to get a list of all flags or --help=substring shows help for
flags which include specified substring in either in the name, or description or
path.


predict: Apply a model on a dataset and export the predictions to disk.

  Flags from cli/predict.cc:
    --dataset (Typed path to dataset i.e. [type]:[path] format.); default: "";
    --key (If set, copies the column "key" in the output prediction file. This
      key column cannot be an input feature of the model.); default: "";
    --model (Model directory.); default: "";
    --num_records_by_shard_in_output (Number of records per output shards. Only
      valid if the output path is sharded (e.g. contains @10).); default: -1;
    --output (Output prediction specified with [type]:[path] format. e.g.
      "csv:/path/to/dataset.csv".); default: "";

Try --helpfull to get a list of all flags or --help=substring shows help for
flags which include specified substring in either in the name, or description or
path.


infer_dataspec: Infers the dataspec of a dataset i.e. the name, type and meta-data of the dataset columns.

  Flags from cli/infer_dataspec.cc:
    --dataset (Typed path to training dataset i.e. [type]:[path] format.);
      default: "";
    --guide (Path to an optional dataset specification guide
      (DataSpecificationGuide Text proto). Use to override the automatic type
      detection of the columns.); default: "";
    --output (Output dataspec path.); default: "";

Try --helpfull to get a list of all flags or --help=substring shows help for
flags which include specified substring in either in the name, or description or
path.


evaluate: Evaluates a model.

  Flags from cli/evaluate.cc:
    --dataset (Typed path to dataset i.e. [type]:[path] format.); default: "";
    --model (Model directory.); default: "";
    --options (Path to optional evaluation configuration.
      proto::EvaluationOptions Text proto.); default: "";

Try --helpfull to get a list of all flags or --help=substring shows help for
flags which include specified substring in either in the name, or description or
path.


convert_dataset: Converts a dataset from one format to another. The dataspec of the dataset should be available.

  Flags from cli/convert_dataset.cc:
    --dataspec (Input data specification path. This file is generally created
      with :infer_dataspec and inspected with :show_dataspec.); default: "";
    --dataspec_is_binary (If true, the dataspec is a binary proto. If false
      (default), the dataspec is a text proto. The :infer_dataspec cli generates
      text proto dataspec, while the dataspec contained in a model is encoded as
      a binary proto.); default: false;
    --ignore_missing_columns (If false (default), fails if one of the column in
      the dataspec is missing. If true, fill missing columns with "missing
      values".); default: false;
    --input (Input dataset specified with [type]:[path] format.); default: "";
    --output (Output dataset specified with [type]:[path] format.); default: "";
    --shard_size (Number of record per output shards. Only valid if the output
      path is sharded (e.g. contains @10). This flag is required as this
      conversion is greedy. If num_records_by_shard is too low, all the
      remaining examples will be put in the last shard.); default: -1;

Try --helpfull to get a list of all flags or --help=substring shows help for
flags which include specified substring in either in the name, or description or
path.


benchmark_inference: Benchmarks the inference time of a model with the available inference engines.

  Flags from cli/benchmark_inference.cc:
    --batch_size (Number of examples per batch. Note that some engine are not
      impactedby the batch size.); default: 100;
    --dataset (Typed path to dataset i.e. [type]:[path] format.); default: "";
    --generic (Evaluates the slow engine i.e. model->predict(). The generic
      engine is slow and mostly a reference. Disable it if the benchmark runs
      for too long.); default: true;
    --model (Path to model.); default: "";
    --num_runs (Number of times the dataset is run. Higher values increase the
      precision of the timings, but increase the duration of the benchmark.);
      default: 20;
    --warmup_runs (Number of runs through the dataset before the benchmark.);
      default: 1;

Try --helpfull to get a list of all flags or --help=substring shows help for
flags which include specified substring in either in the name, or description or
path.


edit_model: Edits a trained model.

  Flags from cli/edit_model.cc:
    --input (Input model directory.); default: "__NO__SET__";
    --new_file_prefix (New prefix in the filenames.); default: "__NO__SET__";
    --new_label_name (New label name.); default: "__NO__SET__";
    --new_weights_name (New weights name.); default: "__NO__SET__";
    --output (Output model directory.); default: "__NO__SET__";
    --pure_serving (Clear the model from any information that is not required
      for model serving.This includes debugging, model interpretation and other
      meta-data. Can reduce significantly the size of the model.);
      default: "__NO__SET__";

Try --helpfull to get a list of all flags or --help=substring shows help for
flags which include specified substring in either in the name, or description or
path.


synthetic_dataset: Create a synthetic dataset.

  Flags from cli/utils/synthetic_dataset.cc:
    --options (Optional path to text serialized
      proto::SyntheticDatasetOptions.); default: "";
    --ratio_test (Fraction of the dataset (which size is defined in "options")
      is send to the test dataset. The "test" flag can be empty iff.
      ratio_valid=0.); default: 0.3;
    --ratio_valid (Fraction of the dataset (which size is defined in "options")
      is send to the validation dataset. The "valid" flag can be empty iff.
      ratio_valid=0.); default: 0;
    --test (Optional [type]:[path] path to the output test dataset.);
      default: "";
    --train ([type]:[path] path to the output training dataset.); default: "";
    --valid (Optional [type]:[path] path to the output validation dataset.);
      default: "";

Try --helpfull to get a list of all flags or --help=substring shows help for
flags which include specified substring in either in the name, or description or
path.
```
