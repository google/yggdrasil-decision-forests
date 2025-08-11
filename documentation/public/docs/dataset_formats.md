# Supported dataset formats

This list details the supported dataset formats within YDF and their respective
advantages and limitations. YDF offers two primary methods for inputting
datasets:

-   **Python Dataset Objects:** These include familiar structures like Pandas
    DataFrames or dictionaries of NumPy arrays.

-   **Typed-Paths as a string:** Examples include `"csv:/dataset/train.csv"` for
    CSV files or `"tfrecord:/dataset/train@*"` for TensorFlow Records.

**Python Dataset Objects**

Python dataset objects offer significant flexibility, allowing you to easily
apply various preprocessing operations directly to your data. However, they can
be less efficient for very large datasets because the entire dataset must be
loaded into memory. As a general guideline, Python dataset objects are best
suited for datasets containing fewer than 100 million examples.

We recommend using a **dictionary of NumPy arrays** for Python datasets, though
**Pandas DataFrames** are also well-supported.

Note: For internal users. Loading large datasets directly into a Pandas
DataFrame using the %%f1 magic query can be unacceptably slow for anything more
than a few thousand examples. Instead, we recommend exporting your data first
(e.g., using PLX, optionally with pre-filtering or preprocessing). Once
exported, load your data efficiently using the `capacitor:` format.

**Typed-Paths**

In contrast, typed-paths are far more memory-efficient and are required for
distributed training. Their main drawback is reduced flexibility; preprocessing
needs materializing the output.

Currently, Avro files are the recommended format for typed-path datasets.
TensorFlow Records are also well-supported.

**Available formats**

Format                              | Availability | As python object               | Typed-path prefix | Remarks
----------------------------------- | ------------ | ------------------------------ | ----------------- | -------
Dict of numpy array                 | Public       | Native                         |                   | Efficient; Recommended for small datasets
Pandas dataframe                    | Public       | Native                         |                   | Popular format
csv                                 | Public       |                                | csv:              | Popular format; No support for multi-dimentionnal features.
Avro                                | Public       |                                | avro:             | Efficient; Recommended for large datasets
TensorFlow Records (gzip)           | Public       | with ydf.util.read_tf_recordio | tfrecord:         | Somehow efficient
TensorFlow Records (non-compressed) | Public       | with ydf.util.read_tf_recordio | tfrecordv2:       | Inefficient; To avoid
TensorFlow Tensor                   | Public       | Native                         |                   | Inefficient; To avoid
TensorFlow Dataset                  | Public       | Native                         |                   | Inefficient; To avoid
Xarray                              | Public       | Native                         |                   |
SSTable of TF Examples              | Internal     |                                | sstable+tfe:      |
RecordIO of TF Examples             | Internal     |                                | recordio+tfe:     | Efficient; Recommended for large datasets
RecordIO of YDF Examples            | Internal     |                                | recordio+ygge:    | Very efficient; For advanced users
Capacitor                           | Internal     |                                | capacitor:        | Very efficient; No support for multi-dimentionnal features.
