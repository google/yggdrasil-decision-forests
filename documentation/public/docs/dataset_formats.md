# Supported dataset formats

This page lists the supported dataset formats and their potential limitations.

In YDF, there are two ways to pass datasets:

-   A **Python dataset object** e.g. a Pandas DataFrame.
-   A **typed-path** e.g. "csv:/path/to/dataset".

Using Python object datasets offers flexibility, as you can easily apply
preprocessing operations to the data. However, they are less efficient for large
datasets because the entire dataset needs to be loaded into memory. As a general
rule, Python object datasets are best suited for datasets with fewer than 100
million examples.

On the other hand, typed-paths are more memory-efficient and are required for
distributed training. However, they are less flexible (preprocessing requires
materializing the output) and have limited support for certain formats and
feature semantics.

The recommended way to provide Python datasets is to use a dictionary of NumPy
arrays. Pandas dataframe as also well supported. The current recommendation for
typed-path datasets is to use TensorFlow Records, but this may change in the
future.

Format                              | Availability | As python object               | Typed-path prefix | Remarks
----------------------------------- | ------------ | ------------------------------ | ----------------- | -------
Dict of numpy array                 | Public       | Native                         |                   | Efficient; Recommended for small datasets
Pandas dataframe                    | Public       | Native                         |                   |
csv                                 | Public       |                                | csv:              | No support for multi-dimentionnal features.
TensorFlow Records (gzip)           | Public       | with ydf.util.read_tf_recordio | tfrecord:         | Efficient; Recommended for large datasets
TensorFlow Records (non-compressed) | Public       | with ydf.util.read_tf_recordio | tfrecordv2:       | Inefficient; To avoid
TensorFlow Tensor                   | Public       | Native                         |                   | Inefficient; To avoid
TensorFlow Dataset                  | Public       | Native                         |                   | Inefficient; To avoid
Xarray                              | Public       | Native                         |                   |
SSTable of TF Examples              | Internal     |                                | sstable+tfe:      |
RecordIO of TF Examples             | Internal     |                                | recordio+tfe:     | Efficient; Recommended for large datasets
RecordIO of YDF Examples            | Internal     |                                | recordio+ygge:    | Very efficient; For advanced users
Capacitor                           | Internal     |                                | capacitor:        | Very efficient; No support for multi-dimentionnal features.
