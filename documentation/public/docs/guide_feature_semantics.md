# Feature Semantics

When training a model, YDF needs to understand how to interpret the feature
values in the training data. A feature might, for example, be a numerical
quantity, a category or a set of tags. The interpretation of a feature’s values
is called **feature semantic**.

The semantic of a feature is related to, but different from, the feature’s
*representation*, that is, the (technical) data type of the feature. For
example, a feature represented by 64-bit integers might have a numerical
semantic or a categorical semantic.

In basic cases, YDF detects the semantics of a feature automatically, so you
only need to check them after the training (e.g. use [model.describe()](py_api/GenericModel/#ydf.GenericModel.describe)). If YDF
does not detect the correct semantic, you can manually override it. Using the
wrong semantic negatively impacts the training speed and quality of a model.
Also, YDF is not able to consume all the types of features. In such cases,
features need to be pre-preprocessed into a supported semantic.

This guide explains the different semantics available in YDF, how to check /
select them, and gives recommendations on how to feed different types of
features into the model.

This guide assumes basic familiarity with YDF, e.g. the
[Getting Started](tutorial/getting_started)
tutorial.

## Introduction: How to specify feature semantics

Unless it is given additional information, YDF automatically determines the
feature semantics when training the model:

```python
model = ydf.RandomForestLearner(label="label").train(ds)
# The "Dataspec" tab of the model description shows the feature semantics
# used for this model.
model.describe()
```

It is possible to override the feature semantics manually using the
`ydf.Semantic` enum:

```python
model = ydf.RandomForestLearner(
    features=[("f1", ydf.Semantic.NUMERICAL), ("f2", ydf.Semantic.CATEGORICAL)],
    include_all_columns=True,  # Also use not explicitly defined features.
    label="label"
).train(ds)
model.describe()
```

Currently, YDF supports 5 input feature semantics. New semantics are added from
time to time following our research. The semantics are:

*   `ydf.Semantic.NUMERICAL`
*  `ydf.Semantic.CATEGORICAL`
*   `ydf.Semantic.BOOLEAN`
*   `ydf.Semantic.CATEGORICAL_SET`
*   `ydf.Semantic.DISCRETIZED_NUMERICAL`

The next section will explain the individual semantics in more detail.

## Feature semantics

### ydf.Semantic.NUMERICAL

NUMERICAL features represent quantities,
amounts, or more generally, any ordered values. For example, age (in years),
duration (in seconds), net worth (in dollars), number of requests (in
count), scores (in points), and even the median of a distribution, are
NUMERICAL features.

YDF automatically recognizes integer and floating-point values as NUMERICAL.

```python
# A dataset with 4 numerical features.
dataset = {
"age": np.array([1, 55, 24, 8]),
"number of cats": np.array([0, 10, 4, 2]),
"net worth": np.array([0.0, 123456.78, -4000.0, 315.42]),
"score": np.array([1.1, 5.2, math.nan, 1.2]),
}
```

### ydf.Semantic.CATEGORICAL

CATEGORICAL features represent categories,
enum values, tags, or more generally, any unordered values. For example,
*species* (among cat, bird, or fish), *blood type* (among A, B, AB, O),
*country,* language, and *project status (in planning, in progress, done,
canceled)*.YDF automatically recognizes string features as CATEGORICAL.

Additional considerations for categorical features are:

*   **Don’t bucket**: With Neural networks, numerical features are sometimes
    bucketed into categorical brackets (e.g., 0-5, 5-10, 10-20). This is not
    beneficial for YDF. If you can, feed directly the value as numerical. If
    you only have the bucketed data, also feed it as a numerical feature.
*   **Don’t use one-hot-encoding:** One-hot encoding for categorical
    features consistently underperforms when using tree algorithms and
    should not be used in YDF.
*   **Preprocessing:** To avoid overfitting, YDF automatically replaces rare
    categorical values with OOD (“out of dictionary”). This generally leads
    to better models. This behavior is controlled by hyperparameters
    `max_vocab_count` (to limit the number of categories) and
    `min_vocab_frequency` (to prune rare categories).
*   **Unknown Values:** During inference, any unknown categorical value is
    treated as OOD which is distinct from a missing value. For example,
    consider a model with a feature “species” that has values “cat”, “bird”
    and “fish” in the training dataset. If, during model inference, an
    instance has value “tiger” for the “species” feature, the model will
    implicitly transform “tiger” to the OOD token.
*   **Python Enums**: Numerical Python enums should generally have a
    categorical semantic, but, as they are integers, are automatically
    recognized as NUMERICAL. Enums should therefore be specified manually as
    CATEGORICAL.

    ```python
    # A dataset with 4 categorical features.
    dataset = {
    "species": np.array(["cat", "bird", "bird", "fish", "fish"]),
    "country": np.array(["US", "US", "Switzerland", "India", "India"]),
    "month": np.array([1, 1, 4, 6, 1]),  # CATEGORICAL features can be integer.
    "blood type": np.array(["A", "B", "AB", "B", ""]), # Missing values are empty.
    }
    model = ydf.RandomForestLearner(
    label="blood type",
    # Since integers are auto-detected as NUMERICAL, specify the semantic manually.
    features=[("month", ydf.Semantic.CATEGORICAL)],
    min_vocab_frequency=2, # Prune vocabulary items that appears only once
    ).train(dataset)
    # Check that all features and the label are CATEGORICAL.
    # Check model.data_spec() column to see the feature's categories.
    model.describe()
    ```

### ydf.Semantic.BOOLEAN

The value of a Boolean features can only be true, false, or
missing. Examples are “has subscribed”, “is spam”, “is in stock” etc.
Boolean features are a special case of categorical features. YDF
automatically recognizes boolean features as BOOLEAN for most dataset
formats.

The label of the model can never be BOOLEAN. Note that binary classification
uses CATEGORICAL labels.

```python
# A dataset with 3 boolean features.
dataset = {
"has subscribed": np.array([True, False, False, True]),
"spam": np.array([1,0,1,1], dtype=bool),  # Ensure the dtype is not integer.
"happy": np.array([True, False, False, True]),
}
```

!!! warning

    Avoid IDs as features: Many datasets have features of type
    "ID" / "identifier", "unique_hash", etc. that are (nearly) unique for each
    example in the dataset. These features are not useful for training a machine
    learning model, they slow down model training and might increase model size.
    It is therefore important to remove these values from the dataset before
    training.

## Special semantics

### ydf.Semantic.DISCRETIZED_NUMERICAL

DISCRETIZED_NUMERICAL is not really a new semantic. Instead, it is used to
tell the learning algorithm to optimize training with a special
discretization algorithm. Any NUMERICAL feature can be configured as
DISCRETIZED_NUMERICAL. Training will be faster (generally ~2x) but it can
hurt the model quality. Setting all the NUMERICAL features as
DISCRETIZED_NUMERICAL is equivalent to setting hyperparameter
`detect_numerical_as_discretized_numerical=True`.

```python
data = {
"age": np.array([1, 55, 24, 8]),
"net worth": np.array([0.0, 123456.78, -4000.0, 315.42]),
"weight": np.array([9, 63, 70, np.nan]),
}
model = ydf.RandomForestLearner(
label="weight",
discretize_numerical_columns=False,  # Default
features=[("net worth", ydf.Semantic.DISCRETIZED_NUMERICAL)],
task=ydf.Task.REGRESSION,
).train(data)
# `net worth` is DISCRETIZED_NUMERICAL,  `age` and `weight` are NUMERICAL.
model.describe()
```

### ydf.Semantic.CATEGORICAL_SET

The value of a categorical-set feature is
a set of categorical values. In other words, while a
ydf.Semantic.CATEGORICAL can only have one value, a
ydf.Semantic.CATEGORICAL_SET feature can have none, one, or many categorical
values.. Use this for sets of discrete values, such as tokenized text or tag
sets (e.g. a webpage talks about {politics, elections, united-states}).

When text features, the tokenization is important to consider. Splitting on
spaces works okay in English, but poorly in Chinese. For example, "A cat
sits on a tree" becomes {a, cat, on, sits, tree}. Using a more powerful
tokenizer might be better. Since a set does not encode position, {a, cat,
on, sits, tree} is equivalent to {a, tree, sits, on, cat}. One solution is
to use multi-grams (e.g., bi-grams) that encode consecutive works. For
example, the bi-grams in our example are {a_cat, cat_sit, sit_on, on_a,
a_tree}. \

*   **Preprocessing:** As with CATEGORICAL features, YDF automatically
    replaces rare categorical values with OOD (“out of dictionary”) for
    CATEGORICAL_SET. This behavior is controlled by hyperparameters
    `max_vocab_count` (to limit the number of categories) and
    `min_vocab_frequency` (to prune rare categories).
*   **Training speed**: CATEGORICAL_SET features are slower to train than
    NUMERICAL or CATEGORICAL features. Don't use CATEGORICAL_SET instead of
    CATEGORICAL features (the result will be the same, but the model will
    train slowly).
*   **Tokenization**: When using CATEGORICAL_SET for text features, the text
    must be tokenized before it is fed to YDF. CSV files are tokenized by 
    whitespace automatically, see the section on CSV files for details.

```python
# A dataset with 2 categorical set features and one categorical feature.
dataset = {
    "title": [["Next", "week", "are", "us", "elections"], ["Reform", "started", "this", "month"], ["Funniest", "politics", "speeches"]],
    "tags": [["politics", "election"], ["politics"], ["funny", "politics"]],
    "interesting": ["yes", "yes", "no"],
}
model = ydf.RandomForestLearner(
    label="interesting",
    min_vocab_frequency=1, # Don't prune the categories.
    features=[("title", ydf.Semantic.CATEGORICAL_SET), ("tags", ydf.Semantic.CATEGORICAL_SET)],
).train(dataset)
```

### Multi-dimensional features

YDF supports constant-size vectors as
features. This is typically used when dealing with vector embeddings. For
example, consider a text feature `text` that is transformed (during
preprocessing) with a Universal Sentence Encoder model to a numerical vector
of 512 entries. This vector can be fed directly to YDF.

YDF “unrolls” each entry of the vector to an individual feature. These
features are named `text.0_of_512`, `text.1_of_512`, etc. Note that all
vectors must have the exact same size - if the vectors have different sizes,
consider the CATEGORICAL_SET semantic. See
[here](tutorial/multidimensional_feature)
for a more detailed example.

```python
# A dataset with a two-dimensional numerical feature
# and a two-dimensional categorical feature.
dataset = {
"categorical_vector": np.array([["a", "b"], ["a", "c"], ["b", "c"]]),
"numerical_embeeding": np.array([[1, 2], [3, 4], [5, 6]]),
"label": np.array([1, 2, 1]),
}
model = ydf.RandomForestLearner(
label="label",
).train(dataset)
# `categorical_vector` is unrolled to two CATEGORICAL features:
# `categorical_vector.0_of_2` and `categorical_vector.1_of_2`.
# `numerical_embeeding` is unrolled to two NUMERICAL features:
# `numerical_embeeding.0_of_2` and `numerical_embeeding.1_of_2`.
model.describe()
```

!!! note

    ydf.Semantic.HASH is used internally only and cannot be used for decision
    tree training.


## Not natively supported semantics

There are some features that YDF cannot consume natively, but instead need to be
preprocessed. This section details the most common scenarios.

### Timestamps

Timestamps should be converted to the NUMERICAL semantic. A popular choice is to
decompose the timestamp into calendar features such as day of the week, week of
the month, hour of the day, etc.. Simply converting a timestamps into a
numerical unix time generally does not work well.

!!! warning

    The presence of timestamps often indicates that the dataset is containing 
    time-series data (see next section).


### Time-series

Time series datasets require advanced feature preprocessing for good model
quality. Check the
[special guide](tutorial/time_sequences)
in the YDF documentation for more information.

!!! warning

    Time series require careful modeling to prevent future leakage or poor model
    quality. For complex problems, using a preprocessing tool such as 
    [Temporian](https://temporian.readthedocs.io) is recommended.


### Repeated proto messages

Repeated proto messages must be “flattened” to individual features. Repeated
numerical entries, creating statistical features such as maximum, minimum, mean,
median, variance, … is useful. Repeated categorical entries can be transformed
to CATEGORICAL_SET features.

### Images

Decision Forest models are not state-of-the-art model architecture for image
processing. In many cases, exploring other model architectures (notably neural
networks) should be preferred.

## Details by dataset format

### Numpy

**Scalar Types**

The following table shows which **scalar Numpy data** types correspond to which
YDF semantics and which casts can be performed by YDF after manually specifying
a feature semantic.

| Numpy\YDF | **NUMERICAL** [3] | **CATEGORICAL** | **BOOLEAN** | **CATEGORICAL SET** [6] | **DISCRETIZED NUMERICAL** |
|---|---|---|---|---|---|
| **int** [1] | <span style="background-color: lightgreen">Default</span> | <span style="background-color: #ffd966">Cast</span> [4] | <span style="background-color: #FFCCCB">No support</span> | <span style="background-color: #FFCCCB">No support</span> | <span style="background-color: #ffd966">Cast</span> |
| **float** [2] | <span style="background-color: lightgreen">Default</span> | <span style="background-color: #FFCCCB">No support</span> | <span style="background-color: #FFCCCB">No support</span> | <span style="background-color: #FFCCCB">No support</span> | <span style="background-color: #ffd966">Cast</span> |
| **bool** | <span style="background-color: #ffd966">Cast</span> | <span style="background-color: #ffd966">Cast</span> [5] | <span style="background-color: lightgreen">Default</span> | <span style="background-color: #FFCCCB">No support</span> | <span style="background-color: #ffd966">Cast</span> |
| **str** | <span style="background-color: #FFCCCB">No support</span> | <span style="background-color: lightgreen">Default</span> | <span style="background-color: #FFCCCB">No support</span> | <span style="background-color: #FFCCCB">No support</span> | <span style="background-color: #FFCCCB">No support</span> |
| **bytes** | <span style="background-color: #FFCCCB">No support</span> | <span style="background-color: lightgreen">Default</span> | <span style="background-color: #FFCCCB">No support</span> | <span style="background-color: #FFCCCB">No support</span> | <span style="background-color: #FFCCCB">No support</span> |

[1]: Includes unsigned integers

[2]: float128 is not supported

[3]: YDF internally casts numerical values to float32.

[4]: Internally, the values are cast to string and sorted by frequency. Label
values are ordered lexicographically (for strings) or increasing (for integers).

[5]: Internally, the values are cast to “false” and “true”, with “false” coming
first.

[6]: Use type `object` for support for CATEGORICAL SET, see below.

**Two-dimensional arrays (i.e. matrices)**

YDF unrolls two-dimensional arrays(i.e. matrices) by column.

**Object**

YDF inspects features given as numpy arrays of type `object` and treats them
differently based on their content.

If the array’s first element is a **scalar type**, YDF attempts to cast the
array to type `bytes` and treat it as CATEGORICAL.

If the array’s first element is a **Python list or Numpy array** (irrespective
of dtype), YDF checks if it contains only lists (arrays) and fails otherwise. If
all lists have the same length, YDF attempts to cast the sub-lists (sub-arrays)
to type `np.bytes` and treat the entire feature as a matrix of CATEGORICAL
features. This matrix is then unrolled into individual features.

If the sub-lists (sub-arrays) have different sizes, YDF attempts to cast the
sub-lists (sub-arrays) to type `np.bytes` and treat the entire feature with
semantic CATEGORICAL_SET.

Any other types or combinations of types are not supported.

**Missing values**

Missing NUMERICAL values are `np.Nan`, missing CATEGORICAL values are empty
strings. Missing BOOLEAN or CATEGORICAL_SET values cannot be represented.

### Python lists

YDF can consume Python lists as features. However, automatic semantic detection
is not enabled for Python lists. Furthermore, multi-dimensional features (except
CATEGORICAL_SET) cannot be fed with Python lists.

### CSV

**Automatic Semantic detection**

*   Columns with only 0 and 1 are recognized as BOOLEAN.
*   Columns with only numeric values are recognized as NUMERICAL.
*   Other columns are recognized as CATEGORICAL (see below).
*   Multidimensional features are not supported.

Example code:

```python
"""!cat mycsv.csv
num,bool,cat,catset,label
1.0,1,a,x y,1
1.5,0,b,y z,2
2.0,1,1,x y z,3"""
model = ydf.RandomForestLearner(
label="label",
# Note that CATEGORICAL_SET columns are tokenized by whitespace.
features=[("catset", ydf.Semantic.CATEGORICAL_SET)],
min_vocab_frequency=1,
include_all_columns=True,
).train("csv:mycsv.csv")
# Column "num" is NUMERICAL.
# Column "bool" is BOOLEAN
# Column "cat" is CATEGORICAL
# Column "catset" is CATEGORICAL_SET (as specified).
model.describe()
```

!!! warning

    Only the first 100000 rows are inspected to determine a column's type. Adapt
    max_num_scanned_rows_to_infer_semantic to increase this value.

**Automatic tokenization**

When reading from CSV files, YDF tokenizes features with semantic
CATEGORICAL_SET by splitting along whitespace.

Some YDF surfaces may automatically infer the type of string columns containing
whitespace as having semantic CATEGORICAL_SET. Note that the Python API of YDF
does not automatically infer type CATEGORICAL_SET, even if reading CSV files.

**Missing values**

Missing values are represented by string `na` or empty strings.

### Avro

**Automatic Semantic detection**

Avro columns are typed, so a column’s type which YDF is used by default.

*   Boolean columns are recognized as BOOLEAN
*   Long, Int, Float and Double columns are recognized as NUMERICAL
*   Bytes and String columns are recognized as CATEGORICAL
*   Array columns are either unrolled based on the type of data in the array.
    Arrays of Bytes or Strings may furthermore have type CATEGORICAL_SET.
*   Nested arrays are currently not supported but may be supported in the
    future.

## Advanced: How decision forests use feature semantics

This section is not required for using YDF, but it might offer additional
context on the individual semantics and how they are used by common decision
forest learning algorithms.

Recall that decision trees recursively split the dataset until a stopping
condition is reached. The feature semantics both dictate which types of splits
are considered, and how the algorithm finds the best split.

*   **NUMERICAL:** The learning algorithm creates splits in the data based on
    thresholds (e.g., "age >= 30").

    For even more powerful models,
    [enable oblique splits](guide_how_to_improve_model/#use-oblique-trees).
    This allows YDF to learn splits that combine multiple numerical features
    (e.g., "0.3 \* age + 0.7 \* income >= 50"). This is particularly helpful for
    smaller datasets but requires more training time.

*   **CATEGORICAL**: The learning algorithm creates splits by grouping values
    into sets (e.g., "country in {USA, Switzerland, Luxemburg}"). This is more
    expressive than numerical splits but computationally more expensive.

*   **BOOLEAN:** This semantic provides a memory- and speed-optimized way to
    handle boolean data compared to using semantics NUMERICAL or CATEGORICAL.

    **CATEGORICAL_SET:** The learning algorithm creates splits based on set
    intersections (e.g., "`{a, cat, on, sits, tree}` intersects `{cat, dog,
    bird}`"). This semantic is computationally expensive and can be slow to
    train. For in-depth information about categorical sets, see
    [Guillame-Bert et al., 2020.](https://arxiv.org/abs/2009.09991)

### Missing Values

YDF effectively handles missing values, which are represented differently for
each semantic. During training, YDF uses global imputation:

*   **NUMERICAL and DISCRETIZED_NUMERICAL:** Missing values are replaced with
    the mean of the feature.
*   **CATEGORICAL and BOOLEAN:** Missing values are replaced with the most
    frequent value.
*   **CATEGORICAL_SET:** Missing values are always routed to the negative branch
    of a split.

If the hyperparameter `allow_na_conditions` is enabled, the learning algorithm
can also create splits of the form “feature is NA”. Note that this is usually
not necessary: Global imputation replaces missing values with a “special” value,
which is quickly learned by the algorithm.

!!! note

    Missing values are not allowed in the label column.

### Label column semantic

The semantic of the label column is determined by the model task:

*   **REGRESSION:** Requires NUMERICAL labels.
*   **CLASSIFICATION:** Requires CATEGORICAL labels.
*   **RANKING:** Requires NUMERICAL labels with integer values.
*   **CATEGORICAL_UPLIFT:** Requires NUMERICAL labels.
*   **NUMERICAL_UPLIFT:** Requires NUMERICAL labels.

Changing the task changes the model's loss function and output, resulting in a
fundamentally different model. If the label column cannot be interpreted with
the chosen task’s semantic, model training fails.

### The data spec

Internally, YDF models store information about the feature semantics in the
"data spec". A model's data spec is a proto message that can be displayed with
`model.data_spec()`. The data spec also contains information about
the feature values seen during training. Some of this information may be used
during training and inference. In particular, the data spec stores the following
information:

*   For NUMERICAL and DISCRETIZED_NUMERICAL features, statistical information
    and, if relevant, bucketization.
*   For CATEGORICAL and CATEGORICAL_SET features, the dictionary of values seen
    during training and their frequency. Note that the ordering is not
    guaranteed to be stable across versions and should not be relied upon.
*   For BOOLEAN values, the frequency of true and false values seen during
    training.

A summary of the data spec is shown in `model.describe()`. The raw data spec
is mainly useful for debugging issues with YDF.
