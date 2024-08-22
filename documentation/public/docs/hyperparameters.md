# Hyper-parameters

Learners and their respective hyperparameters are listed on this page.

## Usage

With the **Python** and **TensorFlow Decision Forests APIs**, hyperparameters
are provided as **constructor arguments**. For example:

```python
import ydf
model = ydf.RandomForestLearner(num_trees=1000).train(...)

import tensorflow_decision_forests as tfdf
model = tfdf.keras.RandomForestModel(num_trees=1000)
```

With the **C++** and **CLI APIs**, the hyper-parameters are passed in the
**training configuration protobuffer**. For example:

```c++
learner: "RANDOM_FOREST"
[yggdrasil_decision_forests.model.random_forest.proto.random_forest_config] {
  num_trees: 1000
}
```
## GRADIENT_BOOSTED_TREES

A [Gradient Boosted Trees](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)
(GBT), also known as Gradient Boosted Decision Trees (GBDT) or Gradient Boosted
Machines (GBM), is a set of shallow decision trees trained sequentially. Each
tree is trained to predict and then "correct" for the errors of the previously
trained trees (more precisely each tree predict the gradient of the loss
relative to the model output).

### Protobuffer training configuration

The hyper-parameter protobuffers are used with the C++ and CLI APIs.

-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto">learner/abstract_learner.proto</a>
-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto">learner/decision_tree/decision_tree.proto</a>
-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto">learner/gradient_boosted_trees/gradient_boosted_trees.proto</a>

### Hyper-parameter templates

**Hyper-parameter templates** are pre-configured sets of hyperparameters that
are updated with the best values that have been found. Unlike default values,
which never change, hyper-parameter templates are updated when new features are
available in YDF. Hyper-parameter templates can be copied manually using the C++
or CLI APIs, or they can be passed to the `hyperparameter_template constructor`
argument in the Python and TensorFlow Decision Forests APIs.

**better_default@1**

A configuration that is generally better than the default parameters without
being more expensive.

-   `growing_strategy`: BEST_FIRST_GLOBAL

**benchmark_rank1@1**

Top ranking hyper-parameters on our benchmark slightly modified to run in
reasonable time.

-   `growing_strategy`: BEST_FIRST_GLOBAL
-   `categorical_algorithm`: RANDOM
-   `split_axis`: SPARSE_OBLIQUE
-   `sparse_oblique_normalization`: MIN_MAX
-   `sparse_oblique_num_projections_exponent`: 1

### Hyper-parameters

#### [adapt_subsample_for_maximum_training_duration](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   Control how the maximum training duration (if set) is applied. If false, the
    training stop when the time is used. If true, the size of the sampled
    datasets used train individual trees are adapted dynamically so that all the
    trees are trained in time.

#### [allow_na_conditions](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   If true, the tree training evaluates conditions of the type `X is NA` i.e.
    `X is missing`.

#### [apply_link_function](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   If true, applies the link function (a.k.a. activation function), if any, before returning the model prediction. If false, returns the pre-link function model output.<br>For example, in the case of binary classification, the pre-link function output is a logic while the post-link function is a probability.

#### [categorical_algorithm](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** CART **Possible values:** CART, ONE_HOT,
    RANDOM

-   How to learn splits on categorical attributes.<br>- `CART`: CART algorithm. Find categorical splits of the form "value \in mask". The solution is exact for binary classification, regression and ranking. It is approximated for multi-class classification. This is a good first algorithm to use. In case of overfitting (very small dataset, large dictionary), the "random" algorithm is a good alternative.<br>- `ONE_HOT`: One-hot encoding. Find the optimal categorical split of the form "attribute == param". This method is similar (but more efficient) than converting converting each possible categorical value into a boolean feature. This method is available for comparison purpose and generally performs worse than other alternatives.<br>- `RANDOM`: Best splits among a set of random candidate. Find the a categorical split of the form "value \in mask" using a random search. This solution can be seen as an approximation of the CART algorithm. This method is a strong alternative to CART. This algorithm is inspired from section "5.1 Categorical Variables" of "Random Forest", 2001.

#### [categorical_set_split_greedy_sampling](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 0.1 **Possible values:** min:0 max:1

-   For categorical set splits e.g. texts. Probability for a categorical value
    to be a candidate for the positive set. The sampling is applied once per
    node (i.e. not at every step of the greedy optimization).

#### [categorical_set_split_max_num_items](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** -1 **Possible values:** min:-1

-   For categorical set splits e.g. texts. Maximum number of items (prior to the
    sampling). If more items are available, the least frequent items are
    ignored. Changing this value is similar to change the "max_vocab_count"
    before loading the dataset, with the following exception: With
    `max_vocab_count`, all the remaining items are grouped in a special
    Out-of-vocabulary item. With `max_num_items`, this is not the case.

#### [categorical_set_split_min_item_frequency](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 1 **Possible values:** min:1

-   For categorical set splits e.g. texts. Minimum number of occurrences of an
    item to be considered.

#### [compute_permutation_variable_importance](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   If true, compute the permutation variable importance of the model at the end
    of the training using the validation dataset. Enabling this feature can
    increase the training time significantly.

#### [dart_dropout](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 0.01 **Possible values:** min:0 max:1

-   Dropout rate applied when using the DART i.e. when forest_extraction=DART.

#### [early_stopping](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Categorical **Default:** LOSS_INCREASE **Possible values:** NONE,
    MIN_LOSS_FINAL, LOSS_INCREASE

-   Early stopping detects the overfitting of the model and halts it training using the validation dataset. If not provided directly, the validation dataset is extracted from the training dataset (see "validation_ratio" parameter):<br>- `NONE`: No early stopping. All the num_trees are trained and kept.<br>- `MIN_LOSS_FINAL`: All the num_trees are trained. The model is then truncated to minimize the validation loss i.e. some of the trees are discarded as to minimum the validation loss.<br>- `LOSS_INCREASE`: Classical early stopping. Stop the training when the validation does not decrease for `early_stopping_num_trees_look_ahead` trees.

#### [early_stopping_initial_iteration](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Integer **Default:** 10 **Possible values:** min:0

-   0-based index of the first iteration considered for early stopping
    computation. Increasing this value prevents too early stopping due to noisy
    initial iterations of the learner.

#### [early_stopping_num_trees_look_ahead](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Integer **Default:** 30 **Possible values:** min:1

-   Rolling number of trees used to detect validation loss increase and trigger
    early stopping.

#### [focal_loss_alpha](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 0.5 **Possible values:** min:0 max:1

-   EXPERIMENTAL. Weighting parameter for focal loss, positive samples weighted
    by alpha, negative samples by (1-alpha). The default 0.5 value means no
    active class-level weighting. Only used with focal loss i.e.
    `loss="BINARY_FOCAL_LOSS"`

#### [focal_loss_gamma](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 2 **Possible values:** min:0

-   EXPERIMENTAL. Exponent of the misprediction exponent term in focal loss,
    corresponds to gamma parameter in https://arxiv.org/pdf/1708.02002.pdf. Only
    used with focal loss i.e. `loss="BINARY_FOCAL_LOSS"`

#### [forest_extraction](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Categorical **Default:** MART **Possible values:** MART, DART

-   How to construct the forest:<br>- MART: For Multiple Additive Regression Trees. The "classical" way to build a GBDT i.e. each tree tries to "correct" the mistakes of the previous trees.<br>- DART: For Dropout Additive Regression Trees. A modification of MART proposed in http://proceedings.mlr.press/v38/korlakaivinayak15.pdf. Here, each tree tries to "correct" the mistakes of a random subset of the previous trees.

#### [goss_alpha](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 0.2 **Possible values:** min:0 max:1

-   Alpha parameter for the GOSS (Gradient-based One-Side Sampling; "See
    LightGBM: A Highly Efficient Gradient Boosting Decision Tree") sampling
    method.

#### [goss_beta](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 0.1 **Possible values:** min:0 max:1

-   Beta parameter for the GOSS (Gradient-based One-Side Sampling) sampling
    method.

#### [growing_strategy](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** LOCAL **Possible values:** LOCAL,
    BEST_FIRST_GLOBAL

-   How to grow the tree.<br>- `LOCAL`: Each node is split independently of the other nodes. In other words, as long as a node satisfy the splits "constraints (e.g. maximum depth, minimum number of observations), the node will be split. This is the "classical" way to grow decision trees.<br>- `BEST_FIRST_GLOBAL`: The node with the best loss reduction among all the nodes of the tree is selected for splitting. This method is also called "best first" or "leaf-wise growth". See "Best-first decision tree learning", Shi and "Additive logistic regression : A statistical view of boosting", Friedman for more details.

#### [honest](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   In honest trees, different training examples are used to infer the structure
    and the leaf values. This regularization technique trades examples for bias
    estimates. It might increase or reduce the quality of the model. See
    "Generalized Random Forests", Athey et al. In this paper, Honest trees are
    trained with the Random Forest algorithm with a sampling without
    replacement.

#### [honest_fixed_separation](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   For honest trees only i.e. honest=true. If true, a new random separation is
    generated for each tree. If false, the same separation is used for all the
    trees (e.g., in Gradient Boosted Trees containing multiple trees).

#### [honest_ratio_leaf_examples](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 0.5 **Possible values:** min:0 max:1

-   For honest trees only i.e. honest=true. Ratio of examples used to set the
    leaf values.

#### [in_split_min_examples_check](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   Whether to check the `min_examples` constraint in the split search (i.e.
    splits leading to one child having less than `min_examples` examples are
    considered invalid) or before the split search (i.e. a node can be derived
    only if it contains more than `min_examples` examples). If false, there can
    be nodes with less than `min_examples` training examples.

#### [keep_non_leaf_label_distribution](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   Whether to keep the node value (i.e. the distribution of the labels of the
    training examples) of non-leaf nodes. This information is not used during
    serving, however it can be used for model interpretation as well as hyper
    parameter tuning. This can take lots of space, sometimes accounting for half
    of the model size.

#### [l1_regularization](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 0 **Possible values:** min:0

-   L1 regularization applied to the training loss. Impact the tree structures
    and lead values.

#### [l2_categorical_regularization](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 1 **Possible values:** min:0

-   L2 regularization applied to the training loss for categorical features.
    Impact the tree structures and lead values.

#### [l2_regularization](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 0 **Possible values:** min:0

-   L2 regularization applied to the training loss for all features except the
    categorical ones.

#### [lambda_loss](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 1 **Possible values:** min:0

-   Lambda regularization applied to certain training loss functions. Only for
    NDCG loss.

#### [loss](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Categorical **Default:** DEFAULT **Possible values:** DEFAULT,
    BINOMIAL_LOG_LIKELIHOOD, SQUARED_ERROR, MULTINOMIAL_LOG_LIKELIHOOD,
    LAMBDA_MART_NDCG5, XE_NDCG_MART, BINARY_FOCAL_LOSS, POISSON,
    MEAN_AVERAGE_ERROR

-   The loss optimized by the model. If not specified (DEFAULT) the loss is selected automatically according to the \"task\" and label statistics. For example, if task=CLASSIFICATION and the label has two possible values, the loss will be set to BINOMIAL_LOG_LIKELIHOOD. Possible values are:<br>- `DEFAULT`: Select the loss automatically according to the task and label statistics.<br>- `BINOMIAL_LOG_LIKELIHOOD`: Binomial log likelihood. Only valid for binary classification.<br>- `SQUARED_ERROR`: Least square loss. Only valid for regression.<br>- `POISSON`: Poisson log likelihood loss. Mainly used for counting problems. Only valid for regression.<br>- `MULTINOMIAL_LOG_LIKELIHOOD`: Multinomial log likelihood i.e. cross-entropy. Only valid for binary or multi-class classification.<br>- `LAMBDA_MART_NDCG5`: LambdaMART with NDCG5.<br>- `XE_NDCG_MART`:  Cross Entropy Loss NDCG. See arxiv.org/abs/1911.09798.<br>- `BINARY_FOCAL_LOSS`: Focal loss. Only valid for binary classification. See https://arxiv.org/pdf/1708.02002.pdf.<br>- `POISSON`: Poisson log likelihood. Only valid for regression.<br>- `MEAN_AVERAGE_ERROR`: Mean average error a.k.a. MAE.<br>

#### [max_depth](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 6 **Possible values:** min:-1

-   Maximum depth of the tree. `max_depth=1` means that all trees will be roots.
    `max_depth=-1` means that tree depth is not restricted by this parameter.
    Values <= -2 will be ignored.

#### [max_num_nodes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 31 **Possible values:** min:-1

-   Maximum number of nodes in the tree. Set to -1 to disable this limit. Only
    available for `growing_strategy=BEST_FIRST_GLOBAL`.

#### [maximum_model_size_in_memory_in_bytes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Real **Default:** -1

-   Limit the size of the model when stored in ram. Different algorithms can
    enforce this limit differently. Note that when models are compiled into an
    inference, the size of the inference engine is generally much smaller than
    the original model.

#### [maximum_training_duration_seconds](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Real **Default:** -1

-   Maximum training duration of the model expressed in seconds. Each learning
    algorithm is free to use this parameter at it sees fit. Enabling maximum
    training duration makes the model training non-deterministic.

#### [mhld_oblique_max_num_attributes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 4 **Possible values:** min:1

-   For MHLD oblique splits i.e. `split_axis=MHLD_OBLIQUE`. Maximum number of
    attributes in the projection. Increasing this value increases the training
    time. Decreasing this value acts as a regularization. The value should be in
    [2, num_numerical_features]. If the value is above the total number of
    numerical features, the value is capped automatically. The value 1 is
    allowed but results in ordinary (non-oblique) splits.

#### [mhld_oblique_sample_attributes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   For MHLD oblique splits i.e. `split_axis=MHLD_OBLIQUE`. If true, applies the
    attribute sampling controlled by the "num_candidate_attributes" or
    "num_candidate_attributes_ratio" parameters. If false, all the attributes
    are tested.

#### [min_examples](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 5 **Possible values:** min:1

-   Minimum number of examples in a node.

#### [missing_value_policy](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** GLOBAL_IMPUTATION **Possible values:**
    GLOBAL_IMPUTATION, LOCAL_IMPUTATION, RANDOM_LOCAL_IMPUTATION

-   Method used to handle missing attribute values.<br>- `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean (in case of numerical attribute) or the most-frequent-item (in case of categorical attribute) computed on the entire dataset (i.e. the information contained in the data spec).<br>- `LOCAL_IMPUTATION`: Missing attribute values are imputed with the mean (numerical attribute) or most-frequent-item (in the case of categorical attribute) evaluated on the training examples in the current node.<br>- `RANDOM_LOCAL_IMPUTATION`: Missing attribute values are imputed from randomly sampled values from the training examples in the current node. This method was proposed by Clinic et al. in "Random Survival Forests" (https://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908043).

#### [num_candidate_attributes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** -1 **Possible values:** min:-1

-   Number of unique valid attributes tested for each node. An attribute is
    valid if it has at least a valid split. If `num_candidate_attributes=0`, the
    value is set to the classical default value for Random Forest: `sqrt(number
    of input attributes)` in case of classification and
    `number_of_input_attributes / 3` in case of regression. If
    `num_candidate_attributes=-1`, all the attributes are tested.

#### [num_candidate_attributes_ratio](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** -1 **Possible values:** min:-1 max:1

-   Ratio of attributes tested at each node. If set, it is equivalent to
    `num_candidate_attributes = number_of_input_features x
    num_candidate_attributes_ratio`. The possible values are between ]0, and 1]
    as well as -1. If not set or equal to -1, the `num_candidate_attributes` is
    used.

#### [num_trees](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Integer **Default:** 300 **Possible values:** min:0

-   Maximum number of decision trees. The effective number of trained tree can
    be smaller if early stopping is enabled.

#### [pure_serving_model](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   Clear the model from any information that is not required for model serving.
    This includes debugging, model interpretation and other meta-data. The size
    of the serialized model can be reduced significatively (50% model size
    reduction is common). This parameter has no impact on the quality, serving
    speed or RAM usage of model serving.

#### [random_seed](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Integer **Default:** 123456

-   Random seed for the training of the model. Learners are expected to be
    deterministic by the random seed.

#### [sampling_method](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Categorical **Default:** RANDOM **Possible values:** NONE, RANDOM,
    GOSS, SELGB

-   Control the sampling of the datasets used to train individual trees.<br>- NONE: No sampling is applied. This is equivalent to RANDOM sampling with \"subsample=1\".<br>- RANDOM (default): Uniform random sampling. Automatically selected if "subsample" is set.<br>- GOSS: Gradient-based One-Side Sampling. Automatically selected if "goss_alpha" or "goss_beta" is set.<br>- SELGB: Selective Gradient Boosting. Automatically selected if "selective_gradient_boosting_ratio" is set. Only valid for ranking.<br>

#### [selective_gradient_boosting_ratio](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 0.01 **Possible values:** min:0 max:1

-   Ratio of the dataset used to train individual tree for the selective
    Gradient Boosting (Selective Gradient Boosting for Effective Learning to
    Rank; Lucchese et al;
    http://quickrank.isti.cnr.it/selective-data/selective-SIGIR2018.pdf)
    sampling method.

#### [shrinkage](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 0.1 **Possible values:** min:0 max:1

-   Coefficient applied to each tree prediction. A small value (0.02) tends to
    give more accurate results (assuming enough trees are trained), but results
    in larger models. Analogous to neural network learning rate. Fixed to 1.0
    for DART models.

#### [sorting_strategy](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** PRESORT **Possible values:** IN_NODE,
    PRESORT, FORCE_PRESORT, AUTO

-   How are sorted the numerical features in order to find the splits<br>- AUTO: Selects the most efficient method among IN_NODE, FORCE_PRESORT, and LAYER.<br>- IN_NODE: The features are sorted just before being used in the node. This solution is slow but consumes little amount of memory.<br>- FORCE_PRESORT: The features are pre-sorted at the start of the training. This solution is faster but consumes much more memory than IN_NODE.<br>- PRESORT: Automatically choose between FORCE_PRESORT and IN_NODE.<br>.

#### [sparse_oblique_max_num_projections](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 6000 **Possible values:** min:1

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Maximum number of projections (applied after the num_projections_exponent).<br>Oblique splits try out max(p^num_projections_exponent, max_num_projections) random projections for choosing a split, where p is the number of numerical features. Increasing "max_num_projections" increases the training time but not the inference time. In late stage model development, if every bit of accuracy if important, increase this value.<br>The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020) does not define this hyperparameter.

#### [sparse_oblique_normalization](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** NONE **Possible values:** NONE,
    STANDARD_DEVIATION, MIN_MAX

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before applying the sparse oblique projections.<br>- `NONE`: No normalization.<br>- `STANDARD_DEVIATION`: Normalize the feature by the estimated standard deviation on the entire train dataset. Also known as Z-Score normalization.<br>- `MIN_MAX`: Normalize the feature by the range (i.e. max-min) estimated on the entire train dataset.

#### [sparse_oblique_num_projections_exponent](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 2 **Possible values:** min:0

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections to test at each node.<br>Increasing this value very likely improves the quality of the model, drastically increases the training time, and doe not impact the inference time.<br>Oblique splits try out max(p^num_projections_exponent, max_num_projections) random projections for choosing a split, where p is the number of numerical features. Therefore, increasing this `num_projections_exponent` and possibly `max_num_projections` may improve model quality, but will also significantly increase training time.<br>Note that the complexity of (classic) Random Forests is roughly proportional to `num_projections_exponent=0.5`, since it considers sqrt(num_features) for a split. The complexity of (classic) GBDT is roughly proportional to `num_projections_exponent=1`, since it considers all features for a split.<br>The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020) recommends values in [1/4, 2].

#### [sparse_oblique_projection_density_factor](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 2 **Possible values:** min:0

-   Density of the projections as an exponent of the number of features. Independently for each projection, each feature has a probability "projection_density_factor / num_features" to be considered in the projection.<br>The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020) calls this parameter `lambda` and recommends values in [1, 5].<br>Increasing this value increases training and inference time (on average). This value is best tuned for each dataset.

#### [sparse_oblique_weights](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** BINARY **Possible values:** BINARY,
    CONTINUOUS

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Possible values:<br>- `BINARY`: The oblique weights are sampled in {-1,1} (default).<br>- `CONTINUOUS`: The oblique weights are be sampled in [-1,1].

#### [split_axis](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** AXIS_ALIGNED **Possible values:**
    AXIS_ALIGNED, SPARSE_OBLIQUE, MHLD_OBLIQUE

-   What structure of split to consider for numerical features.<br>- `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This is the "classical" way to train a tree. Default value.<br>- `SPARSE_OBLIQUE`: Sparse oblique splits (i.e. random splits on a small number of features) from "Sparse Projection Oblique Random Forests", Tomita et al., 2020.<br>- `MHLD_OBLIQUE`: Multi-class Hellinger Linear Discriminant splits from "Classification Based on Multivariate Contrast Patterns", Canete-Sifuentes et al., 2029

#### [subsample](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 1 **Possible values:** min:0 max:1

-   Ratio of the dataset (sampling without replacement) used to train individual
    trees for the random sampling method. If \"subsample\" is set and if
    \"sampling_method\" is NOT set or set to \"NONE\", then \"sampling_method\"
    is implicitly set to \"RANDOM\". In other words, to enable random
    subsampling, you only need to set "\"subsample\".

#### [uplift_min_examples_in_treatment](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 5 **Possible values:** min:0

-   For uplift models only. Minimum number of examples per treatment in a node.

#### [uplift_split_score](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** KULLBACK_LEIBLER **Possible values:**
    KULLBACK_LEIBLER, KL, EUCLIDEAN_DISTANCE, ED, CHI_SQUARED, CS,
    CONSERVATIVE_EUCLIDEAN_DISTANCE, CED

-   For uplift models only. Splitter score i.e. score optimized by the splitters. The scores are introduced in "Decision trees for uplift modeling with single and multiple treatments", Rzepakowski et al. Notation: `p` probability / average value of the positive outcome, `q` probability / average value in the control group.<br>- `KULLBACK_LEIBLER` or `KL`: - p log (p/q)<br>- `EUCLIDEAN_DISTANCE` or `ED`: (p-q)^2<br>- `CHI_SQUARED` or `CS`: (p-q)^2/q<br>

#### [use_goss](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false


#### [use_hessian_gain](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   Use true, uses a formulation of split gain with a hessian term i.e.
    optimizes the splits to minimize the variance of "gradient / hessian.
    Available for all losses except regression.

#### [validation_interval_in_trees](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Integer **Default:** 1 **Possible values:** min:1

-   Evaluate the model on the validation set every
    "validation_interval_in_trees" trees. Increasing this value reduce the cost
    of validation and can impact the early stopping policy (as early stopping is
    only tested during the validation).

#### [validation_ratio](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 0.1 **Possible values:** min:0 max:1

-   Fraction of the training dataset used for validation if not validation
    dataset is provided. The validation dataset, whether provided directly or
    extracted from the training dataset, is used to compute the validation loss,
    other validation metrics, and possibly trigger early stopping (if enabled).
    When early stopping is disabled, the validation dataset is only used for
    monitoring and does not influence the model directly. If the
    "validation_ratio" is set to 0, early stopping is disabled (i.e., it implies
    setting early_stopping=NONE).

## RANDOM_FOREST

A [Random Forest](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
is a collection of deep CART decision trees trained independently and without
pruning. Each tree is trained on a random subset of the original training
dataset (sampled with replacement).

The algorithm is unique in that it is robust to overfitting, even in extreme
cases e.g. when there are more features than training examples.

It is probably the most well-known of the Decision Forest training algorithms.

### Protobuffer training configuration

The hyper-parameter protobuffers are used with the C++ and CLI APIs.

-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto">learner/abstract_learner.proto</a>
-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto">learner/decision_tree/decision_tree.proto</a>
-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/random_forest/random_forest.proto">learner/random_forest/random_forest.proto</a>

### Hyper-parameter templates

**Hyper-parameter templates** are pre-configured sets of hyperparameters that
are updated with the best values that have been found. Unlike default values,
which never change, hyper-parameter templates are updated when new features are
available in YDF. Hyper-parameter templates can be copied manually using the C++
or CLI APIs, or they can be passed to the `hyperparameter_template constructor`
argument in the Python and TensorFlow Decision Forests APIs.

**better_default@1**

A configuration that is generally better than the default parameters without
being more expensive.

-   `winner_take_all`: true

**benchmark_rank1@1**

Top ranking hyper-parameters on our benchmark slightly modified to run in
reasonable time.

-   `winner_take_all`: true
-   `categorical_algorithm`: RANDOM
-   `split_axis`: SPARSE_OBLIQUE
-   `sparse_oblique_normalization`: MIN_MAX
-   `sparse_oblique_num_projections_exponent`: 1

### Hyper-parameters

#### [adapt_bootstrap_size_ratio_for_maximum_training_duration](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/random_forest/random_forest.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   Control how the maximum training duration (if set) is applied. If false, the
    training stop when the time is used. If true, adapts the size of the sampled
    dataset used to train each tree such that `num_trees` will train within
    `maximum_training_duration`. Has no effect if there is no maximum training
    duration specified.

#### [allow_na_conditions](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   If true, the tree training evaluates conditions of the type `X is NA` i.e.
    `X is missing`.

#### [bootstrap_size_ratio](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/random_forest/random_forest.proto)

-   **Type:** Real **Default:** 1 **Possible values:** min:0

-   Number of examples used to train each trees; expressed as a ratio of the
    training dataset size.

#### [bootstrap_training_dataset](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/random_forest/random_forest.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   If true (default), each tree is trained on a separate dataset sampled with
    replacement from the original dataset. If false, all the trees are trained
    on the entire same dataset. If bootstrap_training_dataset:false, OOB metrics
    are not available. bootstrap_training_dataset=false is used in "Extremely
    randomized trees"
    (https://link.springer.com/content/pdf/10.1007%2Fs10994-006-6226-1.pdf).

#### [categorical_algorithm](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** CART **Possible values:** CART, ONE_HOT,
    RANDOM

-   How to learn splits on categorical attributes.<br>- `CART`: CART algorithm. Find categorical splits of the form "value \in mask". The solution is exact for binary classification, regression and ranking. It is approximated for multi-class classification. This is a good first algorithm to use. In case of overfitting (very small dataset, large dictionary), the "random" algorithm is a good alternative.<br>- `ONE_HOT`: One-hot encoding. Find the optimal categorical split of the form "attribute == param". This method is similar (but more efficient) than converting converting each possible categorical value into a boolean feature. This method is available for comparison purpose and generally performs worse than other alternatives.<br>- `RANDOM`: Best splits among a set of random candidate. Find the a categorical split of the form "value \in mask" using a random search. This solution can be seen as an approximation of the CART algorithm. This method is a strong alternative to CART. This algorithm is inspired from section "5.1 Categorical Variables" of "Random Forest", 2001.

#### [categorical_set_split_greedy_sampling](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 0.1 **Possible values:** min:0 max:1

-   For categorical set splits e.g. texts. Probability for a categorical value
    to be a candidate for the positive set. The sampling is applied once per
    node (i.e. not at every step of the greedy optimization).

#### [categorical_set_split_max_num_items](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** -1 **Possible values:** min:-1

-   For categorical set splits e.g. texts. Maximum number of items (prior to the
    sampling). If more items are available, the least frequent items are
    ignored. Changing this value is similar to change the "max_vocab_count"
    before loading the dataset, with the following exception: With
    `max_vocab_count`, all the remaining items are grouped in a special
    Out-of-vocabulary item. With `max_num_items`, this is not the case.

#### [categorical_set_split_min_item_frequency](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 1 **Possible values:** min:1

-   For categorical set splits e.g. texts. Minimum number of occurrences of an
    item to be considered.

#### [compute_oob_performances](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/random_forest/random_forest.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   If true, compute the Out-of-bag evaluation (then available in the summary
    and model inspector). This evaluation is a cheap alternative to
    cross-validation evaluation.

#### [compute_oob_variable_importances](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/random_forest/random_forest.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   If true, compute the Out-of-bag feature importance (then available in the
    summary and model inspector). Note that the OOB feature importance can be
    expensive to compute.

#### [growing_strategy](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** LOCAL **Possible values:** LOCAL,
    BEST_FIRST_GLOBAL

-   How to grow the tree.<br>- `LOCAL`: Each node is split independently of the other nodes. In other words, as long as a node satisfy the splits "constraints (e.g. maximum depth, minimum number of observations), the node will be split. This is the "classical" way to grow decision trees.<br>- `BEST_FIRST_GLOBAL`: The node with the best loss reduction among all the nodes of the tree is selected for splitting. This method is also called "best first" or "leaf-wise growth". See "Best-first decision tree learning", Shi and "Additive logistic regression : A statistical view of boosting", Friedman for more details.

#### [honest](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   In honest trees, different training examples are used to infer the structure
    and the leaf values. This regularization technique trades examples for bias
    estimates. It might increase or reduce the quality of the model. See
    "Generalized Random Forests", Athey et al. In this paper, Honest trees are
    trained with the Random Forest algorithm with a sampling without
    replacement.

#### [honest_fixed_separation](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   For honest trees only i.e. honest=true. If true, a new random separation is
    generated for each tree. If false, the same separation is used for all the
    trees (e.g., in Gradient Boosted Trees containing multiple trees).

#### [honest_ratio_leaf_examples](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 0.5 **Possible values:** min:0 max:1

-   For honest trees only i.e. honest=true. Ratio of examples used to set the
    leaf values.

#### [in_split_min_examples_check](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   Whether to check the `min_examples` constraint in the split search (i.e.
    splits leading to one child having less than `min_examples` examples are
    considered invalid) or before the split search (i.e. a node can be derived
    only if it contains more than `min_examples` examples). If false, there can
    be nodes with less than `min_examples` training examples.

#### [keep_non_leaf_label_distribution](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   Whether to keep the node value (i.e. the distribution of the labels of the
    training examples) of non-leaf nodes. This information is not used during
    serving, however it can be used for model interpretation as well as hyper
    parameter tuning. This can take lots of space, sometimes accounting for half
    of the model size.

#### [max_depth](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 16 **Possible values:** min:-1

-   Maximum depth of the tree. `max_depth=1` means that all trees will be roots.
    `max_depth=-1` means that tree depth is not restricted by this parameter.
    Values <= -2 will be ignored.

#### [max_num_nodes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 31 **Possible values:** min:-1

-   Maximum number of nodes in the tree. Set to -1 to disable this limit. Only
    available for `growing_strategy=BEST_FIRST_GLOBAL`.

#### [maximum_model_size_in_memory_in_bytes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Real **Default:** -1

-   Limit the size of the model when stored in ram. Different algorithms can
    enforce this limit differently. Note that when models are compiled into an
    inference, the size of the inference engine is generally much smaller than
    the original model.

#### [maximum_training_duration_seconds](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Real **Default:** -1

-   Maximum training duration of the model expressed in seconds. Each learning
    algorithm is free to use this parameter at it sees fit. Enabling maximum
    training duration makes the model training non-deterministic.

#### [mhld_oblique_max_num_attributes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 4 **Possible values:** min:1

-   For MHLD oblique splits i.e. `split_axis=MHLD_OBLIQUE`. Maximum number of
    attributes in the projection. Increasing this value increases the training
    time. Decreasing this value acts as a regularization. The value should be in
    [2, num_numerical_features]. If the value is above the total number of
    numerical features, the value is capped automatically. The value 1 is
    allowed but results in ordinary (non-oblique) splits.

#### [mhld_oblique_sample_attributes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   For MHLD oblique splits i.e. `split_axis=MHLD_OBLIQUE`. If true, applies the
    attribute sampling controlled by the "num_candidate_attributes" or
    "num_candidate_attributes_ratio" parameters. If false, all the attributes
    are tested.

#### [min_examples](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 5 **Possible values:** min:1

-   Minimum number of examples in a node.

#### [missing_value_policy](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** GLOBAL_IMPUTATION **Possible values:**
    GLOBAL_IMPUTATION, LOCAL_IMPUTATION, RANDOM_LOCAL_IMPUTATION

-   Method used to handle missing attribute values.<br>- `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean (in case of numerical attribute) or the most-frequent-item (in case of categorical attribute) computed on the entire dataset (i.e. the information contained in the data spec).<br>- `LOCAL_IMPUTATION`: Missing attribute values are imputed with the mean (numerical attribute) or most-frequent-item (in the case of categorical attribute) evaluated on the training examples in the current node.<br>- `RANDOM_LOCAL_IMPUTATION`: Missing attribute values are imputed from randomly sampled values from the training examples in the current node. This method was proposed by Clinic et al. in "Random Survival Forests" (https://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908043).

#### [num_candidate_attributes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 0 **Possible values:** min:-1

-   Number of unique valid attributes tested for each node. An attribute is
    valid if it has at least a valid split. If `num_candidate_attributes=0`, the
    value is set to the classical default value for Random Forest: `sqrt(number
    of input attributes)` in case of classification and
    `number_of_input_attributes / 3` in case of regression. If
    `num_candidate_attributes=-1`, all the attributes are tested.

#### [num_candidate_attributes_ratio](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** -1 **Possible values:** min:-1 max:1

-   Ratio of attributes tested at each node. If set, it is equivalent to
    `num_candidate_attributes = number_of_input_features x
    num_candidate_attributes_ratio`. The possible values are between ]0, and 1]
    as well as -1. If not set or equal to -1, the `num_candidate_attributes` is
    used.

#### [num_oob_variable_importances_permutations](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/random_forest/random_forest.proto)

-   **Type:** Integer **Default:** 1 **Possible values:** min:1

-   Number of time the dataset is re-shuffled to compute the permutation
    variable importances. Increasing this value increase the training time (if
    "compute_oob_variable_importances:true") as well as the stability of the oob
    variable importance metrics.

#### [num_trees](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/random_forest/random_forest.proto)

-   **Type:** Integer **Default:** 300 **Possible values:** min:0

-   Number of individual decision trees. Increasing the number of trees can
    increase the quality of the model at the expense of size, training speed,
    and inference latency.

#### [pure_serving_model](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   Clear the model from any information that is not required for model serving.
    This includes debugging, model interpretation and other meta-data. The size
    of the serialized model can be reduced significatively (50% model size
    reduction is common). This parameter has no impact on the quality, serving
    speed or RAM usage of model serving.

#### [random_seed](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Integer **Default:** 123456

-   Random seed for the training of the model. Learners are expected to be
    deterministic by the random seed.

#### [sampling_with_replacement](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/random_forest/random_forest.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   If true, the training examples are sampled with replacement. If false, the
    training samples are sampled without replacement. Only used when
    "bootstrap_training_dataset=true". If false (sampling without replacement)
    and if "bootstrap_size_ratio=1" (default), all the examples are used to
    train all the trees (you probably do not want that).

#### [sorting_strategy](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** PRESORT **Possible values:** IN_NODE,
    PRESORT, FORCE_PRESORT, AUTO

-   How are sorted the numerical features in order to find the splits<br>- AUTO: Selects the most efficient method among IN_NODE, FORCE_PRESORT, and LAYER.<br>- IN_NODE: The features are sorted just before being used in the node. This solution is slow but consumes little amount of memory.<br>- FORCE_PRESORT: The features are pre-sorted at the start of the training. This solution is faster but consumes much more memory than IN_NODE.<br>- PRESORT: Automatically choose between FORCE_PRESORT and IN_NODE.<br>.

#### [sparse_oblique_max_num_projections](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 6000 **Possible values:** min:1

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Maximum number of projections (applied after the num_projections_exponent).<br>Oblique splits try out max(p^num_projections_exponent, max_num_projections) random projections for choosing a split, where p is the number of numerical features. Increasing "max_num_projections" increases the training time but not the inference time. In late stage model development, if every bit of accuracy if important, increase this value.<br>The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020) does not define this hyperparameter.

#### [sparse_oblique_normalization](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** NONE **Possible values:** NONE,
    STANDARD_DEVIATION, MIN_MAX

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before applying the sparse oblique projections.<br>- `NONE`: No normalization.<br>- `STANDARD_DEVIATION`: Normalize the feature by the estimated standard deviation on the entire train dataset. Also known as Z-Score normalization.<br>- `MIN_MAX`: Normalize the feature by the range (i.e. max-min) estimated on the entire train dataset.

#### [sparse_oblique_num_projections_exponent](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 2 **Possible values:** min:0

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections to test at each node.<br>Increasing this value very likely improves the quality of the model, drastically increases the training time, and doe not impact the inference time.<br>Oblique splits try out max(p^num_projections_exponent, max_num_projections) random projections for choosing a split, where p is the number of numerical features. Therefore, increasing this `num_projections_exponent` and possibly `max_num_projections` may improve model quality, but will also significantly increase training time.<br>Note that the complexity of (classic) Random Forests is roughly proportional to `num_projections_exponent=0.5`, since it considers sqrt(num_features) for a split. The complexity of (classic) GBDT is roughly proportional to `num_projections_exponent=1`, since it considers all features for a split.<br>The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020) recommends values in [1/4, 2].

#### [sparse_oblique_projection_density_factor](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 2 **Possible values:** min:0

-   Density of the projections as an exponent of the number of features. Independently for each projection, each feature has a probability "projection_density_factor / num_features" to be considered in the projection.<br>The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020) calls this parameter `lambda` and recommends values in [1, 5].<br>Increasing this value increases training and inference time (on average). This value is best tuned for each dataset.

#### [sparse_oblique_weights](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** BINARY **Possible values:** BINARY,
    CONTINUOUS

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Possible values:<br>- `BINARY`: The oblique weights are sampled in {-1,1} (default).<br>- `CONTINUOUS`: The oblique weights are be sampled in [-1,1].

#### [split_axis](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** AXIS_ALIGNED **Possible values:**
    AXIS_ALIGNED, SPARSE_OBLIQUE, MHLD_OBLIQUE

-   What structure of split to consider for numerical features.<br>- `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This is the "classical" way to train a tree. Default value.<br>- `SPARSE_OBLIQUE`: Sparse oblique splits (i.e. random splits on a small number of features) from "Sparse Projection Oblique Random Forests", Tomita et al., 2020.<br>- `MHLD_OBLIQUE`: Multi-class Hellinger Linear Discriminant splits from "Classification Based on Multivariate Contrast Patterns", Canete-Sifuentes et al., 2029

#### [uplift_min_examples_in_treatment](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 5 **Possible values:** min:0

-   For uplift models only. Minimum number of examples per treatment in a node.

#### [uplift_split_score](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** KULLBACK_LEIBLER **Possible values:**
    KULLBACK_LEIBLER, KL, EUCLIDEAN_DISTANCE, ED, CHI_SQUARED, CS,
    CONSERVATIVE_EUCLIDEAN_DISTANCE, CED

-   For uplift models only. Splitter score i.e. score optimized by the splitters. The scores are introduced in "Decision trees for uplift modeling with single and multiple treatments", Rzepakowski et al. Notation: `p` probability / average value of the positive outcome, `q` probability / average value in the control group.<br>- `KULLBACK_LEIBLER` or `KL`: - p log (p/q)<br>- `EUCLIDEAN_DISTANCE` or `ED`: (p-q)^2<br>- `CHI_SQUARED` or `CS`: (p-q)^2/q<br>

#### [winner_take_all](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/random_forest/random_forest.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   Control how classification trees vote. If true, each tree votes for one
    class. If false, each tree vote for a distribution of classes.
    winner_take_all_inference=false is often preferable.

## CART

A CART (Classification and Regression Trees) a decision tree. The non-leaf nodes
contains conditions (also known as splits) while the leaf nodes contain
prediction values. The training dataset is divided in two parts. The first is
used to grow the tree while the second is used to prune the tree.

### Protobuffer training configuration

The hyper-parameter protobuffers are used with the C++ and CLI APIs.

-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto">learner/abstract_learner.proto</a>
-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/cart/cart.proto">learner/cart/cart.proto</a>
-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto">learner/decision_tree/decision_tree.proto</a>

### Hyper-parameters

#### [allow_na_conditions](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   If true, the tree training evaluates conditions of the type `X is NA` i.e.
    `X is missing`.

#### [categorical_algorithm](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** CART **Possible values:** CART, ONE_HOT,
    RANDOM

-   How to learn splits on categorical attributes.<br>- `CART`: CART algorithm. Find categorical splits of the form "value \in mask". The solution is exact for binary classification, regression and ranking. It is approximated for multi-class classification. This is a good first algorithm to use. In case of overfitting (very small dataset, large dictionary), the "random" algorithm is a good alternative.<br>- `ONE_HOT`: One-hot encoding. Find the optimal categorical split of the form "attribute == param". This method is similar (but more efficient) than converting converting each possible categorical value into a boolean feature. This method is available for comparison purpose and generally performs worse than other alternatives.<br>- `RANDOM`: Best splits among a set of random candidate. Find the a categorical split of the form "value \in mask" using a random search. This solution can be seen as an approximation of the CART algorithm. This method is a strong alternative to CART. This algorithm is inspired from section "5.1 Categorical Variables" of "Random Forest", 2001.

#### [categorical_set_split_greedy_sampling](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 0.1 **Possible values:** min:0 max:1

-   For categorical set splits e.g. texts. Probability for a categorical value
    to be a candidate for the positive set. The sampling is applied once per
    node (i.e. not at every step of the greedy optimization).

#### [categorical_set_split_max_num_items](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** -1 **Possible values:** min:-1

-   For categorical set splits e.g. texts. Maximum number of items (prior to the
    sampling). If more items are available, the least frequent items are
    ignored. Changing this value is similar to change the "max_vocab_count"
    before loading the dataset, with the following exception: With
    `max_vocab_count`, all the remaining items are grouped in a special
    Out-of-vocabulary item. With `max_num_items`, this is not the case.

#### [categorical_set_split_min_item_frequency](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 1 **Possible values:** min:1

-   For categorical set splits e.g. texts. Minimum number of occurrences of an
    item to be considered.

#### [growing_strategy](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** LOCAL **Possible values:** LOCAL,
    BEST_FIRST_GLOBAL

-   How to grow the tree.<br>- `LOCAL`: Each node is split independently of the other nodes. In other words, as long as a node satisfy the splits "constraints (e.g. maximum depth, minimum number of observations), the node will be split. This is the "classical" way to grow decision trees.<br>- `BEST_FIRST_GLOBAL`: The node with the best loss reduction among all the nodes of the tree is selected for splitting. This method is also called "best first" or "leaf-wise growth". See "Best-first decision tree learning", Shi and "Additive logistic regression : A statistical view of boosting", Friedman for more details.

#### [honest](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   In honest trees, different training examples are used to infer the structure
    and the leaf values. This regularization technique trades examples for bias
    estimates. It might increase or reduce the quality of the model. See
    "Generalized Random Forests", Athey et al. In this paper, Honest trees are
    trained with the Random Forest algorithm with a sampling without
    replacement.

#### [honest_fixed_separation](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   For honest trees only i.e. honest=true. If true, a new random separation is
    generated for each tree. If false, the same separation is used for all the
    trees (e.g., in Gradient Boosted Trees containing multiple trees).

#### [honest_ratio_leaf_examples](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 0.5 **Possible values:** min:0 max:1

-   For honest trees only i.e. honest=true. Ratio of examples used to set the
    leaf values.

#### [in_split_min_examples_check](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   Whether to check the `min_examples` constraint in the split search (i.e.
    splits leading to one child having less than `min_examples` examples are
    considered invalid) or before the split search (i.e. a node can be derived
    only if it contains more than `min_examples` examples). If false, there can
    be nodes with less than `min_examples` training examples.

#### [keep_non_leaf_label_distribution](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   Whether to keep the node value (i.e. the distribution of the labels of the
    training examples) of non-leaf nodes. This information is not used during
    serving, however it can be used for model interpretation as well as hyper
    parameter tuning. This can take lots of space, sometimes accounting for half
    of the model size.

#### [max_depth](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 16 **Possible values:** min:-1

-   Maximum depth of the tree. `max_depth=1` means that all trees will be roots.
    `max_depth=-1` means that tree depth is not restricted by this parameter.
    Values <= -2 will be ignored.

#### [max_num_nodes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 31 **Possible values:** min:-1

-   Maximum number of nodes in the tree. Set to -1 to disable this limit. Only
    available for `growing_strategy=BEST_FIRST_GLOBAL`.

#### [maximum_model_size_in_memory_in_bytes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Real **Default:** -1

-   Limit the size of the model when stored in ram. Different algorithms can
    enforce this limit differently. Note that when models are compiled into an
    inference, the size of the inference engine is generally much smaller than
    the original model.

#### [maximum_training_duration_seconds](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Real **Default:** -1

-   Maximum training duration of the model expressed in seconds. Each learning
    algorithm is free to use this parameter at it sees fit. Enabling maximum
    training duration makes the model training non-deterministic.

#### [mhld_oblique_max_num_attributes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 4 **Possible values:** min:1

-   For MHLD oblique splits i.e. `split_axis=MHLD_OBLIQUE`. Maximum number of
    attributes in the projection. Increasing this value increases the training
    time. Decreasing this value acts as a regularization. The value should be in
    [2, num_numerical_features]. If the value is above the total number of
    numerical features, the value is capped automatically. The value 1 is
    allowed but results in ordinary (non-oblique) splits.

#### [mhld_oblique_sample_attributes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   For MHLD oblique splits i.e. `split_axis=MHLD_OBLIQUE`. If true, applies the
    attribute sampling controlled by the "num_candidate_attributes" or
    "num_candidate_attributes_ratio" parameters. If false, all the attributes
    are tested.

#### [min_examples](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 5 **Possible values:** min:1

-   Minimum number of examples in a node.

#### [missing_value_policy](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** GLOBAL_IMPUTATION **Possible values:**
    GLOBAL_IMPUTATION, LOCAL_IMPUTATION, RANDOM_LOCAL_IMPUTATION

-   Method used to handle missing attribute values.<br>- `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean (in case of numerical attribute) or the most-frequent-item (in case of categorical attribute) computed on the entire dataset (i.e. the information contained in the data spec).<br>- `LOCAL_IMPUTATION`: Missing attribute values are imputed with the mean (numerical attribute) or most-frequent-item (in the case of categorical attribute) evaluated on the training examples in the current node.<br>- `RANDOM_LOCAL_IMPUTATION`: Missing attribute values are imputed from randomly sampled values from the training examples in the current node. This method was proposed by Clinic et al. in "Random Survival Forests" (https://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908043).

#### [num_candidate_attributes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** -1 **Possible values:** min:-1

-   Number of unique valid attributes tested for each node. An attribute is
    valid if it has at least a valid split. If `num_candidate_attributes=0`, the
    value is set to the classical default value for Random Forest: `sqrt(number
    of input attributes)` in case of classification and
    `number_of_input_attributes / 3` in case of regression. If
    `num_candidate_attributes=-1`, all the attributes are tested.

#### [num_candidate_attributes_ratio](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** -1 **Possible values:** min:-1 max:1

-   Ratio of attributes tested at each node. If set, it is equivalent to
    `num_candidate_attributes = number_of_input_features x
    num_candidate_attributes_ratio`. The possible values are between ]0, and 1]
    as well as -1. If not set or equal to -1, the `num_candidate_attributes` is
    used.

#### [pure_serving_model](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   Clear the model from any information that is not required for model serving.
    This includes debugging, model interpretation and other meta-data. The size
    of the serialized model can be reduced significatively (50% model size
    reduction is common). This parameter has no impact on the quality, serving
    speed or RAM usage of model serving.

#### [random_seed](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Integer **Default:** 123456

-   Random seed for the training of the model. Learners are expected to be
    deterministic by the random seed.

#### [sorting_strategy](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** IN_NODE **Possible values:** IN_NODE,
    PRESORT, FORCE_PRESORT, AUTO

-   How are sorted the numerical features in order to find the splits<br>- AUTO: Selects the most efficient method among IN_NODE, FORCE_PRESORT, and LAYER.<br>- IN_NODE: The features are sorted just before being used in the node. This solution is slow but consumes little amount of memory.<br>- FORCE_PRESORT: The features are pre-sorted at the start of the training. This solution is faster but consumes much more memory than IN_NODE.<br>- PRESORT: Automatically choose between FORCE_PRESORT and IN_NODE.<br>.

#### [sparse_oblique_max_num_projections](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 6000 **Possible values:** min:1

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Maximum number of projections (applied after the num_projections_exponent).<br>Oblique splits try out max(p^num_projections_exponent, max_num_projections) random projections for choosing a split, where p is the number of numerical features. Increasing "max_num_projections" increases the training time but not the inference time. In late stage model development, if every bit of accuracy if important, increase this value.<br>The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020) does not define this hyperparameter.

#### [sparse_oblique_normalization](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** NONE **Possible values:** NONE,
    STANDARD_DEVIATION, MIN_MAX

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before applying the sparse oblique projections.<br>- `NONE`: No normalization.<br>- `STANDARD_DEVIATION`: Normalize the feature by the estimated standard deviation on the entire train dataset. Also known as Z-Score normalization.<br>- `MIN_MAX`: Normalize the feature by the range (i.e. max-min) estimated on the entire train dataset.

#### [sparse_oblique_num_projections_exponent](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 2 **Possible values:** min:0

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Controls of the number of random projections to test at each node.<br>Increasing this value very likely improves the quality of the model, drastically increases the training time, and doe not impact the inference time.<br>Oblique splits try out max(p^num_projections_exponent, max_num_projections) random projections for choosing a split, where p is the number of numerical features. Therefore, increasing this `num_projections_exponent` and possibly `max_num_projections` may improve model quality, but will also significantly increase training time.<br>Note that the complexity of (classic) Random Forests is roughly proportional to `num_projections_exponent=0.5`, since it considers sqrt(num_features) for a split. The complexity of (classic) GBDT is roughly proportional to `num_projections_exponent=1`, since it considers all features for a split.<br>The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020) recommends values in [1/4, 2].

#### [sparse_oblique_projection_density_factor](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 2 **Possible values:** min:0

-   Density of the projections as an exponent of the number of features. Independently for each projection, each feature has a probability "projection_density_factor / num_features" to be considered in the projection.<br>The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020) calls this parameter `lambda` and recommends values in [1, 5].<br>Increasing this value increases training and inference time (on average). This value is best tuned for each dataset.

#### [sparse_oblique_weights](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** BINARY **Possible values:** BINARY,
    CONTINUOUS

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Possible values:<br>- `BINARY`: The oblique weights are sampled in {-1,1} (default).<br>- `CONTINUOUS`: The oblique weights are be sampled in [-1,1].

#### [split_axis](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** AXIS_ALIGNED **Possible values:**
    AXIS_ALIGNED, SPARSE_OBLIQUE, MHLD_OBLIQUE

-   What structure of split to consider for numerical features.<br>- `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This is the "classical" way to train a tree. Default value.<br>- `SPARSE_OBLIQUE`: Sparse oblique splits (i.e. random splits on a small number of features) from "Sparse Projection Oblique Random Forests", Tomita et al., 2020.<br>- `MHLD_OBLIQUE`: Multi-class Hellinger Linear Discriminant splits from "Classification Based on Multivariate Contrast Patterns", Canete-Sifuentes et al., 2029

#### [uplift_min_examples_in_treatment](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 5 **Possible values:** min:0

-   For uplift models only. Minimum number of examples per treatment in a node.

#### [uplift_split_score](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** KULLBACK_LEIBLER **Possible values:**
    KULLBACK_LEIBLER, KL, EUCLIDEAN_DISTANCE, ED, CHI_SQUARED, CS,
    CONSERVATIVE_EUCLIDEAN_DISTANCE, CED

-   For uplift models only. Splitter score i.e. score optimized by the splitters. The scores are introduced in "Decision trees for uplift modeling with single and multiple treatments", Rzepakowski et al. Notation: `p` probability / average value of the positive outcome, `q` probability / average value in the control group.<br>- `KULLBACK_LEIBLER` or `KL`: - p log (p/q)<br>- `EUCLIDEAN_DISTANCE` or `ED`: (p-q)^2<br>- `CHI_SQUARED` or `CS`: (p-q)^2/q<br>

#### [validation_ratio](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/cart/cart.proto)

-   **Type:** Real **Default:** 0.1 **Possible values:** min:0 max:1

-   Ratio of the training dataset used to create the validation dataset for
    pruning the tree. If set to 0, the entire dataset is used for training, and
    the tree is not pruned.

## DISTRIBUTED_GRADIENT_BOOSTED_TREES

Exact distributed version of the Gradient Boosted Tree learning algorithm. See
the documentation of the non-distributed Gradient Boosted Tree learning
algorithm for an introduction to GBTs.

### Protobuffer training configuration

The hyper-parameter protobuffers are used with the C++ and CLI APIs.

-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto">learner/abstract_learner.proto</a>
-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto">learner/decision_tree/decision_tree.proto</a>
-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.proto">learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.proto</a>
-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto">learner/gradient_boosted_trees/gradient_boosted_trees.proto</a>

### Hyper-parameters

#### [apply_link_function](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   If true, applies the link function (a.k.a. activation function), if any, before returning the model prediction. If false, returns the pre-link function model output.<br>For example, in the case of binary classification, the pre-link function output is a logic while the post-link function is a probability.

#### [force_numerical_discretization](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   If false, only the numerical column safisfying
    "max_unique_values_for_discretized_numerical" will be discretized. If true,
    all the numerical columns will be discretized. Columns with more than
    "max_unique_values_for_discretized_numerical" unique values will be
    approximated with "max_unique_values_for_discretized_numerical" bins. This
    parameter will impact the model training.

#### [max_depth](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 6 **Possible values:** min:-1

-   Maximum depth of the tree. `max_depth=1` means that all trees will be roots.
    `max_depth=-1` means that tree depth is not restricted by this parameter.
    Values <= -2 will be ignored.

#### [max_unique_values_for_discretized_numerical](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.proto)

-   **Type:** Integer **Default:** 16000 **Possible values:** min:1

-   Maximum number of unique value of a numerical feature to allow its
    pre-discretization. In case of large datasets, discretized numerical
    features with a small number of unique values are more efficient to learn
    than classical / non-discretized numerical features. This parameter does not
    impact the final model. However, it can speed-up or slown the training.

#### [maximum_model_size_in_memory_in_bytes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Real **Default:** -1

-   Limit the size of the model when stored in ram. Different algorithms can
    enforce this limit differently. Note that when models are compiled into an
    inference, the size of the inference engine is generally much smaller than
    the original model.

#### [maximum_training_duration_seconds](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Real **Default:** -1

-   Maximum training duration of the model expressed in seconds. Each learning
    algorithm is free to use this parameter at it sees fit. Enabling maximum
    training duration makes the model training non-deterministic.

#### [min_examples](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 5 **Possible values:** min:1

-   Minimum number of examples in a node.

#### [num_candidate_attributes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** -1 **Possible values:** min:-1

-   Number of unique valid attributes tested for each node. An attribute is
    valid if it has at least a valid split. If `num_candidate_attributes=0`, the
    value is set to the classical default value for Random Forest: `sqrt(number
    of input attributes)` in case of classification and
    `number_of_input_attributes / 3` in case of regression. If
    `num_candidate_attributes=-1`, all the attributes are tested.

#### [num_candidate_attributes_ratio](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** -1 **Possible values:** min:-1 max:1

-   Ratio of attributes tested at each node. If set, it is equivalent to
    `num_candidate_attributes = number_of_input_features x
    num_candidate_attributes_ratio`. The possible values are between ]0, and 1]
    as well as -1. If not set or equal to -1, the `num_candidate_attributes` is
    used.

#### [num_trees](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Integer **Default:** 300 **Possible values:** min:0

-   Maximum number of decision trees. The effective number of trained tree can
    be smaller if early stopping is enabled.

#### [pure_serving_model](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   Clear the model from any information that is not required for model serving.
    This includes debugging, model interpretation and other meta-data. The size
    of the serialized model can be reduced significatively (50% model size
    reduction is common). This parameter has no impact on the quality, serving
    speed or RAM usage of model serving.

#### [random_seed](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Integer **Default:** 123456

-   Random seed for the training of the model. Learners are expected to be
    deterministic by the random seed.

#### [shrinkage](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Real **Default:** 0.1 **Possible values:** min:0 max:1

-   Coefficient applied to each tree prediction. A small value (0.02) tends to
    give more accurate results (assuming enough trees are trained), but results
    in larger models. Analogous to neural network learning rate. Fixed to 1.0
    for DART models.

#### [use_hessian_gain](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   Use true, uses a formulation of split gain with a hessian term i.e.
    optimizes the splits to minimize the variance of "gradient / hessian.
    Available for all losses except regression.

#### [worker_logs](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/distributed_gradient_boosted_trees.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   If true, workers will print training logs.

## ISOLATION_FOREST

An [Isolation Forest](https://ieeexplore.ieee.org/abstract/document/4781136) is
a collection of decision trees trained without labels and independently to
partition the feature space. The Isolation Forest prediction is an anomaly score
that indicates whether an example originates from a same distribution to the
training examples. We refer to Isolation Forest as both the original algorithm
by Liu et al. and its extensions.

### Protobuffer training configuration

The hyper-parameter protobuffers are used with the C++ and CLI APIs.

-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto">learner/abstract_learner.proto</a>
-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto">learner/decision_tree/decision_tree.proto</a>
-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/isolation_forest/isolation_forest.proto">learner/isolation_forest/isolation_forest.proto</a>

### Hyper-parameters

#### [allow_na_conditions](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   If true, the tree training evaluates conditions of the type `X is NA` i.e.
    `X is missing`.

#### [categorical_algorithm](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** CART **Possible values:** CART, ONE_HOT,
    RANDOM

-   How to learn splits on categorical attributes.<br>- `CART`: CART algorithm. Find categorical splits of the form "value \in mask". The solution is exact for binary classification, regression and ranking. It is approximated for multi-class classification. This is a good first algorithm to use. In case of overfitting (very small dataset, large dictionary), the "random" algorithm is a good alternative.<br>- `ONE_HOT`: One-hot encoding. Find the optimal categorical split of the form "attribute == param". This method is similar (but more efficient) than converting converting each possible categorical value into a boolean feature. This method is available for comparison purpose and generally performs worse than other alternatives.<br>- `RANDOM`: Best splits among a set of random candidate. Find the a categorical split of the form "value \in mask" using a random search. This solution can be seen as an approximation of the CART algorithm. This method is a strong alternative to CART. This algorithm is inspired from section "5.1 Categorical Variables" of "Random Forest", 2001.

#### [categorical_set_split_greedy_sampling](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 0.1 **Possible values:** min:0 max:1

-   For categorical set splits e.g. texts. Probability for a categorical value
    to be a candidate for the positive set. The sampling is applied once per
    node (i.e. not at every step of the greedy optimization).

#### [categorical_set_split_max_num_items](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** -1 **Possible values:** min:-1

-   For categorical set splits e.g. texts. Maximum number of items (prior to the
    sampling). If more items are available, the least frequent items are
    ignored. Changing this value is similar to change the "max_vocab_count"
    before loading the dataset, with the following exception: With
    `max_vocab_count`, all the remaining items are grouped in a special
    Out-of-vocabulary item. With `max_num_items`, this is not the case.

#### [categorical_set_split_min_item_frequency](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 1 **Possible values:** min:1

-   For categorical set splits e.g. texts. Minimum number of occurrences of an
    item to be considered.

#### [growing_strategy](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** LOCAL **Possible values:** LOCAL,
    BEST_FIRST_GLOBAL

-   How to grow the tree.<br>- `LOCAL`: Each node is split independently of the other nodes. In other words, as long as a node satisfy the splits "constraints (e.g. maximum depth, minimum number of observations), the node will be split. This is the "classical" way to grow decision trees.<br>- `BEST_FIRST_GLOBAL`: The node with the best loss reduction among all the nodes of the tree is selected for splitting. This method is also called "best first" or "leaf-wise growth". See "Best-first decision tree learning", Shi and "Additive logistic regression : A statistical view of boosting", Friedman for more details.

#### [honest](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   In honest trees, different training examples are used to infer the structure
    and the leaf values. This regularization technique trades examples for bias
    estimates. It might increase or reduce the quality of the model. See
    "Generalized Random Forests", Athey et al. In this paper, Honest trees are
    trained with the Random Forest algorithm with a sampling without
    replacement.

#### [honest_fixed_separation](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   For honest trees only i.e. honest=true. If true, a new random separation is
    generated for each tree. If false, the same separation is used for all the
    trees (e.g., in Gradient Boosted Trees containing multiple trees).

#### [honest_ratio_leaf_examples](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 0.5 **Possible values:** min:0 max:1

-   For honest trees only i.e. honest=true. Ratio of examples used to set the
    leaf values.

#### [in_split_min_examples_check](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   Whether to check the `min_examples` constraint in the split search (i.e.
    splits leading to one child having less than `min_examples` examples are
    considered invalid) or before the split search (i.e. a node can be derived
    only if it contains more than `min_examples` examples). If false, there can
    be nodes with less than `min_examples` training examples.

#### [keep_non_leaf_label_distribution](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** true **Possible values:** true, false

-   Whether to keep the node value (i.e. the distribution of the labels of the
    training examples) of non-leaf nodes. This information is not used during
    serving, however it can be used for model interpretation as well as hyper
    parameter tuning. This can take lots of space, sometimes accounting for half
    of the model size.

#### [max_depth](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** -2 **Possible values:** min:-2

-   Maximum depth of the tree. `max_depth=1` means that all trees will be roots.
    `max_depth=-1` means that tree depth unconstrained by this parameter.
    `max_depth=-2` means that the maximum depth is log2(number of sampled
    examples per tree) (default).

#### [max_num_nodes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 31 **Possible values:** min:-1

-   Maximum number of nodes in the tree. Set to -1 to disable this limit. Only
    available for `growing_strategy=BEST_FIRST_GLOBAL`.

#### [maximum_model_size_in_memory_in_bytes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Real **Default:** -1

-   Limit the size of the model when stored in ram. Different algorithms can
    enforce this limit differently. Note that when models are compiled into an
    inference, the size of the inference engine is generally much smaller than
    the original model.

#### [maximum_training_duration_seconds](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Real **Default:** -1

-   Maximum training duration of the model expressed in seconds. Each learning
    algorithm is free to use this parameter at it sees fit. Enabling maximum
    training duration makes the model training non-deterministic.

#### [mhld_oblique_max_num_attributes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 4 **Possible values:** min:1

-   For MHLD oblique splits i.e. `split_axis=MHLD_OBLIQUE`. Maximum number of
    attributes in the projection. Increasing this value increases the training
    time. Decreasing this value acts as a regularization. The value should be in
    [2, num_numerical_features]. If the value is above the total number of
    numerical features, the value is capped automatically. The value 1 is
    allowed but results in ordinary (non-oblique) splits.

#### [min_examples](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 5 **Possible values:** min:1

-   Minimum number of examples in a node.

#### [missing_value_policy](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** GLOBAL_IMPUTATION **Possible values:**
    GLOBAL_IMPUTATION, LOCAL_IMPUTATION, RANDOM_LOCAL_IMPUTATION

-   Method used to handle missing attribute values.<br>- `GLOBAL_IMPUTATION`: Missing attribute values are imputed, with the mean (in case of numerical attribute) or the most-frequent-item (in case of categorical attribute) computed on the entire dataset (i.e. the information contained in the data spec).<br>- `LOCAL_IMPUTATION`: Missing attribute values are imputed with the mean (numerical attribute) or most-frequent-item (in the case of categorical attribute) evaluated on the training examples in the current node.<br>- `RANDOM_LOCAL_IMPUTATION`: Missing attribute values are imputed from randomly sampled values from the training examples in the current node. This method was proposed by Clinic et al. in "Random Survival Forests" (https://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908043).

#### [num_candidate_attributes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 0 **Possible values:** min:-1

-   Number of unique valid attributes tested for each node. An attribute is
    valid if it has at least a valid split. If `num_candidate_attributes=0`, the
    value is set to the classical default value for Random Forest: `sqrt(number
    of input attributes)` in case of classification and
    `number_of_input_attributes / 3` in case of regression. If
    `num_candidate_attributes=-1`, all the attributes are tested.

#### [num_candidate_attributes_ratio](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** -1 **Possible values:** min:-1 max:1

-   Ratio of attributes tested at each node. If set, it is equivalent to
    `num_candidate_attributes = number_of_input_features x
    num_candidate_attributes_ratio`. The possible values are between ]0, and 1]
    as well as -1. If not set or equal to -1, the `num_candidate_attributes` is
    used.

#### [num_trees](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/isolation_forest/isolation_forest.proto)

-   **Type:** Integer **Default:** 300 **Possible values:** min:0

-   Number of individual decision trees. Increasing the number of trees can
    increase the quality of the model at the expense of size, training speed,
    and inference latency.

#### [pure_serving_model](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   Clear the model from any information that is not required for model serving.
    This includes debugging, model interpretation and other meta-data. The size
    of the serialized model can be reduced significatively (50% model size
    reduction is common). This parameter has no impact on the quality, serving
    speed or RAM usage of model serving.

#### [random_seed](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Integer **Default:** 123456

-   Random seed for the training of the model. Learners are expected to be
    deterministic by the random seed.

#### [sorting_strategy](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** AUTO **Possible values:** IN_NODE,
    PRESORT, FORCE_PRESORT, AUTO

-   How are sorted the numerical features in order to find the splits<br>- AUTO: Selects the most efficient method among IN_NODE, FORCE_PRESORT, and LAYER.<br>- IN_NODE: The features are sorted just before being used in the node. This solution is slow but consumes little amount of memory.<br>- FORCE_PRESORT: The features are pre-sorted at the start of the training. This solution is faster but consumes much more memory than IN_NODE.<br>- PRESORT: Automatically choose between FORCE_PRESORT and IN_NODE.<br>.

#### [sparse_oblique_normalization](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** NONE **Possible values:** NONE,
    STANDARD_DEVIATION, MIN_MAX

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Normalization applied on the features, before applying the sparse oblique projections.<br>- `NONE`: No normalization.<br>- `STANDARD_DEVIATION`: Normalize the feature by the estimated standard deviation on the entire train dataset. Also known as Z-Score normalization.<br>- `MIN_MAX`: Normalize the feature by the range (i.e. max-min) estimated on the entire train dataset.

#### [sparse_oblique_projection_density_factor](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Real **Default:** 2 **Possible values:** min:0

-   Density of the projections as an exponent of the number of features. Independently for each projection, each feature has a probability "projection_density_factor / num_features" to be considered in the projection.<br>The paper "Sparse Projection Oblique Random Forests" (Tomita et al, 2020) calls this parameter `lambda` and recommends values in [1, 5].<br>Increasing this value increases training and inference time (on average). This value is best tuned for each dataset.

#### [sparse_oblique_weights](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** BINARY **Possible values:** BINARY,
    CONTINUOUS

-   For sparse oblique splits i.e. `split_axis=SPARSE_OBLIQUE`. Possible values:<br>- `BINARY`: The oblique weights are sampled in {-1,1} (default).<br>- `CONTINUOUS`: The oblique weights are be sampled in [-1,1].

#### [split_axis](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** AXIS_ALIGNED **Possible values:**
    AXIS_ALIGNED, SPARSE_OBLIQUE

-   What structure of split to consider for numerical features.<br>- `AXIS_ALIGNED`: Axis aligned splits (i.e. one condition at a time). This is the "classical" way to train a tree. Default value.<br>- `SPARSE_OBLIQUE`: Sparse oblique splits (i.e. random splits on a small number of features) from "Sparse Projection Oblique Random Forests", Tomita et al., 2020. This includes the splits described in "Extended Isolation Forests" (Sahand Hariri et al., 2018).

#### [subsample_count](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/isolation_forest/isolation_forest.proto)

-   **Type:** Integer **Default:** 256 **Possible values:** min:0

-   Number of examples used to grow each tree. Only one of "subsample_ratio" and
    "subsample_count" can be set. By default, sample 256 examples per tree. Note
    that this parameter also restricts the tree's maximum depth to log2(examples
    used per tree) unless max_depth is set explicitly.

#### [subsample_ratio](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/isolation_forest/isolation_forest.proto)

-   **Type:** Real **Default:** 1 **Possible values:** min:0

-   Ratio of number of training examples used to grow each tree. Only one of
    "subsample_ratio" and "subsample_count" can be set. By default, sample 256
    examples per tree. Note that this parameter also restricts the tree's
    maximum depth to log2(examples used per tree) unless max_depth is set
    explicitly.

#### [uplift_min_examples_in_treatment](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Integer **Default:** 5 **Possible values:** min:0

-   For uplift models only. Minimum number of examples per treatment in a node.

#### [uplift_split_score](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto)

-   **Type:** Categorical **Default:** KULLBACK_LEIBLER **Possible values:**
    KULLBACK_LEIBLER, KL, EUCLIDEAN_DISTANCE, ED, CHI_SQUARED, CS,
    CONSERVATIVE_EUCLIDEAN_DISTANCE, CED

-   For uplift models only. Splitter score i.e. score optimized by the splitters. The scores are introduced in "Decision trees for uplift modeling with single and multiple treatments", Rzepakowski et al. Notation: `p` probability / average value of the positive outcome, `q` probability / average value in the control group.<br>- `KULLBACK_LEIBLER` or `KL`: - p log (p/q)<br>- `EUCLIDEAN_DISTANCE` or `ED`: (p-q)^2<br>- `CHI_SQUARED` or `CS`: (p-q)^2/q<br>

## HYPERPARAMETER_OPTIMIZER

### Protobuffer training configuration

The hyper-parameter protobuffers are used with the C++ and CLI APIs.

-   <a href="https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto">learner/abstract_learner.proto</a>

### Hyper-parameters

#### [maximum_model_size_in_memory_in_bytes](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Real **Default:** -1

-   Limit the size of the model when stored in ram. Different algorithms can
    enforce this limit differently. Note that when models are compiled into an
    inference, the size of the inference engine is generally much smaller than
    the original model.

#### [maximum_training_duration_seconds](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Real **Default:** -1

-   Maximum training duration of the model expressed in seconds. Each learning
    algorithm is free to use this parameter at it sees fit. Enabling maximum
    training duration makes the model training non-deterministic.

#### [pure_serving_model](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Categorical **Default:** false **Possible values:** true, false

-   Clear the model from any information that is not required for model serving.
    This includes debugging, model interpretation and other meta-data. The size
    of the serialized model can be reduced significatively (50% model size
    reduction is common). This parameter has no impact on the quality, serving
    speed or RAM usage of model serving.

#### [random_seed](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/learner/abstract_learner.proto)

-   **Type:** Integer **Default:** 123456

-   Random seed for the training of the model. Learners are expected to be
    deterministic by the random seed.
