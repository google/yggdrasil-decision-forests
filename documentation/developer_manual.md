# Developer Manual

## Table of Contents

<!--ts-->

*   [Developer Manual](#developer-manual)
    *   [Table of Contents](#table-of-contents)
    *   [Design principles](#design-principles)
    *   [Where to change the code](#where-to-change-the-code)
    *   [Integration with existing code](#integration-with-existing-code)
    *   [How to test the code](#how-to-test-the-code)
    *   [Models and Learners](#models-and-learners)

<!--te-->

## Design principles

Yggdrasil Decision Forests (YDF) follows the following design principles:

-   **Ease of use:** Modeling is possible with minimal amount of configuration.

    -   The API shields but does not hide the complexity. Default parameters and
        logics with reasonable default values are used abundantly.

    -   Default parameters and logics are visible to the user.

-   **Safety:** The code is safe to use and the API is not prone to user errors.

    -   To parameter are created in a way that minimize the risk of
        misunderstanding and misuse.

    -   The code is peer-reviewed and unit-tested.

    -   Changes are backward compatible with existing models and training
        configuration. Notably, new logics do not change the semantic of a
        training configuration.

    -   Both the training and inference code is deterministic, unless stated
        otherwise.

    -   Error messages are informative to the user and helpful in identifying
        the most likely causes of the errors.

-   **Extensibility:** The API is flexible-enough to facilitate a steady growth
    of its collection of the latest learning algorithms from the literature.

    -   **Extensibility** vs **code speed directory** vs **development cost** is
        prioritized in this order.

-   **Quality:** New algorithms are accurately evaluated.

    -   New learning algorithms are evaluated on multiple datasets. Metrics are
        documented: quality , training speed, inference speed, model size,
        training stability.

    -   Code is read many more times than it's written. Code should be written
        to facilitate understanding (comments where needed) of the reader,
        specially in light of the complexity of the algorithms involved.

## Where to change the code

The [directory structure document](directory_structure.md) summarizes the code
organization.

For example, a new learning algorithm will be located in `learner`, while a new
dataset format will be implemented in `dataset`.

Ideally, new logics should be implemented as early as possible in the dependency
chain to maximize re-usability. For example, a new *splitter* compatible with
all types of forests will be located in `learner/decision_tree` instead of
`learner/random_forest`.

## Integration with existing code

New or changes of algorithms are integrated in one of two ways:

1.  C++ registration mechanism (see `utils/registration.h`). For example, this
    method is used for dataset formats, learning algorithm, model
    implementation, serving code and computation distribution.

2.  One-of or enum in a protobuffer configuration. For example, this method is
    used for all hyper-parameters.

## How to test the code

New code should be tested on what they do :).

In addition to fine-grain testing, code involving new training algorithms should
be tested using the end-to-end testing utility `utils/test_utils.h`.

## Models and Learners

The training logic of a learner is contained in the
[C++ Abstract Learner](../yggdrasil_decision_forests/learner/abstract_learner.h)
class. A minimal learner should have a "Train" function returning a model (see
next section).

Optionally, learners can implement the following logics:

-   Distributed training
-   Export of training logs
-   Generic hyper-parameters and default search space for tuning

The
[CART learner](../yggdrasil_decision_forests/learner/cart)
is an example of an basic learner. The
[Gradient Boosted Tree learner](../yggdrasil_decision_forests/learner/gradient_boosted_trees)
is an example of an advanced learner.

A **model** is a class that extends the
[abstract model class](../yggdrasil_decision_forests/model/abstract_model.h).
A minimal model overrides the 1) name, 2) load/save, and 3) predict functions.

Optional, models can implements the following logics:

-   Text description
-   Feature importances
-   Internal validation (e.g. out-of-bag)
-   Optimized inference engines

A *model* can be stored on disk in a directory. The structure and content of the
directory depend on the model implementation (and framework; remember that a
simpleML model can wrap other framework model formats). A model is loaded in
memory before being applied, processed or analysed. A model is standalone.
