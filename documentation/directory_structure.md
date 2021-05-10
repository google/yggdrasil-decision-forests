# Directory Structure

The project is organised as follow:

```
├── examples: Collection of usage examples.
├── configure: Project configuration.
├── documentation: User and developer documentation.
├── third_party: Bazel configure for dependencies.
├── tools: Tools for the management of the project and code.
└── yggdrasil_decision_forests: The library
    ├── cli: Core command-line-interface binaries.
    │   └── utils: Non-utilities command-line-interface binaries.
    ├── configure: Building configurations.
    ├── dataset: Dataset related logic.
    ├── learner: Implementation of learning algorithms.
    │   ├── decision_tree: Utility function for the training of DTs.
    │   ├── gradient_boosted_trees: GBDT learning algorithm.
    │   └── random_forest: RF learning algorithm.
    ├── metric: Evaluation and metric logic.
    ├── model: Implementation of models i.e. the inference.
    │   ├── decision_tree: Utility function for the inference of DTs.
    │   ├── gradient_boosted_trees: GBDT model.
    │   └── random_forest: RF model.
    ├── serving: Logic for the efficient inference of model.
    │   └── decision_forest: Specialization of the inference for DTs.
    ├── test_data: Data used for unit testing.
    │   ├── dataset: Dataset samples.
    │   ├── model: Model samples.
    │   └── prediction: Prediction samples.
    └── utils: Various utilities.
```
