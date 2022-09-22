# Features

This page lists all the features of YDF.

## Learning algorithms

-   CART
-   Random Forest
-   Gradient Boosted (Decision) Trees
-   Distributed Gradient Boosted (Decision) Trees

## Meta-learning algorithm

-   Automatic hyper-parameter optimizer

## Supported of problems

-   Classification (binary and multi-class)
-   Regression
-   Ranking
-   Uplift
-   Weighted

## Supported input features

-   Automatic feature type detection and dictionary building.
-   Numerical
-   Categorical
-   Boolean
-   Categorical-set
-   Missing

## Inference

-   VPred
-   QuickScorer Extended
-   Get leafs

## Model evaluation

-   Classification
    -   Accuracy
    -   AUC (Area under the curve) of the ROC curve
    -   AUC of the Precision-Recall curve
    -   ROC curve
    -   Precision-Recall curve
    -   Precision @ Recall
    -   Recall @ Precision
    -   Precision @ Volum
    -   Recall @ False positive rate
    -   False positive rate @ Recall
    -   Cross-entropy loss
-   Regression
    -   RMSE
    -   MSE
-   Ranking
    -   NDCG
    -   MRR
    -   Precision @ 1
-   Uplift
    -   AUUC
    -   Qini
    -   Cate Calibration
-   Confidence interval
    -   Bootstrapping
    -   Closed-form (for a subset of metrics)

## API

-   CLI
-   C++
-   Python / TensorFlow
-   Go
-   JavaScript

## Dataset format

-   CSV
-   TF.Example proto of in TFRecord container

## Model analysis

-   Variable importance
-   Decision tree plotting
-   Tree structure statistics
