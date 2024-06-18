# Datasets for unit testing

## Adult

Full name: Adult (also known as Census)

Url: https://archive.ics.uci.edu/ml/datasets/adult

Donors: Ronny Kohavi and Barry Becker

## DNA

Full name: Molecular Biology (Splice-junction Gene Sequences) Data Set

Url:
https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)

Donors: G. Towell, M. Noordewier, and J. Shavlik

## Iris

Full name: Iris

Url: https://archive.ics.uci.edu/ml/datasets/iris

Creator: R.A. Fisher

Donors: Michael Marshall

## SST Binary

Full name: Stanford Sentiment Treebank; Binary classification

Url: https://nlp.stanford.edu/sentiment/index.html

Authors: Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher
Manning, Andrew Ng and Christopher Potts

## Toy

Full name: Yggdrasil Toy

Url: *This location*

Author: Mathieu Guillame-Bert

CSV and TFRecord showing how Yggdrasil Decision Forests represents various types
of features. The dataset contains 5 examples with different types of features (
numerical, categorical as integer, categorical as string, categorical set, and
boolean). Contains some missing values.

## Yggdrasil Synthetic Ranking

Full name: Yggdrasil Synthetic Ranking

Url: *This location*

Author: Mathieu Guillame-Bert

Small ranking dataset containing 500 groups with 10 items each. The
label/relevance is in [0,5]. Contains numerical and categorical features.
Contains missing values. Does NOT contains categorical-set and multi-dimensional
features. The NDCG@5 of random predictions is ~0.50. A basic model without
tuning can reach an NDCG@5 of ~0.77.

Generated with the Yggdrasil Synthetic Dataset Generator with the command:

```
echo "ranking{} num_categorical_set: 0" > /tmp/synthetic_ranking_config.pbtxt
bazel run -c opt --copt=-mavx2 //third_party/yggdrasil_decision_forests/cli/utils:synthetic_dataset -- \
    --alsologtostderr \
    --options=/tmp/synthetic_ranking_config.pbtxt\
    --train=csv:/tmp/synthetic_ranking_train.csv \
    --test=csv:/tmp/synthetic_ranking_test.csv \
    --ratio_test=0.2
```

## Sim PTE

Full name: Simulations for Personalized Treatment Effects

Generated with the R's Uplift package:
https://cran.r-project.org/web/packages/uplift/uplift.pdf

Creator: Leo Guelman <leo.guelman@gmail.com>

Code:

```r
library(uplift)

set.seed(123)

train <- sim_pte(n = 1000, p = 20, rho = 0, sigma = sqrt(2), beta.den = 4)
test <- sim_pte(n = 2000, p = 20, rho = 0, sigma = sqrt(2), beta.den = 4)

train$treat <- ifelse(train$treat == 1, 2, 1)
test$treat <- ifelse(test$treat == 1, 2, 1)

train$y <- ifelse(train$y == 1, 2, 1)
test$y <- ifelse(test$y == 1, 2, 1)

train$ts = NULL
test$ts = NULL

write.csv(train,"yggdrasil_decision_forests/test_data/dataset/sim_pte_train.csv", row.names=F, quote=F)
write.csv(test,"yggdrasil_decision_forests/test_data/dataset/sim_pte_test.csv", row.names=F, quote=F)
```

## Gaussians

Generate two gaussians for anomaly detection similarly as:
https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html

```python
def gen_ds(n_samples: int = 120, n_outliers: int = 40, seed: int = 0):
    np.random.seed(seed)
    covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
    cluster_1 = 0.4 * np.random.randn(n_samples, 2) @ covariance + np.array(
        [2, 2]
    )
    cluster_2 = 0.3 * np.random.randn(n_samples, 2) + np.array([-2, -2])
    outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))
    features = np.concatenate([cluster_1, cluster_2, outliers])
    labels = np.concatenate([
        np.zeros((2 * n_samples), dtype=bool),
        np.ones((n_outliers), dtype=bool),
    ])
    return features, labels
```
