# Datasets for unit testing

## Adult

Full name: Adult (also known as Cencus)

Url: https://archive.ics.uci.edu/ml/datasets/adult

Donors: Ronny Kohavi and Barry Becker

## DNA

Full name: Molecular Biology (Splice-junction Gene Sequences) Data Set

Url: https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)

Donors: G. Towell, M. Noordewier, and J. Shavlik

## Iris

Full name: Iris

Url: https://archive.ics.uci.edu/ml/datasets/iris

Creator: R.A. Fisher

Donors: Michael Marshall

## SST Binary

Full name: Stanford Sentiment Treebank; Binary classification

Url: https://nlp.stanford.edu/sentiment/index.html

Authors: Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts

## Toy

Full name: Yggdrasil Toy

Url: *This location*

Author: Mathieu Guillame-Bert

CSV and TFRecord showing how Yggdrasil Decision Forests represents various types of features. The dataset contains 5 examples with different types of features (numerical, categorical as integer, categorical as string, categorical set, and boolean). Contains some missing values.

## Yggdrasil Synthetic Ranking

Full name: Yggdrasil Synthetic Ranking

Url: *This location*

Author: Mathieu Guillame-Bert

Small ranking dataset containing 500 groups with 10 items each. The label/relevance is in [0,5]. Contains numerical and categorical features. Contains missing values. Does NOT contains categorical-set and multi-dimensional features. The NDCG@5 of random predictions is ~0.50. A basic model without tuning can reach an NDCG@5 of ~0.77.

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
