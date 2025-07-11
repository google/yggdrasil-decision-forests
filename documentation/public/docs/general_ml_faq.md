# General ML FAQ

This page compiles general ML questions often asked by YDF users.

## How many training examples are needed to train a model?

The number of training examples to train a good model depends on the complexity
of the patterns the model must learn. Therefore, the answer is inherently
problem-specific. A simple way to assess whether adding more training data would
be beneficial is as follows: Suppose you currently have N training examples and
M testing examples. Train two versions of your model:

-   Model A: Trained on the full dataset of N examples.
-   Model B: Trained on a slightly reduced dataset (e.g., N - 10% randomly
    selected examples).

Then evaluate both models using the same M test examples. If both models perform
similarly (i.e., removing 5% of the training data does not hurt the model), it
is likely that adding 5% more training examples will not significantly improve
performance because adding more training data generally has a diminishing
return.

However, if Model A (with more data) performs noticeably better than Model B,
this suggests that additional training data is likely to enhance your model's
quality.

## My model does not perform well. The AUC is only 0.7. What should I do?

**Background:** AUC (or AUC-ROC) stands for Area Under the Receiver Operating
Characteristic curve. It's a performance metric used for binary classifier
models. An AUC of 1 signifies perfect predictions, while an AUC of 0.5 indicates
a model that performs no better than random chance.

Before trying to improve your model, first determine if your current performance
is truly bad. An AUC of 0.7, for example, could be bad in some problems but
excellent in others. In applied machine learning, success often means being
better than an existing solution. Therefore, try to understand the AUC of the
status-quo.

While AUC is a good general metric, it often doesn't align with business value.
For production models, metrics like precision@recall or recall@precision are
generally more suitable as they directly relate to business impact (e.g.,
monetary gain/loss). Therefore, identify the metric that matters for your
product and use it to compare your model to the status-quo.

If your model needs improvement, start by adding more features and feature
engineering. Understanding which features are valuable (using variable
importance) and exploring new or differently preprocessed features is generally
the most impactful thing you can do. Next, tune your model's parameters. The
""How to Improve a Model"" page lists helpful hyperparameters, and automated
tuning can also be effective.

## How to train a model with user id features?

ID features (e.g., user ids) refers to a categorical value that lack
generalizable information by themself (e.g. "user_23452", "user_3465").

If each user ID is unique, it has no predictive power for new users. In this
case, replace the user ID feature with descriptive features about the user. For
example, instead of "user_123," use features like age, or preferences to allow
your model to generalize.

If user IDs appear a small number of times, you can still extract value by
grouping examples with the same ID. A simple way is to augment your training
data with label statistics related to each ID. For instance, replace each user
ID with the average (or other relevant statistic) of the label for that specific
ID. This lets the model leverage aggregated historical data for that ID.
