# Why decision forests?

Decision forests are a type of machine learning model for supervised learning
(classification, regression, ranking, and uplifting).

## Benefits

Decision forests have a number of strong benefits:

-   They have few hyperparameters, and those parameters have good default values
    providing good results. This means decision forests are **easy to train**.
-   They natively handle numeric features, categorical features, missing
    features and text. This means they require little to no data preprocessing,
     **saving time and reducing sources for error**.
-   They generally give good results out of the box and are robust to noisy
    data. This means decision forests is a **great first technique to try** when
    faced with a new machine learning problem.
-   They offer interpretable properties. This means you can **understand how**
    they make their predictions.
-   They infer and train very fast. This means you can run them on **small
    devices** or at a **very large scale with little costs**.

``` {note}
Decision forests are used in [60%](https://keras.io/why_keras) of the top-5 solutions in Kaggle competitions.
```

## Where they work well

Decision forests are especially suited for **tabular data** such as databases
or spreadsheets. In addition, decision forests are efficient for
**signal integration** (i.e., aggregating the signals of multiple subsystems;
possibly made with machine learning models).

## Where they don't work well

Decision forests are not well natively suited images, texts, time series data or
graph data. On such type of data, neural networks generally perform better.

For advanced users, training a decision forest **on top** of a pre-trained
neural networks is a gread path for combining the strengths of decision forests
and neural networks on non-tabular data. For more information check out 
[this tutorial on Google Colab](https://www.tensorflow.org/decision_forests/tutorials/intermediate_colab).

``` {note}
Learn more about decision forests algorithms in our
[Decision Forests class](https://developers.google.com/machine-learning/decision-forests)
on the Google Developer website. Understanding how decision forests work is
 **not** necessary for using this library, but helps gaining a deeper
understanding of the algorithm's results. Also, decision forests are fun!
```
