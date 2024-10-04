---
date: 2024-10-04
authors:
  - gbm
---

# How different machine learning models generalize?

Do you know how different machine learning models generalize? Maybe you know and
understand the theory behind models like decision forests, neural networks,
nearest neighbors, or SVMs. You might even have a good sense of which model
works best for certain problems. But do you know how different machine learning
models **see the world** and have you ever **seen** how they generalize
differently?

![Four robots looking at nature](../../non-github-assets/blog/1/intro.png)

<!-- more -->

To answer this question, I'll train different tabular ML models to reconstruct
images. My assumption is that those different models will make different types
of mistakes that can then be linked back to the model mathematical definition.
This is mostly a ludic experiment where I am hoping to get some interesting
pictures, and with some chance, improve my practical understanding of those
models.

This article is not a scientific article in that it lacks details and
meticulousness. Instead, it is fun piece.

To discuss this article, join me in this github discussion.

## Experiment setup

A numerical grayscale image can be represented as a 2D array $M$ of numerical
values, where each element $M[x,y]$ encodes the light intensity of pixel
$(x,y)$. Color images can be encoded similarly, using a separate 2D arrays for
each color channels (e.g., RGB or HSL).

An image can also be represented as a function $f(x,y) := M[x,y]$ defined on the
integer coordinates in the images e.g. $x \in [0, \mbox{width} )$. It is not possible
to directly evaluate this function in between pixels e.g. $f(0.5, 6.1)$. My goal
is to train a machine learning model to learn this function where the x,y
coordinates are the model input features, and the function output is the label.
This gives an interesting property: (Most) ML models consume floating point
values and are able to interpolate in between the training data points a.k.a. to
generalize.

In other words, by evaluating the learned model in between the pixels (e.g.,
$f(0.5, 6.1)$), I can increase the resolution of an image: A image of size
100x100 can be converted into a image of size 1000x1000. This also means that I
don't need all the pixels to train the image. An image with dead pixels or a
missing part can be reconstructed.

**Note #1:** Specialized up-sampling algorithms exist and will produce much better
results. My goal is not to reconstruct the images as well as possible, but
instead to see how tabular ML models fail at this task :)

**Note #2:** Using the model, it is possible to extrapolate outside of the image.
However, since most traditionally used ML models are exceptionally bad at
extrapolation, I will not look at this.

As for the models, I will use the following tabular data models:

- A **decision tree**(DT) a.k.a. CART.
- A **random forest** (RF): A ensemble of decision trees trained with some
    randomness.
- A **gradient boosted trees** (GBT): A set of decision trees trained sequentially
    to each predict the errors of the previous trees.
- An **extremely randomized trees** (ERT): A random forest where the splits (a.k.a
    conditions) are selected at random.
- A **random forest with oblique splits** (Oblique RF): A random forest allowed to
    learn splits looking at multiple attributes at the same time.
- A **k-nearest neighbor** (kNN): Find the k-nearest pixels in the training
    dataset, and return the average.
- A **support vector machine** (SVM): A model that used to be popular that uses
    the distance to a few anchor points as an input feature of a linear model.
- A **feed-forward multi-layer perceptron** (MLP) a.k.a. a neural network: A
    sequence of matrix multiplications interleaved with activation functions.

## Reconstructing a toy image

The first image is a black and white one with an ellipse, a square and a diamond inside of a square. The full image is on the left side, and the right side shows a zoomed-in section.

![Ground truth for the toy image](../../non-github-assets/blog/1/ground_truth.png)

This image contains horizontal and vertical lines which should be easy for decision tree to learn, as well as, diagonal lines and round shapes which should be hard for the trees. This is because with classical decision tree, each condition tests one attribute at a time.

Any sufficiently large model can remember and reconstruct a image if the entire image is used for training. Instead, I reduced the image resolution by 4x and randomly dropped 80% of the remaining pixels, and ask the model to re-generate the original image. This means the models only see around $1.25\% = 1/4^2 * (1-0.8)$ of the original image data. The training image is:

![training data for the toy image](../../non-github-assets/blog/1/toy_training_data.png)

Let's start with a simple [decision tree](https://developers.google.com/machine-learning/decision-forests/decision-trees) (DT) model. Can it reconstruct the image? Here are its predictions.

![decision tree predictions for the toy image](../../non-github-assets/blog/1/toy_dt.png)

**Note:** The models I'll train are able to output probabilities so the predictions could be a grayscale image. While it would make for better looking images, plotting grayscale predictions would make it harder to see the mistakes.

The image reconstructed by the decision tree looks terrible, but what are the problem exactly?

The square in the bottomleft is almost perfect but the ellipse on the top-left
and diamond on the bottom-right all have a "stair steps". A little bit of
background: Here, a decision tree is a binary tree of conditions where each
condition is of the form $x \geq \mbox{some number}$ or $y \geq \mbox{some
number}$. For example, the top of the decision tree can look like.

```raw
x >= 0.220
    ├─(pos)─ y >= 0.780
    |        ├─(pos)─ ...
    |        └─(neg)─ ...
    └─(neg)─ y >= 0.207
                ├─(pos)─ ...
                └─(neg)─ ...
```

Because each condition only check one of the coordinate at a time, this creates strong horizontal or vertical biases in the model generalization. When those lines connect, they form the stair steps we see. I can confirm this by assigning a random color to each leaves of the decision tree:

![decision tree nodes for the toy image](../../non-github-assets/blog/1/toy_dt_nodes.png)

In the plot above, uniform rectangle is a leaf. The bottom-left square is represented nicely by a single block, but many blocks are necessary for the edge of the ellipse and diamond.

In the prediction plot above, there are two littler "outgrowths" on the side of the ellipse edge. Those are the decision trees overfitting a.k.a hallucinating a pattern. Overfitting is a problem with decision trees and a reason they are not used often in the industry.

A decision tree is a simple model. Can a [Random Forest](https://developers.google.com/machine-learning/decision-forests/intro-to-decision-forests) do better? In a Random Forest, many (e.g. 1000) decision trees are averaged together. Those trees are trained with some noise so they are not all the same. Here is what the predictions of a random forest look like.

![random forest predictions for the toy image](../../non-github-assets/blog/1/toy_rf.png)

This is better. The border of the ellipse and diamond are smoother even though they have stair steps. Most notably, there are no longer overfitting outgrowths. This is a good illustration of Breiman saying: [Random Forests do not overfit, [...]](https://www.stat.berkeley.edu/%7Ebreiman/randomforest2001.pdf).

What about [gradient boosted trees](https://developers.google.com/machine-learning/decision-forests/intro-to-gbdt), the famous algorithm that powers XGBoost libraries? The prediction of a GBT looks at follow:

![gradient boosted trees for the toy image](../../non-github-assets/blog/1/toy_gbt.png)

The overall shapes of the ellipse and diamond are also looking better than for the decision tree, though the surface is not as smooth as for the Random Forest. There are also some outgrowth on the diamond. This is expected, since GBTs (unlike Random Forests) can overfit.

What about more exotic decision forest models?

[Extremely randomized trees](https://link.springer.com/article/10.1007/s10994-006-6226-1) are a modification to random forests algorithm to make decision boundaries smoother. The idea is simple: Instead of learning the tree condition using the data, they are selected randomly. Here are the predictions from extremely randomized trees:

![extremely randomized trees predictions for the toy image](../../non-github-assets/blog/1/toy_extrem_trees.png)

As expected, the ellipse and diamond borders are smoother than for random forest, but they are "wiggly" where stair steps used to be. Also, the square corners are now rounder.


[Oblique Random Forest](https://developers.google.com/machine-learning/decision-forests/conditions#axis-aligned_vs_oblique_conditions) are another type of exotic models where a single split can test multiple attributes, generally with a linear equation. This should remove the stair step. Here are the oblique random forest predictions:

![oblique random forests predictions for the toy image](../../non-github-assets/blog/1/toy_oblique_rf.png)

This is much better. The ellipse is smooth and both the diamond and square edges are straight. The square and diamond corners are slightly rounded, but less than for extremely randomized trees. Also, the edge of the ellipse contains flat sections.This make sense: Those are the linear splits.

**Note:** Oblique random forests are not just good in this toy example. In practice, they are very competitive.

Any decision forest model can be made oblique. Here are the predictions of an oblique gradient boosted trees and an oblique decision tree.

![oblique gradient boosted trees predictions for the toy image](../../non-github-assets/blog/1/toy_oblique_gbt.png)
![oblique decision tree predictions for the toy image](../../non-github-assets/blog/1/toy_oblique_dt.png)

The oblique gradient boosted trees is somehow similar to the oblique random forest, with maybe sharper corners and flatter sections around the ellipse.
On the other side, oblique decision trees have a lot of "sharp bit" 🔪!

So far I looked at decision trees and decision forest models. What about other types of models? First, let's plot the predictions of a k-nearest-neighbors (kNN) model. To make a prediction, a k-nearest-neighbors  looks a the k closest values and return the most common label.

![k-nearest-neighbors predictions for the toy image](../../non-github-assets/blog/1/toy_knn.png)

This is interesting. The k-nearest-neighbors  model does not show stair steps like decision forests models. Instead, the edges have a kind of texture and look almost organic. While beautiful, this is not great in practice: This texture is an artifact of the overfitting these models typically suffer from. Note that the external rectangle is also completely missing.

What about support vector machines (SVM)? There are the predictions:

![support vector machine predictions for the toy image](../../non-github-assets/blog/1/toy_svm.png)

While the k-nearest-neighbors was overfitting, the SVM is missing many of the details and the generated plot is super smooth. Note that an ellipse can be described as a linear inequality on the sum of distances to the eclipse focus points, which is exactly what an SVM model can express. If the square, diamond and border was removed from the image, the SVM model would perfectly predict it.

Finally, this list would not be complete without neural networks. To keep things simple, here are the predictions of a multilayer-perceptron with 3 hidden layers of size 200:

![neural network predictions for the toy image](../../non-github-assets/blog/1/toy_ann_1.png)

Here are the predictions of a multilayer-perceptron with 8 hidden layers of size 200:

![another neural network predictions for the toy image](../../non-github-assets/blog/1/toy_ann_2.png)

The predictions are smooth which is great for the ellipse, but bad for the square and diamond. Also, the surrounding rectangle is completely missing. More worrisome, the edge of the predictions look like it is always overshooting (square) or undershooting (diamond) the edge of the original shapes. This shown a specificity of neural networks: While a decision forest a or kNN model learn different part of the feature space independently, neural network models learn patterns globally patterns. This is often great for generalization of complex patterns  that repeat in several places (e.g., the neural network can learn the concept of a circle and reuse it in several places), but this makes the model more complex to train.

Training parameters of a neural network also have a large and hard to predict impact on the final model. For fun of it, there are the predictions of a MLP trained with a single hidden layer, and one trained with 20 hidden layers of size 20:

![another neural network predictions for the toy image](../../non-github-assets/blog/1/toy_ann_3.png)

![another neural network predictions for the toy image](../../non-github-assets/blog/1/toy_ann_4.png)

## Reconstruction of a real image

Now, I'll apply the same reconstruction pattern on a colored picture. I expect the mistakes to be harder to interpret, but I am hoping to get some cool pictures.

I'll use this image of a toy old hot air balloon. This image is great for multiple reasons: It contains round shapes (hard for the decision forests), it contains sharp straight edges (easy for decision forests), and it contains details.

![an image of a ballon](../../non-github-assets/blog/1/ballon_ground_truth.png)

Like before, I've reduce the image resolution drastically and masked some of the pixels. The training image has a resolution of 68x102 pixels (including the dead pixels):

![training dataset for the ballon image](../../non-github-assets/blog/1/ballon_train_data.png)

Starting simple, the predictions of a decision trees are:

![decision tree predictions for the ballon image](../../non-github-assets/blog/1/ballon_dt.png)

You might have expected a broken image but the model is able both to fill the dead pixels and reconstruct the image. However, the reconstructed image still look low resolution. Why?

Essentially the trained decision tree has grown one leaf for each of the pixel in the original training images. Since pixels are squares, they are easy for the decision tree to encode, and the model can easily remember the entire image. However, the model fails to interpolate between the pixels.

What about Random Forest predictions?

![random forest predictions for the ballon image](../../non-github-assets/blog/1/ballon_rf.png)

The random forest model is doing some form of interpolation / smoothing in between the individual pixels. If you look closely, the pixels in the original image are not uniform in the reconstructed image. While a random forest model cannot extrapolate (i.e., make predictions that are not seen in training), the reconstructed image contains new colors: Look at the green-ish at the base of the ballon. This can be explained by the independent interpolation in the different color channels. The model’s interpolation is not a valid interpolation in the color space, so it creates new colors.

The predictions of a gradient boosted trees are:

![gradient boosted tres predictions for the ballon image](../../non-github-assets/blog/1/ballon_gbt.png)

The reconstructed image is even smoother than for the Random Forest. This makes sense: While the decision tree and random forest can create one leaf node for each pixel, the GBT model cannot because each tree is limited (in this experiment, I limited the tree depth to 6). Instead, the GBT overlaps multiple shallow trees to encode groups of pixels. This has two effects: The interpolation in between the color channels are less synchronized leading to the creation of more green, and the model generates drawing artifacts: Look at the ballon strips extending on the brown background at the top.

What about oblique Random Forest?

![oblique random forest predictionsfor the ballon image](../../non-github-assets/blog/1/ballon_oblique_rf.png)

This is a pleasing image. There are a lot of smoothing and pixel interpolation, there are no drawing artifacts, and there is only a small amount of the green-ish color. The model is also able to reconstruct some of the net surrounding the ballon (look at the original image). Interestingly, there are still some horizontal and vertical patterns in the zoomed image. This is caused by the algorithm that learns the oblique splits: horizontal and vertical splits are still possible and the learning algorithm is biased toward them.

All in all, this feels like a color filter I would use to stylize images.

For reference, here is the output of the XBR algorithm. This algorithm is specialized to increase the resolution of very-low resolution images such as sprites of old video game consoles.

![ballon image upsampled using the xbr algorithm](../../non-github-assets/blog/1/ballon_xbr.png)

What about other machine learning models? The predictions of a k-nearest neighbors are:

![knn predictions for the ballon image](../../non-github-assets/blog/1/ballon_knn.png)

Same as before, kNN has a tendency to hallucinate texture.

And, for the final images, let's look at the predictions of neural networks. Since the image is much more complex, I'll give more parameters to the network. Here are the predictions for a few neural network configurations.

![another neural network predictions for the ballon image](../../non-github-assets/blog/1/ballon_ann_2.png)
![another neural network predictions for the ballon image](../../non-github-assets/blog/1/ballon_ann_4.png)

The model cannot reconstruct the image details. Generating those images also took significantly more time than other models tested before. I am sure that with a larger model and more fiddling with the parameters, I could make the neural network learn the image. However, this experiment is a toy illustration of the development complexity and training cost of neural network models.

## Conclusion

This was an interesting experiment, and I hope it gave you a new understanding
on how different model generalize. Something interesting about this approach is
its capacity to consume clouds of points (not just raster images) and
reconstruct 2D but also 3D images (possibly with animations). I think this would
make for a cool follow-up. And, if you experiment with this and get some cool
pictures / videos before, let me know :).
