# Benchmark Inference

The inference speed of a model depends on:

1.  The architecture of the model. Large models with many trees are slower than
    smaller models with few trees. Also, Gradient boosted tree models are
    generally much faster than Random Forest models. For example, a Gradient
    Boosted Trees model with 200 trees will be faster than a Random Forest with
    100 trees.

2.  The [serving API](serving_apis) you are using. See comment regarding the
    speed of each serving API.

3.  (In the case of the C++ API) How well you are using the serving API. For
    example, by reusing the same engine and allocated examples, or by making
    sure not to reindex the input features, in between inference calls.

4.  The speed of your computer. Faster computers run models faster.

The
[c++ benchmark inference](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/cli/benchmark_inference.cc)
and
[go benchmark inference](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/port/go/cli/benchmark_inference/benchmark_inference.go)
tools allow to measure the speed of a model using the C++/CLI and Go APIs,
respectively. For example, the following example measures the inference speed of
a model using the C++ API:

```shell
# Disable CPU power scaling
sudo apt install linux-cpupower
sudo cpupower frequency-set --governor performance

# Benchmark
./benchmark_inference \
  --model=/path/to/model \
  --dataset=csv:/path/to/dataset \
  --batch_size=100 \
  --warmup_runs=10 \
  --num_runs=50
```

The result of the benchmark looks as follows:

```
batch_size : 100  num_runs : 50
time/example(us)  time/batch(us)  method
----------------------------------------
            0.89              89  GradientBoostedTreesQuickScorerExtended [virtual interface]
          5.8475          584.75  GradientBoostedTreesGeneric [virtual interface]
          12.485          1248.5  Generic slow engine
----------------------------------------
```

In this report, we see that three different inference engines are compatible
with the model. The fastest engine called
`GradientBoostedTreesQuickScorerExtended` makes a prediction for one example in
0.89µs (microseconds).
