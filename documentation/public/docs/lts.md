# Long-time-support

## Inference and serving

-   Changes to serving-related code are guaranteed to be backward compatible.
-   Model inference is deterministic: the same example is guaranteed to yield
    the same prediction.
-   Learners and models are extensively tested, including integration testing on
    real datasets; and, there exists no execution path in the serving code that
    crashes as a result of an error; Instead, in case of failure (e.g.,
    malformed input example), the inference code returns a util::Status.

## Training

-   Hyper-parameters' semantics are never modified.
-   The default value of hyper-parameters is never modified.
-   The default value of a newly-introduced hyper-parameter is set in such a way
    that the hyper-parameter is effectively disabled.

## Quality Assurance

The following mechanisms will be put in place to ensure the quality of the
library:

-   Peer-reviewing.
-   Unit testing.
-   Training benchmarks with ranges of acceptable evaluation metrics.
-   Sanitizers.
