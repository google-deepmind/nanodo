# NanoDO: Ultra-minimal ("nano-sized") Transformer decoder-only language model.
Inspired by minGPT/nanoGPT and flax/examples we provide a minimal,
implementation of a Transformer decoder-only language model in Jax.

The purpose is to be maximally hackable, forkable, and readable for researchers,
to enable highly exploratory research. Magic is great for products, but it is
harmful in many cases for research and so we minimize abstraction as a design
goal.

Currently we use:

*   flax for modules
*   optax for optimization
*   orbax for checkpointing.
*   pygrain for checkpointing training data iterator
*   tf.data (soon to be deprecated) and TFDS for data
*   ConfigDict for hyper-parameters.

Not currently supported or left out for simplicity:

*   sharding
*   fast decoding via activation caching
*   label-smoothing


Design opinions:

* Tensors have short names similar to math and have shapes in their names.
 No more shapes in comments. This violates the
  python style guide, but that was written for non-ML code.
* We avoid long docstrings and let code self-document when possible. In
  particular, type hints makes a lot of python documentation redundant.


Current model and training:

*   gelu activation function
*   learned position embedding
*   adamw optimizer
*   shared input and output embedding
*   Use both BOS and EOS
*   No biases on layernorm or weight parameters, which PaLM found to improve
    stability and speed

Current parallelism:

We use Fully Sharded Data Parallel (FSDP) for parallelism. Model parameters
and the optimizer state are sharded among the devices. These shardings are
passed to jit, which is responsible for determining how to all-gather weights
when necessary.

 