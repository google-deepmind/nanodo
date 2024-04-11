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

* Tensors have short names similar to math and have shapes in their names
  following . No more shapes in comments. This violates the
  python style guide, but that was written for non-ML code.
* We avoid long docstrings and let code self-document when possible. In
  particular, type hints makes a lot of python documentation redundant.


Current model and training:

*   gelu activation function
*   learned position embedding
*   adamw optimizer
*   shared input and output embedding
*   Use both BOS and EOS , which works much better
*   No biases on layernorm or weight parameters, which PaLM found to improve
    stability and speed

Current parallelism:

We use Fully Sharded Data Parallel (FSDP) for parallelism. Model parameters
and the optimizer state are sharded among the devices. These shardings are
passed to jit, which is responsible for determining how to all-gather weights
when necessary.

## Usage
- See nanodo/run.sh for an example launch using XManager.
- Hyper-parameters are specified in a ConfigDict, with a default.py and other
 examples defined in configs/*.py.
- For easy, quick testing of your ideas try  or
.
  - With colab, one have full access to the model and the checkpoint, so you
  can inspect the model behavior interactively. For early training behavior,
  it may be simple to hack this colab to inspect model training.

## Differences with T5x (mindo) and flax/examples/lm1b
- Uses BOS in addition to EOS
- No sampling code or support for fast-decoding (so far)
- No embedding normalization by sqrt(D) (seems to be worse)

## Comparison with T5x (mindo)
- This code has no dependencies aside from reasonably standard ones,
 e.g. Flax, Optax, etc.
- No gin.
- Full control over data pipeline, and training loop.
- Code exists in small number of files.
- Less battle-tested

## Comparison with flax/examples/lm1b
- Simpler / shorter
- Uses newer orbax APIs for checkpointing
- Uses new global arrays for parallelism instead of pmap.
- Uses a simplified 'packing', sometimes called "noam packing",
 where examples are all concatenated together, separated only by EOS forming
  one giant sequence, from which examples are generated via sliding window.
  This is what was done in PaLM, but is worth investigating vs 'proper packing'.
- Lower eval loss, probably due to BOS addition

## Future features
- Flash attention support
- Tensor-parallelism
- Re-introduce support for dropout which will be helpful when fine-tuning

## Experimental versions and Forks
The `experimental/` subfolder has alternative model definitions for exploring other modeling choices while keeping the `model.py` as minimal as possible. The experimental model can be specified via the ConfigDict key `experimental_model` as in `configs/example_experimental_model.py`.  `model_factory.py` specifies the mapping to model definition, and is used in `train.py` to instantiate the flax model. If you only want to fork the default model, you can delete the `experimental/` folder and remove the `model_factory` dependency in `train.py`.

If you fork the code, it's useful for us to know so we can inform you of
upstream changes (e.g. bug fixes) and potentially fix your code for you (no guarantees):

[Forks](forks.md)

You can also join g/nanodo-users.