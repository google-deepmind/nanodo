# NanoDO: A minimal ("nano-sized") Transformer decoder-only language model implementation in JAX.
Inspired by minGPT/nanoGPT and flax/examples we provide a minimal
implementation of a Transformer decoder-only language model in Jax.

The purpose is to be maximally hackable, forkable, and readable for researchers,
to enable highly exploratory research. Magic is great for products, but it is
harmful in many cases for research and so we minimize abstraction as a design
goal.

Currently we use:

*   [flax](https://github.com/google/flax) for modules
*   [optax](https://github.com/google-deepmind/optax) for optimization
*   [orbax](https://github.com/google/orbax) for checkpointing
*   [tfds](https://github.com/tensorflow/datasets) for data
*   [pygrain](https://github.com/google/grain) for data loading
*   [ConfigDict](https://github.com/google/ml_collections) for hyper-parameters.


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

## Setup (open-source, Linux/CPU)

```
python3.11 -m venv /tmp/nanodo_test_env
source /tmp/nanodo_test_env/bin/activate
cd [path_to_repo]
pip install -e .

# Run tests
pip install pytest pytest-xdist
PYTHONHASHSEED=0 pytest -n auto -rA

# Run training example:
python nanodo/main.py \
  --config=nanodo/configs/default.py \
  --config.workdir=/tmp/nanodo_workdir \
  --config.vocab_path=tests/testdata/sentencepiece_cc_all.32000.100extra-sentencepiece.model \
  --config.model.L=128 \
  --config.batch_size=2
```

Then point your [Tensorboard](https://github.com/tensorflow/tensorboard) to the workdir:

```
  tensorboard --logdir /tmp/nanodo_workdir
```

To use accelerators, ensure the appropriate JAX package is installed by following these [instructions](https://jax.readthedocs.io/en/latest/installation.html).

## Maintenance

 There are no guarantees that the software will be maintained going forward. The software is designed to be easily forked and modified.

## Citing NanoDO

To cite this repository:

```
@software{nanodo,
  author = {Peter J. Liu and Roman Novak and Jaehoon Lee and Mitchell Wortsman and Lechao Xiao and Katie Everett and Alexander A. Alemi and  Mark Kurzeja and Pierre Marcenac and Izzeddin Gur and Simon Kornblith and Kelvin Xu and Gamaleldin Elsayed and Ian Fischer and Jeffrey Pennington and Ben Adlam and Jascha-Sohl Dickstein},
  title = {NanoDO: A minimal Transformer decoder-only language model implementation in {JAX}.},
  url = {http://github.com/google-deepmind/nanodo},
  version = {0.1.0},
  year = {2024},
}
```


Authors all performed work while at Google Brain / DeepMind. We also thank Anselm Levskaya, and Gellért Weisz for code suggestions, and  Noah Fiedel for project support.

The first published paper to use (a fork of) the library was:

 [Wortsman et al. "Small-scale proxies for large-scale Transformer training instabilities." *ICLR 2024*.](https://openreview.net/forum?id=d8w0pmvXbZ)