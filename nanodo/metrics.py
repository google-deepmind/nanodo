"""Computing metrics tracked during training and evaluation."""

# pylint: disable=invalid-name,g-importing-member,g-import-not-at-top

import dataclasses
from typing import Any, Callable, TYPE_CHECKING

from absl import logging
from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp
from nanodo import optimizer
import numpy as np


if TYPE_CHECKING:
  import ml_collections
  from nanodo import loss as loss_lib
  from flax.training.train_state import TrainState


PyTree = Any


def get_init_metrics(
    step_fn: Callable[["TrainState", jax.Array], "TrainState"],
    state: "TrainState",
    in_BxL: jax.Array,
) -> dict[str, float | int]:
  """Compute metrics only at init, as they are constant throughout training."""
  metrics = _get_costs(step_fn, state, in_BxL)

  n_params_all = _size(state.params)

  n_params_embedding = 0
  if "embed" in state.params:
    n_params_embedding = _size(state.params["embed"])

  if "pos_embed" in state.params:
    n_params_embedding += _size(state.params["pos_embed"])
  n_params_non_embedding = n_params_all - n_params_embedding

  metrics |= {
      "n_params/all": n_params_all,
      "n_params/embedding": n_params_embedding,
      "n_params/non_embedding": n_params_non_embedding,
  }
  metrics |= _counts_from_tree(state.params)

  if "head" in state.params:
    n_params_head = _size(state.params["head"])
    n_params_non_embedding_head = (
        n_params_all - n_params_embedding - n_params_head
    )
    metrics |= {
        "n_params/head": n_params_head,
        "n_params/non_embedding_head": n_params_non_embedding_head,
    }
  return metrics


def get_metrics(
    aux_data: "loss_lib.LossAuxData",
    c: "ml_collections.ConfigDict",
    loss: float,
    state: "TrainState",
    grads: PyTree,
    updates: PyTree,
) -> dict[str, float | jax.Array]:
  """Compute metrics tracked at every training step."""
  # Access final gradient through opt_state.acc_grad
  step = state.opt_state.gradient_step  # pytype: disable=attribute-error
  acc_grads = state.opt_state.acc_grads  # pytype: disable=attribute-error
  # Use Welford algorithm for numerically stable aggregation of mean.
  # TODO: Consider computing Welford var/std as accumulated stats.
  acc_grads = jax.tree.map(
      lambda acc_grads, grads: acc_grads
      + (grads - acc_grads) / (state.opt_state.mini_step + 1),  # pytype: disable=attribute-error
      acc_grads,
      grads,
  )

  lr = optimizer.get_learning_rate_schedule(c.opt)(step)
  # Normalized update scale (w/o global learning rate factor).
  updates = jax.tree.map(lambda x: x / (lr + 1e-20), updates)
  metrics = {
      "__train_loss": loss,  # dummy scalar to be first alphabetically in XM.
      "train_loss": loss,
      "log_perplexity": aux_data.log_perplexity,
      "train_ntokens": aux_data.ntokens,
      "learning_rate": jnp.array(lr),
      "train_fraction": step / c.opt.num_train_steps,
      "train_tokens_seen": aux_data.ntokens * step,

      **_global_stats_from_tree("grads/all/", acc_grads),
      **_global_stats_from_tree("params/all/", state.params),
      **_global_stats_from_tree("updates/all/", updates),
  }
  if c.get("log_internal_metrics", False):
    metrics |= {
        **_stats_from_state(aux_data.state),
        **_stats_from_tree("grads/", acc_grads),
        **_stats_from_tree("params/", state.params),
        **_stats_from_tree("updates/", updates),
    }
  return metrics


def aggregate_microbatch_metrics(
    microbatch_metrics: list[dict[str, int | float | jax.Array]]
) -> dict[str, int | float | jax.Array]:
  """Accumulate train metrics weighted by `train_ntokens`.

  Accumulates train metrics with micro-batching logic. The logic assumes the
  default metrics are averaging metrics. `train_ntokens` is the only summed
  metrics and metrics including norm-based metrics are correctly computed
  after actual updates.

  Args:
    microbatch_metrics: a list of metric dictionaries, one for each microbatch.

  Returns:
    a single metric dictionary for the entire batch.
  """
  def _is_non_accumulating_metric(k):
    return (
        k.startswith("grads/") or
        k.startswith("params/") or
        k.startswith("updates/")
    )

  # Accumulate
  metrics = {}
  for m in microbatch_metrics:
    train_ntokens = float(m["train_ntokens"])
    for k, v in m.items():
      multiplier = train_ntokens if k != "train_ntokens" else 1.0
      if _is_non_accumulating_metric(k):
        metrics[k] = v
      elif k in metrics:
        metrics[k] += multiplier * v
      else:
        metrics[k] = multiplier * v

  # Normalize
  train_ntokens = metrics["train_ntokens"]
  for k, v in metrics.items():
    if _is_non_accumulating_metric(k):
      continue
    elif k != "train_ntokens":
      metrics[k] = v / train_ntokens

  # Perplexity is exponential of average, so compute after accumulation.
  metrics["train_perplexity"] = jnp.minimum(
      jnp.exp(metrics["log_perplexity"]),
      1.0e4,
  )
  return metrics


def _stats_from_state(state: dict[str, dict[str, float]]) -> dict[str, float]:
  """Convert the intermediates returned by the model into dict."""
  stats = {}
  for k, v in state.items():
    stats |= _tree_to_dict(k + "/", v)
  return stats


def _stats_from_tree(prefix: str, g: PyTree) -> dict[str, float]:
  return _tree_to_dict(prefix, jax.tree.map(_get_stats, g))


def _global_stats_from_tree(prefix: str, g: PyTree) -> dict[str, float]:
  return _tree_to_dict(prefix, _get_stats(g))


def _welford_mean(g: PyTree) -> float:

  def step(mean_and_size, x):
    mean, size = mean_and_size
    new_size = size + x.size
    new_mean = mean * (size / new_size) + jnp.sum(x) / new_size
    return new_mean, new_size

  mean, _ = jax.tree.reduce(step, g, (0., 0))
  return mean


def _get_stats(g: PyTree) -> dict[str, float]:
  mean = _welford_mean(g)
  ms = _welford_mean(jax.tree.map(jnp.square, g))
  stats = {
      "rms": jnp.sqrt(ms),
      "std": jnp.sqrt(jnp.maximum(ms - mean**2, 0.)),
      "mean": mean,
  }
  stats: dict[str, float]
  return stats


def _counts_from_tree(g: PyTree) -> dict[str, int]:
  g = jax.tree.map(jnp.size, g)
  return _tree_to_dict("n_params/", g)


def _tree_to_dict(prefix: str, g: PyTree) -> dict[str, Any]:
  return {prefix + "_".join(z.key for z in k if hasattr(z, "key")): v
          for k, v in jax.tree_util.tree_leaves_with_path(g)}


def _get_costs(f, *args, **kwargs) -> dict[str, float]:
  """Compute FLOPS cost of evaluating `f(*args, **kwargs)`.

  WARNING: `flops_compiled` are returned as `-1` on GPU:
  https://github.com/google/jax/issues/16008, and in general are unreliable on
  CPU/GPU: http://b/202218145.

  Args:
    f: JITtable function.
    *args: args for `f`.
    **kwargs: kwargs for `f`.

  Returns:
    FLOPS cost of evaluating `f(*args, **kwargs)`.
  """
  e = jax.jit(f).lower(*args, **kwargs)
  cost_lowered = e.cost_analysis()

  try:
    cost_compiled = e.compile().cost_analysis()[0]
  except jax.interpreters.xla.xc.XlaRuntimeError as e:
    logging.exception(e)
    cost_compiled = {}

  costs = {}
  # Note that `bytes accessed_lowered` is very bloated since read-write
  # operations overlap a lot.
  for k in ["flops", "bytes accessed"]:
    costs[k + "_lowered"] = cost_lowered[k]
    if k in cost_compiled:
      costs[k + "_compiled"] = cost_compiled[k]

  return costs


def _size(g: PyTree) -> int:
  return jax.tree_util.tree_reduce(lambda x, y: x + jnp.size(y), g, 0)


# TODO: CLean up code below. Seems unnecessarily complex.
#
# A clu version of the welford metric.
#
# Computes a running mean and standard deviation for a set of measurements.
#
# For more details see:
#
#  https://www.johndcook.com/blog/standard_deviation/
#  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
#  Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1983). "Algorithms for
#  computing the sample variance: Analysis and recommendations" (PDF). The
#  American Statistician. 37 (3): 242–247. doi:10.1080/00031305.1983.10483115.
#  JSTOR 2683386. Archived (PDF) from the original on 9 October 2022.
#  Schubert, Erich; Gertz, Michael (9 July 2018). Numerically stable parallel
#  computation of (co-)variance. ACM. p. 10. doi:10.1145/3221269.3223036. ISBN
#  9781450365055. S2CID 49665540.
#
# In particular, what is implemented here is a version of the parallel algorithm
# from Chan et al.  This should be more numerically stable than the naive
# sum of squares minus square of sum method (which loses a lot of precision).
#
# As an example of the usage:
#
#     average = Average.empty()
#     for x in values:
#       update = Average.from_output_value(x)
#       average = average.merge(update)
#     print(average.compute())
#
# This will print the estimate of the mean with its standard error as a string,
# i.e. "1.994+/-0.031"
#
# `average.compute` itself returns a `Summary`, an object
# representing the `mean`, `mean_standard_error` (i.e. the error on the estimate
# of the mean), as well
# as the `variance` of all of the consumed numbers and the total `count` of the
# values consumed.
@flax.struct.dataclass
class Summary:
  """Statistical summary of the data."""

  mean: jnp.ndarray
  mean_standard_error: jnp.ndarray
  variance: jnp.ndarray
  count: jnp.ndarray

  def __str__(self):
    return f"{self.mean}+/-{self.mean_standard_error}"


@flax.struct.dataclass
class Average(clu_metrics.Metric):
  """Computes a running mean and standard deviation from a set of measurements.

  Assumes the resulting value is a scalar but will count all values
  fed in, so will average across all dimensions by default.
  """

  count: jax.Array
  mean: jax.Array
  m2: jax.Array

  @classmethod
  def empty(cls) -> "Average":
    return cls(
        count=jnp.array(0, jnp.int32),
        mean=jnp.array(0, jnp.float32),
        m2=jnp.array(0, jnp.float32),
    )

  @classmethod
  def from_model_output(
      cls, values=jnp.ndarray, mask: jnp.ndarray | None = None, **_
  ) -> "Average":
    if values.ndim == 0:
      values = values[None]
    if mask is None:
      mask = jnp.ones_like(values)
    # Leading dimensions of mask and values must match.
    if mask.shape[0] != values.shape[0]:
      raise ValueError(
          "Argument `mask` must have the same leading dimension as `values`. "
          f"Received mask of dimension {mask.shape} "
          f"and values of dimension {values.shape}."
      )
    # Broadcast mask to the same number of dimensions as values.
    if mask.ndim < values.ndim:
      mask = jnp.expand_dims(
          mask, axis=tuple(np.arange(mask.ndim, values.ndim))
      )
    mask = mask.astype(bool)
    # utils.check_param(mask, dtype=bool, ndim=values.ndim)
    count = jnp.where(
        mask,
        jnp.ones_like(values, dtype=jnp.int32),
        jnp.zeros_like(values, dtype=jnp.int32),
    ).sum()
    total = jnp.where(mask, values, jnp.zeros_like(values)).sum()
    mean = total / count
    delta2 = (values - mean) ** 2
    m2 = jnp.where(mask, delta2, jnp.zeros_like(delta2)).sum()
    return cls(count=count, mean=mean, m2=m2)

  def merge(self, other: "Average"):
    clu_metrics._assert_same_shape(  #  pylint:disable=protected-access
        self.count, other.count
    )

    count = other.count + self.count
    delta = other.mean - self.mean
    # TODO: in cases where na ~ nb >> 1, instead use
    # mean = (self.count * self.mean + other.count * other.mean) / count
    mean = self.mean + delta * other.count / count
    m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / count
    return type(self)(count=count, mean=mean, m2=m2)

  def summary(self) -> Summary:
    """Returns some summary statistics about the data."""
    variance = self.m2 / self.count
    return Summary(
        mean=self.mean,
        mean_standard_error=jnp.sqrt(variance / self.count),
        variance=variance,
        count=self.count,
    )

  def compute(self):
    """Returns the computed summary."""
    return dataclasses.asdict(self.summary())
