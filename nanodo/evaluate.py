"""Functions to evaluate nanodo runs."""

# pylint: disable=invalid-name,g-importing-member,g-import-not-at-top

import functools
import math
import os
from typing import Any, TYPE_CHECKING

from absl import logging
import jax
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from nanodo import data
from nanodo import metrics as metrics_lib
from optax import losses


if TYPE_CHECKING:
  from flax import linen as nn
  import ml_collections
  import tensorflow as tf


PyTree = Any


# Conversion factor to bits per Byte from nats per tokens.
_BPN = 1.0 / math.log(2)

# (tfds_name, vocab_path) -> bits per Bytes.
_TO_BPB = {
    (
        "lm1b:1.1.0",
        "cc_all.32000.100extra.bos.model",
    ): _BPN * (11_063_127 / 41_715_169.0),
    (
        "c4:3.1.0",
        "cc_all.32000.100extra.bos.model",
    ): _BPN * (184_537_826 / 789_615_977.0),
    (
        "huggingface:cerebras__slimpajama_627b",
        "cc_all.32000.100extra.bos.model",
    ): _BPN * (561_018_217 / 2_174_889_064.0),
}


class Evaluator:
  """Executes eval."""

  def __init__(
      self,
      c: "ml_collections.ConfigDict",
      model: "nn.Module",
      eval_ds: "tf.data.Dataset",
      mesh: Mesh,
      shardings: PyTree,
  ):
    self.step_fn = jax.jit(
        functools.partial(_eval_step, model=model, mesh=mesh),
        in_shardings=(
            shardings.params,
            NamedSharding(mesh, P()),
        ),
        out_shardings=(NamedSharding(mesh, P())),
        donate_argnames=("params", "in_BxL"),
    )
    self.c = c
    self.ds = eval_ds
    # Conversion factor to bits per Byte from nats per tokens.
    self.bpB = _TO_BPB.get((c.ds_name, os.path.basename(c.vocab_path)), None)

  def eval(self, params: PyTree) -> dict[str, float]:
    """Run eval with at most one epoch."""
    metrics = metrics_lib.Average()
    i = 0
    for i, batch in enumerate(self.ds.as_numpy_iterator()):
      step_metrics = jax.device_get(self.step_fn(params, batch))
      metrics = metrics.merge(step_metrics)
      if i == self.c.eval_steps:
        logging.info("Ended eval at step %d (batch size %d)", i, batch.shape[0])
        break
    if i < self.c.eval_steps:
      logging.warning("Ran out of data at step %d. Stopping.", i)

    output = {
        "loss": metrics.mean,
        "loss_std": metrics.sem,
        "loss_uc": metrics.mean + 3 * metrics.sem,
    }
    if self.bpB:
      output |= {
          "loss_bpB": output["loss"] * self.bpB,
          "loss_std_bpB": output["loss_std"] * self.bpB,
          "loss_uc_bpB": output["loss_uc"] * self.bpB,
      }

    output = {"eval_" + k: v for k, v in output.items()}
    # Dummy scalar to show high up in XM measurements.
    output["_eval_loss"] = output["eval_loss"]
    return output


def _eval_step(
    params: PyTree,
    in_BxL: jax.Array,
    model: "nn.Module",
    mesh: Mesh | None = None,
) -> metrics_lib.Average:
  """Return evaluation metrics on a single batch of data."""
  if mesh is not None:
    in_BxL = jax.lax.with_sharding_constraint(
        in_BxL, NamedSharding(mesh, P("data"))
    )
  x_BxL, y_BxL, weights_BxL = data.get_in_out(in_BxL)
  logits_BxLxV = model.apply({"params": params}, x_BxL)
  return _compute_unnormed_metrics(logits_BxLxV, y_BxL, weights_BxL)


def _compute_unnormed_metrics(
    logits_BxLxV: jax.Array,
    labels_BxL: jax.Array,
    weights_BxL: jax.Array,
) -> metrics_lib.Average:
  """Compute unnormalized summary metrics."""
  losses_BxL = losses.softmax_cross_entropy_with_integer_labels(
      logits_BxLxV, labels_BxL
  )
  return metrics_lib.Average.from_array(losses_BxL, mask=weights_BxL)
