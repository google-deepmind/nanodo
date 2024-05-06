# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loss functions."""

# pylint: disable=invalid-name,g-import-not-at-top,g-bare-generic

from typing import Any, Callable, TYPE_CHECKING

from flax.struct import dataclass
import jax
import jax.numpy as jnp
from nanodo import data
from optax import losses

if TYPE_CHECKING:
  import ml_collections


PyTree = Any


@dataclass
class LossAuxData:
  ntokens: jax.Array
  state: PyTree
  log_perplexity: jax.Array

# loss(params) function to be used in `jax.value_and_grad`.
LossFn = Callable[[PyTree], tuple[jax.Array, LossAuxData]]

LossFnFactory = Callable[
    [jax.Array, Callable, "ml_collections.ConfigDict"],
    LossFn,
]


def get_default_loss_fn(
    in_BxL: jax.Array,
    apply_fn: Callable,
    c: "ml_collections.ConfigDict",
) -> LossFn:
  """Standard next-token-prediction language modeling loss."""
  def loss_fn(params: PyTree) -> tuple[jax.Array, LossAuxData]:
    x_BxL, y_BxL, weights_BxL = data.get_in_out(in_BxL)

    mutable = (
        "intermediate_acts",) if c.get("log_internal_metrics", False) else ()
    logits_BxLxV, state = apply_fn(
        {"params": params},
        x_BxL,
        mutable=mutable,
    )

    losses_BxL = losses.softmax_cross_entropy_with_integer_labels(
        logits_BxLxV, y_BxL
    )
    ntokens = weights_BxL.sum()
    mean_loss = jnp.sum(losses_BxL * weights_BxL) / ntokens
    return mean_loss, LossAuxData(
        ntokens=ntokens, state=state, log_perplexity=mean_loss)

  return loss_fn
