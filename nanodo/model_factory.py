"""Factory for producing experimental models."""

# pylint: disable=invalid-name,g-import-not-at-top

import dataclasses
from typing import TYPE_CHECKING

from flax import linen as nn
from nanodo import loss as loss_lib
from nanodo import model

if TYPE_CHECKING:
  import ml_collections


def get_model_and_loss(
    c: "ml_collections.ConfigDict",
    vocab_size: int,
) -> tuple[nn.Module, loss_lib.LossFnFactory]:
  """Returns an instantiated (potentially experimental) model."""

  # Default model and loss.
  # Edit this function to return a different model/loss based on the config.
  model_cls = model.TransformerDo
  cfg = _get_default_model_config(c, vocab_size)
  get_loss_fn = loss_lib.get_default_loss_fn

  return model_cls(cfg), get_loss_fn


def _get_default_model_config(
    c: "ml_collections.ConfigDict",
    vocab_size: int,
) -> model.DoConfig:
  """Convert an input config to a `model.DoConfig`."""
  return model.DoConfig(
      D=c.model_dim,
      H=c.num_heads,
      L=c.context_length,
      N=c.num_layers,
      V=vocab_size,
      F=c.mlp_dim,
      dtype=c.dtype,
      remat=c.get("remat", False),
      fsdp_enabled=c.get("fsdp_enabled", True),
  )
