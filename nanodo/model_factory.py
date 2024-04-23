"""Factory for producing experimental models."""

# pylint: disable=invalid-name,g-import-not-at-top

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

  # default model and configs
  model_cls = model.TransformerDo
  model_cfg = model.DoConfig
  get_loss_fn = loss_lib.get_default_loss_fn

  cfg = model_cfg(**c.model, V=vocab_size)
  return model_cls(cfg), get_loss_fn
