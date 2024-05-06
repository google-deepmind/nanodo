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
"""Factory for producing experimental models."""

# pylint: disable=invalid-name,g-import-not-at-top,unused-import

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
  m = model
  get_loss_fn = loss_lib.get_default_loss_fn

  cfg = m.DoConfig(**c.model, V=vocab_size)  # pytype:disable=attribute-error
  module = m.TransformerDo(cfg)  # pytype:disable=attribute-error
  return module, get_loss_fn
