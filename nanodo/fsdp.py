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
"""Utils to assist with FSDP."""

# pylint: disable=g-importing-member

from typing import Any
from flax import linen as nn


DoConfig = Any  # model.DoConfig; model.py imports this module.


# For a tensor with dims (n1, n2, ..., nk) a partitioning must be specified of
# size (p1, p2, ..., pk).
# Here we partition over one dim only, so exactly one pi = "data" and the rest
# should be None. This means, partition the tensor on dim i over the "data" axis
# and not on the rest. Note that the "data" axis is the axis used for data
# parallel, and corresponds to the number of devices.
# The condition is that ni must be divisible by number of devices, so this
# partitioning therefore chooses the partitioning axis to be the model dim
# as this is usually divisible by number of devices.
def init(layer_type: str, docfg: DoConfig) -> nn.initializers.Initializer:
  """This function specifies the partitioning of various transformer layers."""
  partition_fn = nn.with_partitioning if docfg.fsdp_enabled else lambda x, y: x
  if layer_type == "embedding":  # [V, D]
    return partition_fn(docfg.embed_init, (None, "data"))
  elif layer_type == "attn_in_proj":  # [D, H, Dh]
    return partition_fn(docfg.kernel_init, ("data", None, None))
  elif layer_type == "attn_out_proj":  # [H, Dh, D]
    return partition_fn(docfg.kernel_init, (None, None, "data"))
  elif layer_type == "mlp_kernel":  # [D, F]
    return partition_fn(docfg.kernel_init, ("data", None))
  elif layer_type == "head":  # [D, V]
    if hasattr(docfg, "head_init"):
      return partition_fn(docfg.head_init, ("data", None))
    else:
      return partition_fn(docfg.kernel_init, ("data", None))
  else:
    raise ValueError(f"unrecognized layer type: {layer_type}")
