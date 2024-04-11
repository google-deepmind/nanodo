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
  elif layer_type in ["layer_norm", "rms_norm"]:  # [D,]
    return nn.initializers.ones
  elif layer_type == "rms_norm_indirect_scale":  # [D,]
    return nn.initializers.zeros
  else:
    raise ValueError(f"unrecognized layer type: {layer_type}")
