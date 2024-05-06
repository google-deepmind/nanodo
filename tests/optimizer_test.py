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
"""Tests for `../optimizer.py`."""

# pylint: disable=invalid-name,g-importing-member

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
from nanodo import model
from nanodo import optimizer


jax.config.parse_flags_with_absl()
jax.config.update("jax_numpy_rank_promotion", "raise")


def _get_test_opt_config() -> ml_collections.ConfigDict:
  c = ml_collections.config_dict.create(
      num_train_steps=10_000,
      peak_learning_rate=0.01,
      init_learning_rate=0.001,
      final_learning_rate=0.0001,
      warmup_steps=10,
      decay_steps=100,
      weight_decay=0.1,
  )
  return c


class OptimizerTest(parameterized.TestCase):

  @parameterized.parameters("cosine", "rsqrt")
  def test_create_lr(self, decay_type: str):
    c = _get_test_opt_config()
    c.decay_type = decay_type
    lr_fn = optimizer.get_learning_rate_schedule(c)

    self.assertGreater(lr_fn(0), 0)
    self.assertLess(jnp.abs(lr_fn(0) - c.init_learning_rate), 1e-9)
    self.assertGreater(lr_fn(1), lr_fn(0))
    self.assertEqual(lr_fn(c.warmup_steps), c.peak_learning_rate)
    if decay_type == "rsqrt":
      self.assertEqual(
          lr_fn(c.warmup_steps + 1),
          optimizer._rsqrt_schedule(
              init_value=lr_fn(c.warmup_steps), shift=1 + c.warmup_steps
          )(1),
      )
    else:
      self.assertEqual(lr_fn(c.num_train_steps), c.final_learning_rate)

  def test_create_lr_no_warmup(self):
    c = _get_test_opt_config()
    c.warmup_steps = 0
    lr_fn = optimizer.get_learning_rate_schedule(c)
    self.assertGreater(lr_fn(0), 0)
    self.assertLess(jnp.abs(lr_fn(0) - c.peak_learning_rate), 1e-9)
    self.assertGreater(lr_fn(1), 0)
    self.assertGreater(lr_fn(0), lr_fn(1))
    self.assertEqual(lr_fn(c.num_train_steps), c.final_learning_rate)

  def test_scale_by_dict(self):
    docfg = model.DoConfig(
        D=128, H=16, L=256, N=4, V=1024, F=4 * 4, fsdp_enabled=False)
    m = model.TransformerDo(docfg)
    init_rng = jax.random.PRNGKey(42)
    in_BxL = jnp.ones((2, 256), dtype=jnp.int32)
    initial_variables = jax.jit(m.init)(
        init_rng,
        in_BxL,
    )
    multiplier = 10
    residual = 1. - multiplier
    opt = optimizer._scale_by_dict({"kernel": multiplier})
    params = jax.tree_util.tree_map(jnp.zeros_like, initial_variables)
    grads = jax.tree_util.tree_map(jnp.ones_like, initial_variables)
    opt_state = opt.init(params)
    updates, _ = opt.update(grads, opt_state)
    delta = jax.tree_util.tree_map(
        lambda u, v: u - multiplier * v, updates, grads)

    def _assert_close(x, scalar=0.):
      error = jnp.linalg.norm(x - scalar) / x.size
      self.assertLess(error, 1e-8)

    for i in range(docfg.N):
      for name in ["key", "value", "query", "attn_out_proj"]:
        x = delta["params"][f"blocks_{i}"]["CausalAttn_0"][name]["kernel"]
        _assert_close(x)
      for name in ["Dense_0", "Dense_1"]:
        x = delta["params"][f"blocks_{i}"]["Mlp_0"][name]["kernel"]
        _assert_close(x)
      for name in ["LayerNorm_0", "LayerNorm_1"]:
        x = delta["params"][f"blocks_{i}"][name]["scale"]
        _assert_close(x, scalar=residual)
    _assert_close(delta["params"]["embed"]["embedding"], scalar=residual)
    _assert_close(delta["params"]["pos_embed"]["embedding"], scalar=residual)
    _assert_close(delta["params"]["out_ln"]["scale"], scalar=residual)


if __name__ == "__main__":
  absltest.main()
