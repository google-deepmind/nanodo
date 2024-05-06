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
"""Tests `../model.py`."""

# pylint: disable=invalid-name

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from nanodo import model
from optax import losses


jax.config.parse_flags_with_absl()
jax.config.update("jax_numpy_rank_promotion", "raise")


class ModelTest(chex.TestCase):

  @chex.variants(with_jit=True, without_jit=True)
  def test_full_model(self):
    B, L = (2, 128)
    cfg = model.DoConfig(D=16, H=4, L=L, N=4, V=256, F=4 * 4)

    k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
    x_BxL = jax.random.randint(k1, (B, L), 0, cfg.V, jnp.int32)
    m = model.TransformerDo(cfg)
    params = m.init(k2, x_BxL)
    y_BxLxV = self.variant(m.apply)(params, x_BxL)

    chex.assert_tree_all_finite(y_BxLxV)
    chex.assert_shape(y_BxLxV, (B, L, cfg.V))

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ["beginning", 0],
      ["near-beginning", 5],
      ["near-end", 200],
  )
  def test_causality(self, token_loc: int):
    """Tests model's prediction is causal.

    Ensures the TransformerDo block maintains causality for autoregressive
    modeling. Causality is asserted by ensuring the gradients of the learned
    positional encoding satisfies three criteria.

    If the loss at token offset X is being examined:
    1. The gradients, w.r.t. the loss of the positional encoding _before_ this
       token should be non-zero. This ensures we are looking at previous tokens
       and their positions to infer this token.
    2. The gradients at X should be non-zero: for next-token prediction, we
       slide our window one step at a time. Thus, the input at X is predicting
       the output at X.
    3. The gradients after X should be strictly zero: the output at X should not
       use information from the future to predict itself.

    The positional encoding, since it is directly learned, maintains
    independence cross-token which makes its gradients reliable proxies for
    changes made on the token level.

    Args:
      token_loc: location of the current 'token' under investigation.
    """
    cfg = model.DoConfig(D=16, H=4, L=256, N=4, V=512, F=16)
    m = model.TransformerDo(cfg)

    def loss(params, x: chex.Array, y: chex.Array) -> chex.Array:
      # Simple cross-entropy loss function. This loss function emulates the loss
      # used in nanodo/train.py.
      loss = losses.softmax_cross_entropy_with_integer_labels(
          m.apply(params, x),
          y,
      )
      # Jax only computes the derivatives of scalar-valued functions.
      return loss[0][token_loc]

    x_1xL = jnp.arange(cfg.L)[None, :]
    params = m.init(jax.random.PRNGKey(42), x_1xL)
    grads = self.variant(jax.grad(loss))(params, x=x_1xL, y=x_1xL + 1)
    # The learned positional embedding is token-wise independent. Taking the
    # gradient, with respect to the loss, gives us a proxy for perturbing
    # a single token.
    pos_grads = grads.get("params").get("pos_embed").get("embedding").value

    # Ensure the computation succeeded.
    chex.assert_tree_all_finite(pos_grads)

    # Before the current token, the gradient should be non-zero somewhere.
    # If the current token is zero, this is a noop.
    if token_loc:
      chex.assert_scalar_in(
          float(jnp.sum(jnp.square(pos_grads[0:token_loc]))),
          1e-2,
          1000,
      )

    # At the current token, the gradient should be non-zero somewhere.
    chex.assert_scalar_in(
        float(jnp.sum(jnp.square(pos_grads[token_loc]))),
        1e-2,
        1000,
    )

    # After the current token, the gradient of the loss is precisely zero
    # everywhere.
    after_token = pos_grads[token_loc + 1 :]
    chex.assert_trees_all_close(
        after_token,
        jnp.zeros_like(after_token),
        atol=1e-5,
    )

  def test_heads_divides_dimension(self):
    cfg = model.DoConfig(D=16, H=3, L=256, N=4, V=256, F=4 * 4)
    m = model.TransformerDo(cfg)

    x_BxL = jnp.ones((2, cfg.L), dtype=jnp.int32)
    with self.assertRaises(AssertionError):
      m.init(jax.random.PRNGKey(42), x_BxL)

  @chex.variants(with_jit=True, without_jit=True)
  def test_mlp(self):
    B = 3
    L = 4
    D = 16
    dtype = jnp.bfloat16
    cfg = model.DoConfig(D=D, H=4, L=L, N=4, V=256, F=4 * 4, dtype=dtype)
    m = model.Mlp(cfg)
    x_BxLxD = jnp.ones((B, L, D), dtype=dtype)
    params = m.init(jax.random.PRNGKey(42), x_BxLxD)
    out_BxLxD = self.variant(m.apply)(params, x_BxLxD)

    chex.assert_shape(out_BxLxD, (B, L, D))
    chex.assert_type(out_BxLxD, dtype)
    chex.assert_tree_all_finite(out_BxLxD)

  @chex.variants(with_jit=True, without_jit=True)
  def test_remat_forward(self):
    B, L = (2, 128)
    cfg_base = model.DoConfig(D=16, H=4, L=L, N=4, V=256, F=4 * 4, remat=False)
    cfg_remat = model.DoConfig(D=16, H=4, L=L, N=4, V=256, F=4 * 4, remat=True)

    k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
    x_BxL = jax.random.randint(k1, (B, L), 0, cfg_base.V, jnp.int32)
    m = model.TransformerDo(cfg_base)
    m_remat = model.TransformerDo(cfg_remat)
    params = m.init(k2, x_BxL)
    params_remat = m_remat.init(k2, x_BxL)

    chex.assert_trees_all_equal(params, params_remat)

    y_BxLxV = self.variant(m.apply)(params, x_BxL)
    y_remat_BxLxV = self.variant(m_remat.apply)(params_remat, x_BxL)

    chex.assert_tree_all_finite(y_BxLxV)
    chex.assert_shape(y_BxLxV, (B, L, cfg_base.V))
    error = jnp.linalg.norm(y_BxLxV - y_remat_BxLxV) / y_BxLxV.size
    self.assertLess(error, 1e-8)


if __name__ == "__main__":
  absltest.main()
