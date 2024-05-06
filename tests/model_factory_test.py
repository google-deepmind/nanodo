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
"""Test experimental models and model factory."""

# pylint: disable=invalid-name

from absl.testing import absltest
from absl.testing import parameterized
import chex
import flax
import jax
import jax.numpy as jnp
from nanodo import model as default_model
from nanodo import model_factory
from nanodo.configs import default


jax.config.parse_flags_with_absl()
jax.config.update('jax_numpy_rank_promotion', 'raise')


class ModelTest(parameterized.TestCase):

  def _default_output(self, rng):
    """Set up an example input, output, params and config."""
    B, L = (2, 128)
    # default model
    cfg = default_model.DoConfig(D=16, H=4, L=L, N=4, V=256, F=4 * 4)
    m = default_model.TransformerDo(cfg)
    rng, spl = jax.random.split(rng)
    x_BxL = jax.random.randint(
        rng, minval=0, maxval=cfg.V, dtype=jnp.int32, shape=(B, L)
    )
    params = m.init(spl, x_BxL)
    default_model_out = m.apply(params, x_BxL)

    c = default.get_config()
    c.model.D = cfg.D
    c.model.H = cfg.H
    c.model.L = cfg.L
    c.model.N = cfg.N
    c.V = cfg.V
    c.model.F = cfg.F
    c.model.dtype = 'float32'

    return default_model_out, params, x_BxL, c

  def test_default_model(self):
    rng = jax.random.PRNGKey(42)
    default_model_out, params, x_BxL, c = self._default_output(rng)
    m, _ = model_factory.get_model_and_loss(c, c.V)
    self.assertTrue(jnp.allclose(m.apply(params, x_BxL), default_model_out))


if __name__ == '__main__':
  absltest.main()
